#!/usr/bin/env python3
"""Do the contiguous-prefill flags still beat the DEFAULT path? (peak VRAM + fwd)

WHY THIS EXISTS
---------------
`CUDA_VRAM_PERF_FINDINGS_2026-07-17.md` recommends promoting
`DKV_CONTIGUOUS_PREFILL=1 DKV_CONTIG_UNROTATE=1` on the strength of an A100
table:

    2x contiguous  4.85s  peak 16.74 GB
    1x un-rotate   5.03s  peak 14.09 GB   <- "near dense, fast fwd kept"
    dense          6.07s  peak 13.27 GB

TWO REASONS THAT TABLE CANNOT DECIDE THE DEFAULT TODAY:

1. It never measured the DKV DEFAULT PATH.  Every row is either an opt-in
   contiguous arm or dense.  "-2.6 GB" is 1x measured against 2x -- both
   opt-in.  Against what actually ships, the delta is simply unknown.

2. THE BASELINE MOVED.  `DKV_SDPA_HISTORY` (fused history attention) became
   default-on in 1c293277 on 2026-08-13, three weeks AFTER that doc, and its
   own comment measures it as *strictly better* than CONTIGUOUS_PREFILL:
   "reaches 9.28 s but costs +0.75 GB because it keeps a persistent rotated
   K/V buffer of every token."  So the July winner may now be a regression.

This harness measures the arms that were never compared, on one box, one model,
one prompt, in one process each.

ARMS
    eager     SDPA_HISTORY=0                     the pre-2026-08-13 baseline
    sdpa      (nothing set)                      TODAY'S DEFAULT
    contig2x  CONTIGUOUS_PREFILL=1
    contig1x  CONTIGUOUS_PREFILL=1 + CONTIG_UNROTATE=1

Each arm runs in its OWN SUBPROCESS: `_SDPA_HISTORY` is bound at import time,
so flipping it in-process would silently measure the wrong thing.

WHAT IS REPORTED
    peak_mb   torch.cuda.max_memory_allocated() around prefill+decode.  This is
              the number that OOMs you, and it captures the rotated buffer even
              though that buffer is freed when decode begins (so an
              after-generate reading would MISS it).
    fwd_s     wall time of generate(max_new_tokens=N).  At >=8k this is
              prefill-dominated but is NOT a pure prefill number.
    text      greedy continuation.  The contiguous branch carries
              "UNVALIDATED on GPU -- A/B output_text vs the default path before
              use"; this prints the comparison so that is no longer true.

USAGE
    python colab/bench_prefill_arms_peak.py
    CTXS=8192 ARMS=sdpa,contig1x python colab/bench_prefill_arms_peak.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")

# arm -> env overrides.  Absent key = leave unset (the real default).
ARM_ENV = {
    "eager":    {"DKV_SDPA_HISTORY": "0"},
    "sdpa":     {},
    "contig2x": {"DKV_CONTIGUOUS_PREFILL": "1"},
    "contig1x": {"DKV_CONTIGUOUS_PREFILL": "1", "DKV_CONTIG_UNROTATE": "1"},
    # The rank-cap arms: what DKV_RSVD_MAX_RPROJ=32 actually BUYS.  The cap
    # is a COMPRESS-time lever (batched cuSOLVER cliff at 32x32), and
    # compression runs inside prefill, so fwd_s captures it.
    "rproj_off": {"DKV_RSVD_MAX_RPROJ": "0"},
    "rproj_32":  {"DKV_RSVD_MAX_RPROJ": "32"},
}


def run_arm(arm, ctx):
    import torch
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)
    os.chdir(ACTIVE)
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from niah_recall import build_prompt

    w = DKVHFWrapper(model_id=os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                     config={"quantization": None, "rank": 32, "block_size": 256,
                             "micro_block_size": 256, "preset": "mid"})
    w.ensure_loaded()
    torch.cuda.synchronize()
    weights_b = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    w.active_session = f"arm-{arm}-{ctx}"
    prompt = build_prompt(w.tokenizer, ctx, 0.5)
    n_new = int(os.environ.get("NEW_TOKENS", "8"))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = w.generate(prompt, max_new_tokens=n_new, temperature=0.0, top_p=1.0,
                     repetition_penalty=1.0,
                     query_text="What is the secret passcode? Repeat it exactly.")
    torch.cuda.synchronize()
    fwd_s = time.perf_counter() - t0
    peak_b = torch.cuda.max_memory_allocated()
    after_b = torch.cuda.memory_allocated()
    if isinstance(out, dict):
        text = out.get("text") or out.get("output_text") or str(out)
    else:
        text = str(out)
    print("JSON " + json.dumps({
        "arm": arm, "ctx": ctx,
        "peak_mb": peak_b / 1e6,
        "after_mb": after_b / 1e6,
        "weights_mb": weights_b / 1e6,
        "fwd_s": fwd_s,
        "text": text[-160:],
        "sha": __import__("hashlib").sha1(text.encode("utf-8","replace")).hexdigest()[:12],
    }), flush=True)


def main():
    if os.environ.get("_ARM"):
        return run_arm(os.environ["_ARM"], int(os.environ["_CTX"]))
    ctxs = [int(x) for x in os.environ.get("CTXS", "8192,32768").split(",")]
    arms = os.environ.get("ARMS", "eager,sdpa,contig2x,contig1x").split(",")
    res = []
    for ctx in ctxs:
        for arm in arms:
            env = dict(os.environ, _ARM=arm, _CTX=str(ctx),
                       DKV_RSVD_SEED="1234", DKV_SVD_SEED="1234",
                       PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            env.pop("DKV_CONTIGUOUS_PREFILL", None)
            env.pop("DKV_CONTIG_UNROTATE", None)
            env.pop("DKV_SDPA_HISTORY", None)
            env.pop("DKV_RSVD_MAX_RPROJ", None)
            env.update(ARM_ENV[arm])
            p = subprocess.run([sys.executable, os.path.abspath(__file__)],
                               cwd=REPO, env=env, capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("JSON ")]
            if not line:
                print(f"ctx={ctx} arm={arm}: FAILED\n{p.stdout[-2000:]}"
                      f"\n{p.stderr[-2000:]}", flush=True)
                continue
            r = json.loads(line[-1][5:])
            res.append(r)
            print(f"  ctx={ctx:>6} arm={arm:>9} peak={r['peak_mb']:9.1f} MB "
                  f"after={r['after_mb']:9.1f} MB  fwd={r['fwd_s']:6.2f} s",
                  flush=True)

    print("")
    for ctx in ctxs:
        rows = [r for r in res if r["ctx"] == ctx]
        base = next((r for r in rows if r["arm"] == "sdpa"), None) or (rows[0] if rows else None)
        if not base:
            continue
        print(f"=== ctx={ctx} — deltas vs TODAY'S DEFAULT (sdpa) ===")
        print(f"{'arm':>9} {'peak MB':>10} {'d peak':>9} {'fwd s':>7} {'d fwd':>8}")
        for r in rows:
            dp = r["peak_mb"] - base["peak_mb"]
            df = 100.0 * (r["fwd_s"] - base["fwd_s"]) / base["fwd_s"]
            print(f"{r['arm']:>9} {r['peak_mb']:10.1f} {dp:+9.1f} "
                  f"{r['fwd_s']:7.2f} {df:+7.1f}%")
        print("")
        print("  text (fidelity — contiguous arms must match the default):")
        for r in rows:
            same = "SAME" if r["sha"] == base["sha"] else "DIFF"
            print(f"    {r['arm']:>9} [{same}] {r['text'][:90]!r}")
        print("")
    if res:
        print(f"weights alone: {res[0]['weights_mb']:.0f} MB of every peak above.")


if __name__ == "__main__":
    main()
