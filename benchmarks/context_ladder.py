#!/usr/bin/env python3
"""How far does each model actually go on this card, and what does a 1k cost?

WHY A LADDER AND NOT AN ESTIMATE
--------------------------------
The context ceiling on a 12 GB card is not a property of the weights. At Q4 the
weights are ~5 GB for an 8B model; what actually binds is the sum of the KV
state and DKV's own pool budget policy, and that policy is a FRACTION OF FREE
VRAM, so it moves as the weights change size. A ceiling inherited from another
run, another model, or another preset is not evidence about this one. Every
number here is measured, and the ones that OOM are recorded as OOM rather than
quietly omitted -- an absent row and a failed row mean very different things
when you are deciding whether two models can be reported side by side.

SUBPROCESS PER POINT, ALWAYS
----------------------------
A CUDA OOM does not reliably leave the process usable: the allocator can be
left fragmented and later points then fail for reasons that have nothing to do
with their own size. Each (arm, context) point therefore runs in its own child
process and reports back through a JSON file. The parent survives every OOM and
the ladder keeps climbing, which is the only way to find a ceiling rather than
guess at one.

ON WINDOWS, EXCEEDING VRAM DOES NOT RAISE
-----------------------------------------
Under the WDDM driver model CUDA oversubscribes into host RAM instead of
failing, so a point past the card's capacity comes back `ok` with a peak
LARGER than the physical card and a wall time several times what the trend
predicts. Measured here on granite-4.2-8b, dense, Q4 (12.28 GB card):

    16,384 tok    9.42 GB     10.6 s
    24,576 tok   11.23 GB     65.7 s      <- already thrashing
    32,768 tok   13.04 GB    380.5 s      <- 13.04 GB on a 12.28 GB card

A ladder that trusts the absence of an exception would have recorded 128k as
a success and every latency number above ~24k as real. So each point is
compared against the physical card and flagged `spilled`; a spilled point is
a CEILING, not a pass, and the arm stops there. Any timing from a spilled
point is PCIe bandwidth, not compute, and must not be quoted.

Checkpointed, so a power cut costs the point in flight and nothing else.

USAGE
    python benchmarks/context_ladder.py --model ibm-granite/granite-4.2-8b \
        --arms dense dkv --preset mid --quant nf4 \
        --contexts 4096 8192 16384 32768 49152 65536 98304 131072

    python benchmarks/context_ladder.py --report paper/results/ladder/*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)
from checkpoint import ResumableJSONL                            # noqa: E402

DEFAULT_CONTEXTS = [4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072]

# A point whose allocation passes this fraction of the physical card is treated
# as having spilled to host memory. Not 1.0: the display/driver context holds
# several hundred MB that the process can never have, so the practical wall is
# below the nameplate figure.
SPILL_FRACTION = 0.94

# Seconds-per-1k-tokens growth against the previous rung that counts as the
# memory system taking over. Prefill cost per token drifts up gently with
# context (attention is quadratic in total work but chunked here); a factor
# this large in one step is paging, not arithmetic.
CLIFF_RATIO = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# Filler: an exact token count, from real text
# ─────────────────────────────────────────────────────────────────────────────

def build_filler(tok, n_tokens: int) -> str:
    """A prompt of EXACTLY n_tokens, built from natural English.

    Exactness matters on a ladder: the point of the run is to attribute memory
    to context length, and a builder that undershoots by 3% (as the needle
    prompt builder does) puts the x-axis slightly to the left of where the
    table says it is.
    """
    src = os.path.join(HERE, "berry_paper.txt")
    if os.path.exists(src):
        with open(src, encoding="utf-8", errors="ignore") as f:
            base = f.read()
    else:
        base = ("The quick brown fox jumps over the lazy dog. " * 200)
    if not base.strip():
        base = "The quick brown fox jumps over the lazy dog. " * 200

    ids = tok(base, add_special_tokens=False).input_ids
    if not ids:
        raise SystemExit("filler source tokenized to nothing")
    reps = (n_tokens // len(ids)) + 2
    ids = (ids * reps)[:n_tokens]
    text = tok.decode(ids, skip_special_tokens=True)
    # Decoding then re-encoding is not always length-preserving; converge.
    for _ in range(6):
        cur = len(tok(text, add_special_tokens=False).input_ids)
        if cur == n_tokens:
            break
        if cur > n_tokens:
            text = tok.decode(tok(text, add_special_tokens=False).input_ids[:n_tokens],
                              skip_special_tokens=True)
        else:
            text += " " + tok.decode(ids[:n_tokens - cur], skip_special_tokens=True)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Child: one (arm, context) point
# ─────────────────────────────────────────────────────────────────────────────

def run_point(args) -> None:
    import torch
    res: Dict[str, Any] = {"arm": args.arm, "ctx": args.ctx, "status": "unknown"}
    t_load = time.time()
    try:
        if args.arm == "dkv":
            os.environ.setdefault("DKV_RSVD_SEED", "1234")
            os.environ.setdefault("DKV_SVD_SEED", "1234")
            os.chdir(ACTIVE)
            sys.path.insert(0, ACTIVE)
            from serving.hf_dkv_wrapper import DKVHFWrapper
            w = DKVHFWrapper(model_id=args.model,
                             config={"preset": args.preset,
                                     "quantization": args.quant or None})
            w.ensure_loaded()
            tok = w.tokenizer
        else:
            sys.path.insert(0, HERE)
            from run_longbench_cuda import load_plain
            import kv_baselines as _KB
            tok, model = load_plain(args.model, args.quant,
                                    _KB.needs_eager(args.arm))
        res["load_s"] = time.time() - t_load
        res["weights_gb"] = torch.cuda.memory_allocated() / 1e9

        prompt = build_filler(tok, args.ctx)
        res["ctx_actual"] = len(tok(prompt, add_special_tokens=False).input_ids)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if args.arm == "dkv":
            sid = f"ladder-{args.ctx}"
            w.active_session = sid
            out = w.generate(prompt, max_new_tokens=args.gen, temperature=0.0,
                             top_p=1.0, repetition_penalty=1.0)
            torch.cuda.synchronize()
            res["wall_s"] = time.perf_counter() - t0
            try:
                sess = w.manager.sessions.get(sid) or {}
                res["blocks"] = int(sum(sess.get("num_blocks") or []))
            except Exception:                                    # noqa: BLE001
                res["blocks"] = -1
            try:
                sys.path.insert(0, HERE)
                from run_longbench_cuda import dkv_kv_bytes
                res.update(dkv_kv_bytes(w.manager, res["ctx_actual"], sid))
            except Exception as e:                               # noqa: BLE001
                res["kv_accounting_error"] = f"{type(e).__name__}: {e}"
        else:
            sys.path.insert(0, HERE)
            import kv_baselines as KB
            ids = tok(prompt, add_special_tokens=False).input_ids
            t0 = time.perf_counter()
            # ANY kv_baselines method, not just dense. The point of putting
            # snapkv/streamingllm on the ladder is that they EVICT AFTER
            # PREFILL: to rank prefix tokens SnapKV needs attention from its
            # observation window to every prefix position, so the whole prefix
            # KV must be resident at that moment. If that is right, their PEAK
            # is dense's peak and they cannot reach a context dense cannot,
            # however small the cache they end up keeping. That is a claim about
            # DKV's contribution, so it gets measured rather than asserted.
            r = KB.run_baseline(model, tok, ids, args.arm, "cuda", args.gen,
                                set(), args.chunk,
                                json.loads(args.baseline_params))
            res["wall_s"] = time.perf_counter() - t0
            for k in ("prefill_s", "decode_s", "decode_tps", "ttft_s",
                      "kv_physical_gb", "kv_dense_equiv_gb"):
                res[k] = r.get(k)
        res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        res["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        # The card, not the allocator's opinion of it.
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        res["vram_total_gb"] = total
        # Headroom for the driver/display context, which is not ours to use.
        # Reserved is the honest figure to compare: it is what the allocator
        # actually took from the device.
        usable = total * SPILL_FRACTION
        res["spilled"] = bool(max(res["peak_gb"], res["peak_reserved_gb"]) > usable)
        res["status"] = "spilled" if res["spilled"] else "ok"
    except torch.cuda.OutOfMemoryError as e:                     # noqa: BLE001
        res["status"] = "oom"
        res["error"] = str(e)[:300]
    except Exception as e:                                       # noqa: BLE001
        # A CUDA OOM does not always surface as OutOfMemoryError -- it can come
        # back as a generic RuntimeError from inside a kernel or an allocator
        # helper. Classify on the message so the ladder does not report a real
        # ceiling as an unrelated crash.
        msg = str(e)
        res["status"] = ("oom" if ("out of memory" in msg.lower()
                                   or "CUDA error" in msg) else "error")
        res["error"] = f"{type(e).__name__}: {msg[:300]}"
    with open(args.point_json, "w", encoding="utf-8") as f:
        json.dump(res, f)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def report(paths: List[str]) -> None:
    for p in paths:
        store = ResumableJSONL(p, config=None, strict_config=False, read_only=True)
        recs = list(store.load_latest().values())
        store.close()
        if not recs:
            continue
        meta = {}
        if os.path.exists(p + ".meta.json"):
            with open(p + ".meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        print(f"\n=== {os.path.basename(p)} ===")
        print(f"{meta.get('model','?')}  quant={meta.get('quant','-')}  "
              f"preset={meta.get('preset','-')}")
        print(f"{'arm':>6} {'ctx':>8} {'actual':>8} {'peak GB':>9} {'wall s':>8} "
              f"{'KV GB':>8} {'status':>8}")
        rows = sorted(recs, key=lambda r: (r.get("arm", ""), r.get("ctx", 0)))
        for r in rows:
            kv = r.get("kv_physical_gb")
            print(f"{r.get('arm','?'):>6} {r.get('ctx',0):>8} "
                  f"{r.get('ctx_actual','-'):>8} "
                  f"{(f'{r['peak_gb']:.2f}' if r.get('peak_gb') else '-'):>9} "
                  f"{(f'{r['wall_s']:.1f}' if r.get('wall_s') else '-'):>8} "
                  f"{(f'{kv:.3f}' if kv else '-'):>8} "
                  f"{r.get('status','?'):>8}")
        # Slope: the marginal GB per 1k tokens, from the largest two OK points
        # of each arm. This is the number that predicts the next rung.
        for arm in sorted({r.get("arm") for r in rows}):
            # Spilled points are excluded: their peak is partly host memory
            # and their wall time is bandwidth, so fitting through them would
            # put both the slope and the predicted ceiling in the wrong place.
            ok = [r for r in rows if r.get("arm") == arm and r.get("status") == "ok"
                  and r.get("peak_gb") and r.get("ctx_actual")]
            if len(ok) >= 2:
                a, b = ok[-2], ok[-1]
                dt = (b["ctx_actual"] - a["ctx_actual"]) / 1000.0
                if dt > 0:
                    slope = (b["peak_gb"] - a["peak_gb"]) / dt
                    icept = b["peak_gb"] - slope * b["ctx_actual"] / 1000.0
                    print(f"  {arm}: {slope:.4f} GB per 1k tokens, "
                          f"floor {icept:.2f} GB  "
                          f"(max OK {b['ctx_actual']} tok @ {b['peak_gb']:.2f} GB)")
                    total = b.get("vram_total_gb")
                    if total and slope > 0:
                        pred = (total * SPILL_FRACTION - icept) / slope * 1000.0
                        print(f"         predicted ceiling ~{pred:,.0f} tok on a "
                              f"{total:.1f} GB card")
            spilled = [r for r in rows if r.get("arm") == arm
                       and r.get("status") == "spilled"]
            if spilled:
                f = min(spilled, key=lambda r: r.get("ctx", 0))
                print(f"         first spill at {f.get('ctx')} tok "
                      f"({f.get('peak_gb', 0):.2f} GB) -- timings at and above "
                      f"this length are host-memory bound, not compute")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Before ANY torch.compile: without cl.exe on PATH the Inductor
    # decode path falls back to eager and every latency number here
    # understates DKV. Quality is unaffected; timings are not.
    from msvc_env import ensure_msvc
    ensure_msvc()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-8b")
    ap.add_argument("--arms", nargs="+", default=["dense", "dkv"])
    ap.add_argument("--preset", default="mid", choices=["low", "mid", "high", "ultra"])
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--contexts", type=int, nargs="+", default=DEFAULT_CONTEXTS)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--baseline-params", default="{}",
                    help="JSON params for a kv_baselines arm, e.g. "
                         "'{\"budget\": 2016}'")
    ap.add_argument("--stop-after-oom", type=int, default=1,
                    help="consecutive OOMs at increasing context before this "
                         "arm is considered finished")
    # child-mode
    ap.add_argument("--arm", default="")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--point-json", default="")
    ap.add_argument("--report", nargs="*", default=None)
    args = ap.parse_args()

    if args.report is not None:
        paths: List[str] = []
        for pat in (args.report or ["paper/results/ladder/*.jsonl"]):
            paths.extend(sorted(glob.glob(pat)))
        if not paths:
            raise SystemExit("no ladder files matched")
        return report(paths)

    if args.point_json:
        return run_point(args)

    out = args.out or os.path.join(
        REPO, "paper", "results", "ladder",
        f"{args.model.split('/')[-1]}_{args.preset}_{args.quant}.jsonl")
    cfg = {"model": args.model, "preset": args.preset, "quant": args.quant,
           "gen": args.gen, "chunk": args.chunk}
    store = ResumableJSONL(out, config=cfg)
    done = store.load_done()
    print(f"[ckpt] {out}\n[ckpt] {len(done)} points already recorded")

    tmp = tempfile.mkdtemp(prefix="dkv-ladder-")
    for arm in args.arms:
        ooms = 0
        last_rate = None          # seconds per 1k tokens at the previous rung
        for ctx in sorted(args.contexts):
            key = f"{arm}@{ctx}"
            if key in done:
                print(f"  skip {key} (done)")
                continue
            if ooms >= args.stop_after_oom:
                print(f"  {arm}: stopping, ceiling found below {ctx}")
                store.append(key, arm=arm, ctx=ctx, status="skipped_after_oom")
                continue
            pj = os.path.join(tmp, f"{arm}_{ctx}.json")
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--model", args.model, "--arm", arm, "--ctx", str(ctx),
                   "--preset", args.preset, "--quant", args.quant,
                   "--gen", str(args.gen), "--chunk", str(args.chunk),
                   "--baseline-params", args.baseline_params,
                   "--point-json", pj]
            t0 = time.time()
            print(f"  running {key} ...", flush=True)
            try:
                p = subprocess.run(cmd, cwd=REPO, timeout=args.timeout,
                                   capture_output=True, text=True)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc, p = -9, None
            if os.path.exists(pj):
                with open(pj, encoding="utf-8") as f:
                    res = json.load(f)
            else:
                # The child died without writing -- almost always the OS or the
                # CUDA driver killing it, which at the top of a ladder is a
                # ceiling and not a bug. Recorded as its own status so it is
                # never confused with a measured OOM.
                tail = ""
                if p is not None:
                    tail = (p.stderr or "")[-400:]
                res = {"arm": arm, "ctx": ctx,
                       "status": "timeout" if rc == -9 else "died",
                       "returncode": rc, "stderr_tail": tail}
            # ── second spill signal: the latency cliff ──────────────────────
            # Allocation alone does not catch the onset. Measured on granite
            # dense Q4, the point at 24,576 tokens reported 11.23 GB against a
            # 12.28 GB card -- under any sane fraction of the card, and so not
            # flagged -- while taking 65.7 s where the trend predicted ~16 s.
            # It was already paging. Cost per token is near-linear in context
            # for a compute-bound prefill, so a sudden jump in seconds-per-1k
            # against the previous rung is the earliest honest evidence that
            # the memory system, not the GPU, is setting the pace.
            sec_per_1k = None
            if res.get("status") == "ok" and res.get("wall_s") and res.get("ctx_actual"):
                sec_per_1k = res["wall_s"] / (res["ctx_actual"] / 1000.0)
                if last_rate is not None and sec_per_1k > last_rate * CLIFF_RATIO:
                    res["status"] = "degraded"
                    res["degraded_vs_prev"] = round(sec_per_1k / last_rate, 2)
                    res["sec_per_1k"] = round(sec_per_1k, 3)
                else:
                    res["sec_per_1k"] = round(sec_per_1k, 3)
                    last_rate = sec_per_1k

            store.append(key, **res)
            st = res.get("status")
            print(f"    {key}: {st} peak={res.get('peak_gb','-')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if st in ("oom", "died", "timeout", "spilled", "degraded"):
                # `spilled` counts as a ceiling: the point technically returned,
                # but it returned by borrowing host memory, and every larger
                # context borrows more. Climbing further measures PCIe.
                ooms += 1
            else:
                ooms = 0
    store.close()
    print()
    report([out])


if __name__ == "__main__":
    main()
