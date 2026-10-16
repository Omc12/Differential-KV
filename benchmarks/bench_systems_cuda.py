#!/usr/bin/env python3
"""TTFT and decode throughput vs context length — the systems table.

WHY THIS EXISTS
---------------
The quality and memory legs of this campaign are measured; the SPEED leg is
not, and that is the one an MLSys reviewer reads first. The per-item timings
recorded by the quality harnesses are not a substitute:

  * the early LongBench arms carry `inductor_fused: False` — Inductor never
    compiled, so those latencies UNDERSTATE DKV;
  * DKV's rows record only `e2e_tps`, wall time for prefill AND decode
    together, which is not comparable to the baselines' `decode_tps`;
  * they are single measurements taken while scoring, with no repeats.

MEASURED IDENTICALLY FOR EVERY ARM, which matters more than the method being
clever. systems-v2: ONE call per repeat generates 1+K tokens, and both numbers
come from inside it --

    ttft      = time to the first emitted token (prefill + any compression)
    decode/s  = decode forwards / their wall time, per-token synchronised

(v1 differenced two separate calls, T(1+K) - T(1), which folded one prefill's
jitter into the decode rate; see _call.) No arm's number comes from a different
definition than another's -- which is exactly the trap `e2e_tps` vs
`decode_tps` already was. Peaks are allocated AND reserved, max over repeats;
DKV rows carry their KV footprint (accounting v2) and every DKV_* switch.

Reports the MEDIAN of `--reps` repeats, and the min/max, so a reader can see
the spread rather than trusting one sample. First repeat is discarded as warmup
(kernel autotuning, allocator growth).

REFUSES TO RUN WITHOUT cl.exe unless forced: without it the Inductor decode
path falls back to eager and every number here is wrong in the direction that
flatters the competition.

USAGE
    python benchmarks/bench_systems_cuda.py --model Qwen/Qwen3.5-4B \
        --arms dense dkv snapkv --contexts 8192 16384 32768
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)
from checkpoint import ResumableJSONL                            # noqa: E402
from code_fingerprint import decode_fingerprint                  # noqa: E402


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def run_point(args) -> Dict[str, Any]:
    import torch
    from context_ladder import build_filler
    import kv_baselines as KB

    res: Dict[str, Any] = {"arm": args.arm, "ctx": args.ctx}
    if args.arm == "dkv":
        os.environ.setdefault("DKV_RSVD_SEED", "1234")
        os.environ.setdefault("DKV_SVD_SEED", "1234")
        cwd = os.getcwd()
        os.chdir(ACTIVE)
        sys.path.insert(0, ACTIVE)
        from serving.decode_config import apply_best_decode_defaults
        apply_best_decode_defaults()
        from serving.hf_dkv_wrapper import DKVHFWrapper
        w = DKVHFWrapper(model_id=args.model,
                         config={"preset": args.preset,
                                 "quantization": args.quant or None})
        w.ensure_loaded()
        tok, model = w.tokenizer, None
        os.chdir(cwd)
    else:
        from run_longbench_cuda import load_plain
        tok, model = load_plain(args.model, args.quant, KB.needs_eager(args.arm))
        w = None

    prompt = build_filler(tok, args.ctx)
    ids = tok(prompt, add_special_tokens=False).input_ids
    res["ctx_actual"] = len(ids)
    bparams = json.loads(args.baseline_params)
    K = args.gen

    def _call(n_tokens: int) -> Dict[str, Any]:
        """ONE generate of n_tokens; TTFT and the decode rate both come from it.

        systems-v2 (2026-10-17). v1 took decode/s = K / (T(1+K) - T(1)) from two
        separate calls, each with its own prefill, so the prefill's run-to-run
        jitter landed in the decode number -- at 131k, ~100 s of prefill against
        ~6 s of decode, enough to move it by tens of percent. Now:
          DKV       t0 at generate() entry, one timestamp per emitted token taken
                    right after the sampler's .item() sync (hf_dkv_wrapper);
                    TTFT = t[0] - t0, decode = (n-1) / (t[-1] - t[0]).
          baselines run_baseline's own split: TTFT = prefill + post-hoc
                    compression, decode = tokens / decode loop time; its loop
                    also syncs per token (argmax .item()).
        Both rates are per decode forward, measured inside one call.
        """
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out: Dict[str, Any] = {}
        if args.arm == "dkv":
            sid = f"sys-{args.ctx}-{n_tokens}-{time.time_ns()}"
            w.active_session = sid
            w.generate(prompt, max_new_tokens=n_tokens, temperature=0.0,
                       top_p=1.0, repetition_penalty=1.0)
            torch.cuda.synchronize()
            tm = getattr(w, "last_timing", None) or {}
            ts = tm.get("token_times") or []
            if not ts:
                raise RuntimeError("DKV wrapper recorded no token timestamps")
            out["ttft_s"] = ts[0] - tm["t0"]
            out["decode_tps"] = ((len(ts) - 1) / (ts[-1] - ts[0])
                                 if len(ts) > 1 and ts[-1] > ts[0] else float("nan"))
            out["gen_tokens"] = len(ts)
            out["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
            try:
                from run_longbench_cuda import dkv_kv_bytes
                out.update(dkv_kv_bytes(w.manager, len(ids), sid))
            except Exception as e:                               # noqa: BLE001
                out["kv_error"] = repr(e)
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
        else:
            r = KB.run_baseline(model, tok, ids, args.arm, "cuda", n_tokens,
                                set(), args.chunk, bparams)
            torch.cuda.synchronize()
            out["ttft_s"] = r["ttft_s"]
            out["decode_tps"] = r["decode_tps"]
            out["gen_tokens"] = r["gen_tokens"]
            # run_baseline resets the peak counters between prefill and
            # decode; the point's peak is the larger of the two phases.
            out["peak_gb"] = max(r.get("peak_prefill_gb") or 0,
                                 torch.cuda.max_memory_allocated() / 1e9)
            out["peak_reserved_gb"] = max(r.get("peak_prefill_reserved_gb") or 0,
                                          torch.cuda.max_memory_reserved() / 1e9)
            for k in ("kv_physical_gb", "kv_dense_equiv_gb", "kv_compression_x"):
                out[k] = r.get(k)
        return out

    reps = []
    for rep in range(args.reps + 1):          # +1: first is warmup, discarded
        m = _call(1 + K)
        if rep:
            reps.append(m)

    def _col(k):
        return [m[k] for m in reps if isinstance(m.get(k), (int, float))]

    ttfts, rates = _col("ttft_s"), _col("decode_tps")
    peak, resv = max(_col("peak_gb")), max(_col("peak_reserved_gb"))
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    res.update({
        "ttft_s": _median(ttfts), "ttft_min": min(ttfts), "ttft_max": max(ttfts),
        "decode_tps": _median(rates), "tps_min": min(rates), "tps_max": max(rates),
        # MAX over repeats, allocated AND reserved: Windows pages reserved
        # memory to the host instead of failing, so allocated alone misses a
        # spill (memory note: two detectors).
        "peak_gb": peak, "peak_reserved_gb": resv, "vram_total_gb": total,
        "spilled": bool(max(peak, resv) > total * 0.94),
        "gen_tokens": min(_col("gen_tokens")) if _col("gen_tokens") else None,
        "reps": args.reps, "protocol": "systems-v2",
        "inductor_fused": res.get("inductor_fused", True),
    })
    for k in ("kv_physical_gb", "kv_dense_equiv_gb", "kv_compression_x",
              "kv_acct", "kv_layers", "kv_error"):
        if k in reps[-1]:
            res[k] = reps[-1][k]
    return res


def report(paths: List[str]) -> None:
    # GROUPED BY MODEL. Reporting them together silently mixed granite
    # (4k/8k/16k) with Qwen3.5-4B (8k/16k/32k), so the shared 8k and 16k
    # columns held whichever model was read last -- a table where every number
    # looked plausible and half of them were the wrong model.
    by_model = {}
    for p in paths:
        meta = {}
        if os.path.exists(p + ".meta.json"):
            with open(p + ".meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        model = meta.get("model") or os.path.basename(p)
        st = ResumableJSONL(p, config=None, strict_config=False, read_only=True)
        by_model.setdefault(model, []).extend(
            r for r in st.load_latest().values() if not r.get("error"))
        st.close()
    if not by_model:
        print("no systems results")
        return

    for model in sorted(by_model):
        rows = by_model[model]
        if not rows:
            continue
        arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "dense", a))
        ctxs = sorted({r["ctx"] for r in rows})
        print(f"\n################ {model} ################")
        for title, key in [("TTFT (s, lower better)", "ttft_s"),
                           ("decode throughput (tok/s, higher better)", "decode_tps"),
                           ("peak VRAM allocated (GB)", "peak_gb"),
                           ("peak VRAM reserved (GB)", "peak_reserved_gb")]:
            print(f"\n=== {title} ===")
            print(f"{'arm':>10} " + " ".join(f"{c:>10}" for c in ctxs))
            print("-" * (11 + 11 * len(ctxs)))
            for a in arms:
                cells = []
                for c in ctxs:
                    m = [r for r in rows if r["arm"] == a and r["ctx"] == c]
                    if not m:
                        cells.append(f"{'-':>10}")
                        continue
                    v = m[0].get(key)
                    mark = "*" if m[0].get("spilled") else " "
                    cells.append(f"{v:>9.2f}{mark}" if v is not None else f"{'-':>10}")
                print(f"{a:>10} " + " ".join(cells))
    print("\n* = spilled to host memory; that timing is PCIe bandwidth, not compute.")


def main():
    from msvc_env import ensure_msvc
    fused = ensure_msvc()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--arms", nargs="+", default=["dense", "dkv", "snapkv"])
    ap.add_argument("--preset", default="mid", choices=["low", "mid", "high", "ultra"])
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--baseline-params", default='{"budget": 2016, "window": 32}')
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-eager", action="store_true",
                    help="run even without cl.exe. Every timing will then "
                         "understate DKV; only for debugging.")
    ap.add_argument("--arm", default="")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--point-json", default="")
    ap.add_argument("--report", nargs="*", default=None)
    ap.add_argument("--timeout", type=float, default=5400,
                    help="seconds per point (all repeats); a point past it is "
                         "recorded as an error and the arm stops there")
    args = ap.parse_args()

    if args.report is not None:
        paths = []
        for pat in (args.report or ["paper/results/systems/*.jsonl"]):
            paths += sorted(glob.glob(pat))
        if not paths:
            raise SystemExit("no systems files matched")
        return report(paths)

    if not fused and not args.allow_eager:
        raise SystemExit(
            "\ncl.exe is not available, so Inductor cannot compile the decode\n"
            "path and every timing here would understate DKV. Fix the toolchain\n"
            "or pass --allow-eager if you know that is what you want.\n")

    if args.point_json:
        r = run_point(args)
        r["inductor_fused"] = fused
        with open(args.point_json, "w", encoding="utf-8") as f:
            json.dump(r, f)
        return

    import subprocess
    import tempfile
    tag = args.model.split("/")[-1]
    out = args.out or os.path.join(REPO, "paper", "results", "systems",
                                   f"{tag}_{args.preset}_{args.quant}.jsonl")
    cfg = {"model": args.model, "preset": args.preset, "quant": args.quant,
           "gen": args.gen, "reps": args.reps, "chunk": args.chunk,
           "baseline_params": json.loads(args.baseline_params),
           "dkv_decode_rev": decode_fingerprint(), "protocol": "systems-v2",
           # Every DKV_* switch in force: a hybrid or streaming run must never
           # merge into, or resume from, a low-rank exact store.
           "dkv_env": {k: v for k, v in sorted(os.environ.items())
                       if k.startswith("DKV_")} or None}
    store = ResumableJSONL(out, config=cfg)
    done = store.load_done()
    latest = store.load_latest()
    tmp = tempfile.mkdtemp(prefix="dkv-sys-")

    for arm in args.arms:
        stop = False
        for ctx in sorted(args.contexts):
            key = f"{arm}@{ctx}"
            if key in done:
                print(f"  skip {key}")
                prev = latest.get(key) or {}
                if prev.get("spilled") or prev.get("status") == "skipped_after_spill":
                    stop = True
                continue
            if stop:
                # Past a spill or a failure the timing is PCIe or nothing: record
                # the point as skipped instead of spending hours measuring paging.
                store.append(key, arm=arm, ctx=ctx, status="skipped_after_spill")
                print(f"  {key}: skipped after a spill/failure at a shorter context")
                continue
            pj = os.path.join(tmp, f"{arm}_{ctx}.json")
            # One child per point: an OOM must not poison the arms after it.
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--model", args.model, "--arm", arm, "--ctx", str(ctx),
                   "--preset", args.preset, "--quant", args.quant,
                   "--gen", str(args.gen), "--reps", str(args.reps),
                   "--chunk", str(args.chunk),
                   "--baseline-params", args.baseline_params,
                   "--point-json", pj]
            print(f"  running {key} ...", flush=True)
            try:
                p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                   timeout=args.timeout)
                err = (p.stderr or "")[-300:]
            except subprocess.TimeoutExpired:
                err = f"timeout after {args.timeout:.0f} s (paging?)"
            if os.path.exists(pj):
                with open(pj, encoding="utf-8") as f:
                    r = json.load(f)
            else:
                r = {"arm": arm, "ctx": ctx, "error": err}
                stop = True
            if r.get("spilled"):
                stop = True
            store.append(key, **r)
            print(f"    {key}: ttft={r.get('ttft_s','-')} "
                  f"tps={r.get('decode_tps','-')} peak={r.get('peak_gb','-')}",
                  flush=True)
    store.close()
    report([out])


if __name__ == "__main__":
    main()
