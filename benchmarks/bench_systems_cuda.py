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
clever. TTFT is the time to produce ONE token; the decode rate comes from a
second call for 1+K tokens:

    ttft      = T(1)
    decode/s  = K / (T(1+K) - T(1))

Two calls per point, the same two for dense, DKV and every baseline. No arm's
number comes from a different definition than another's — which is exactly the
trap `e2e_tps` vs `decode_tps` already was.

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

    def _call(n_tokens: int) -> float:
        """Wall time for a full generate of n_tokens, synchronised."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.arm == "dkv":
            sid = f"sys-{args.ctx}-{n_tokens}-{time.time_ns()}"
            w.active_session = sid
            w.generate(prompt, max_new_tokens=n_tokens, temperature=0.0,
                       top_p=1.0, repetition_penalty=1.0)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
        else:
            KB.run_baseline(model, tok, ids, args.arm, "cuda", n_tokens,
                            set(), args.chunk, bparams)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        return dt

    ttfts, rates, peaks = [], [], []
    for rep in range(args.reps + 1):          # +1: first is warmup, discarded
        torch.cuda.reset_peak_memory_stats()
        t1 = _call(1)
        tk = _call(1 + K)
        if rep == 0:
            continue
        ttfts.append(t1)
        dec = tk - t1
        rates.append(K / dec if dec > 0 else float("nan"))
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    res.update({
        "ttft_s": _median(ttfts), "ttft_min": min(ttfts), "ttft_max": max(ttfts),
        "decode_tps": _median(rates), "tps_min": min(rates), "tps_max": max(rates),
        "peak_gb": _median(peaks), "vram_total_gb": total,
        # Above this the run is partly in host memory and the timing is PCIe
        # bandwidth, not compute. Recorded, not silently dropped.
        "spilled": bool(_median(peaks) > total * 0.94),
        "gen_tokens": K, "reps": args.reps,
        "inductor_fused": res.get("inductor_fused", True),
    })
    return res


def report(paths: List[str]) -> None:
    rows = []
    for p in paths:
        st = ResumableJSONL(p, config=None, strict_config=False, read_only=True)
        rows += [r for r in st.load_latest().values() if not r.get("error")]
        st.close()
    if not rows:
        print("no systems results")
        return
    arms = sorted({r["arm"] for r in rows})
    ctxs = sorted({r["ctx"] for r in rows})
    for title, key, unit, hi_good in [("TTFT (s, lower better)", "ttft_s", "s", False),
                                      ("decode throughput (tok/s, higher better)", "decode_tps", "", True),
                                      ("peak VRAM (GB)", "peak_gb", "GB", False)]:
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
           "dkv_decode_rev": decode_fingerprint(), "protocol": "systems-v1"}
    store = ResumableJSONL(out, config=cfg)
    done = store.load_done()
    tmp = tempfile.mkdtemp(prefix="dkv-sys-")

    for arm in args.arms:
        for ctx in sorted(args.contexts):
            key = f"{arm}@{ctx}"
            if key in done:
                print(f"  skip {key}")
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
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            if os.path.exists(pj):
                with open(pj, encoding="utf-8") as f:
                    r = json.load(f)
            else:
                r = {"arm": arm, "ctx": ctx, "error": (p.stderr or "")[-300:]}
            store.append(key, **r)
            print(f"    {key}: ttft={r.get('ttft_s','-')} "
                  f"tps={r.get('decode_tps','-')} peak={r.get('peak_gb','-')}",
                  flush=True)
    store.close()
    report([out])


if __name__ == "__main__":
    main()
