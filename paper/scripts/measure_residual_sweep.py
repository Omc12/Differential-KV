#!/usr/bin/env python3
"""E2 — residual-budget accuracy/memory trade-off for the DiffKV active runtime.

For a fixed context length and forced compressed decode, sweep DIFFKV_MAX_RESIDUAL
(the number of exact fp16 tokens kept per 256-token block) and record, per cell:
  needle_found, decode_tps, mx_decode_peak_gb, analytic store bytes, ratio vs dense.

This isolates the central design trade-off: more residuals -> exact retrieval but
less compression; fewer -> better compression but the buried needle is lost (the
pre-fix failure). Each cell runs in its own subprocess for clean memory isolation.

Usage:
  diffkv_venv/bin/python3 paper/scripts/measure_residual_sweep.py \
      --ctx 16384 32768 --residuals 0 8 16 32 64 --gen 96 \
      --out paper/generated/residual_sweep.json
"""
import os, sys, json, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", nargs="+", type=int, default=[16384, 32768])
    ap.add_argument("--residuals", nargs="+", type=int, default=[0, 8, 16, 32, 64])
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--out", default="paper/generated/residual_sweep.json")
    ap.add_argument("--single", help="internal: R,ctx")
    args = ap.parse_args()

    sys.path.insert(0, ACTIVE); sys.path.insert(0, BENCH); sys.path.insert(0, HERE)
    os.chdir(ACTIVE)

    if args.single:
        # one cell, in this (already env-configured) subprocess
        from measure_active import run_cell
        from bench_common import build_niah_prompt, _load_ref_tokenizer
        R, ctx = args.single.split(","); R = int(R); ctx = int(ctx)
        tok = _load_ref_tokenizer()
        text, ntok = build_niah_prompt(ctx, tok)
        r = run_cell(ctx, args.gen, text)
        r.update({"max_residual": R, "ctx": ctx, "mode": "compressed"})
        print("__CELL__ " + json.dumps(r))
        return

    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = []
    for ctx in args.ctx:
        for R in args.residuals:
            print(f">>> R={R} ctx={ctx}", flush=True)
            env = os.environ.copy()
            env["DIFFKV_COMPRESSED_DECODE"] = "1"   # force compressed
            env["DIFFKV_MAX_RESIDUAL"] = str(R)
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--single", f"{R},{ctx}", "--gen", str(args.gen)]
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            line = [l for l in p.stdout.splitlines() if l.startswith("__CELL__")]
            if line:
                r = json.loads(line[-1][len("__CELL__ "):])
                results.append(r)
                kv = r["kv"]
                print(f"    needle={r['needle_found']} tps={r['decode_tps']:.1f} "
                      f"mx_decode={r['mx_decode_peak_gb']:.2f} "
                      f"store={kv['store_used_bytes']/1e9:.3f}GB "
                      f"ratio={kv['ratio_used_vs_dense']:.2f}x", flush=True)
            else:
                print("    FAILED:\n" + p.stdout[-500:] + "\n" + p.stderr[-800:], flush=True)
                results.append({"max_residual": R, "ctx": ctx, "status": "error",
                                "stderr": p.stderr[-1500:]})
            with open(out_path, "w") as f:
                json.dump({"results": results}, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
