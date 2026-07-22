#!/usr/bin/env python3
"""
run_sweep.py — Calls bench_worker.py directly (the existing, tested worker)
for each (engine, ctx) cell and writes results to clean_{engine}_{ctx}.json.

bench_worker.py already handles:
  - active: MLXDiffKVWrapper with two-pass TTFT timing (isolated process)
  - dense:  mlx_lm direct forward + decode loop (isolated process)

Run order: active 4k→64k, then dense 4k→64k.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
VENV_PY = os.path.join(REPO, "diffkv_venv", "bin", "python3")
WORKER = os.path.join(HERE, "bench_worker.py")
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CONTEXTS = [4096, 8192, 16384, 32768, 65536]
GEN = 128


def run_cell(engine, ctx):
    prompt_file = os.path.join(RESULTS, f"prompt_{ctx}.txt")
    result_file = os.path.join(RESULTS, f"clean_{engine}_{ctx}.json")

    if not os.path.exists(prompt_file):
        print(f"  [SKIP] no prompt file for {ctx}", flush=True)
        return False

    print(f"\n{'='*60}", flush=True)
    print(f"  {engine.upper()} @ {ctx//1024}k", flush=True)
    print(f"{'='*60}", flush=True)

    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

    cmd = [
        VENV_PY, WORKER,
        "--engine", engine,
        "--ctx", str(ctx),
        "--gen", str(GEN),
        "--prompt-file", prompt_file,
        "--result-file", result_file,
        "--dense-model-id", MODEL_ID,
    ]

    proc = subprocess.run(cmd, env=env, cwd=REPO)

    if proc.returncode != 0:
        print(f"  [FAIL] exit {proc.returncode}", flush=True)
        return False

    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        if r.get("status") == "ok":
            print(f"  ✓  prefill={r['prefill_s']:.2f}s  "
                  f"decode={r['decode_tps']:.2f} tok/s  "
                  f"peak={r.get('mx_peak_gb', 0):.2f}GB  "
                  f"needle={'PASS' if r.get('needle_found') else 'FAIL'}", flush=True)
            return True
        elif r.get("status") == "oom":
            print(f"  [OOM]", flush=True)
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", choices=["active", "dense"],
                    default=["active", "dense"])
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    summary = {}
    for engine in args.engines:
        for ctx in args.contexts:
            ok = run_cell(engine, ctx)
            summary[f"{engine}_{ctx}"] = "ok" if ok else "FAIL"

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE  ({elapsed:.0f}s / {elapsed/60:.1f} min)")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k:25s}  {v}")


if __name__ == "__main__":
    main()
