#!/usr/bin/env python3
"""
run_sequential_clean.py — Synchronous sequential sweep with per-test cooldown.

Runs each (engine, context) cell in a completely isolated bench_worker.py process,
printing live status and sleeping 10s between cells so Metal/RAM fully cool down.
Overwrites benchmarks/results/clean_{engine}_{ctx}.json.
"""
import os
import sys
import json
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
VENV_PY = os.path.join(REPO, "diffkv_venv", "bin", "python3")
WORKER = os.path.join(HERE, "bench_worker.py")
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

CONTEXTS = [4096, 8192, 16384, 32768, 65536]
ENGINES = ["active", "dense"]
COOLDOWN_SEC = 10


def run_cell(engine, ctx):
    prompt_file = os.path.join(RESULTS, f"prompt_{ctx}.txt")
    result_file = os.path.join(RESULTS, f"clean_{engine}_{ctx}.json")

    print(f"\n{'=' * 65}")
    print(f"  CELL: {engine.upper()} @ {ctx // 1024}K context")
    print(f"{'=' * 65}")

    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })

    cmd = [
        VENV_PY, WORKER,
        "--engine", engine,
        "--ctx", str(ctx),
        "--gen", "128",
        "--prompt-file", prompt_file,
        "--result-file", result_file,
        "--dense-model-id", MODEL_ID,
    ]

    t_cell_start = time.time()
    proc = subprocess.run(cmd, env=env, cwd=REPO)

    if proc.returncode != 0:
        print(f"  [CELL FAILED] exit code {proc.returncode} for {engine} @ {ctx // 1024}K")
        # Save OOM/failure record so facts/tables can record status
        with open(result_file, "w") as f:
            json.dump({
                "status": "oom" if proc.returncode in (-9, 137) else "error",
                "engine": engine,
                "ctx_target": ctx,
                "exit_code": proc.returncode,
            }, f, indent=2)
        return False

    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        if r.get("status") == "ok":
            print(f"  ✓ SUCCESS: prefill={r['prefill_s']:.2f}s | "
                  f"decode={r['decode_tps']:.2f} tok/s | "
                  f"kv_mem={r.get('kv_mem_gb', 0):.2f}GB | "
                  f"peak_RAM={r.get('mx_peak_gb', 0):.2f}GB | "
                  f"needle={'PASS' if r.get('needle_found') else 'FAIL'}")
            return True
    return False


def main():
    os.makedirs(RESULTS, exist_ok=True)
    total_cells = len(ENGINES) * len(CONTEXTS)
    curr_cell = 0
    t_start = time.time()

    print(f"Starting synchronous sequential benchmark sweep ({total_cells} cells)...")
    print(f"Cooldown between cells: {COOLDOWN_SEC}s\n")

    summary = {}
    for engine in ENGINES:
        for ctx in CONTEXTS:
            curr_cell += 1
            print(f"Progress: [{curr_cell}/{total_cells}]")
            ok = run_cell(engine, ctx)
            summary[f"{engine}_{ctx}"] = "ok" if ok else "FAIL"

            if curr_cell < total_cells:
                print(f"  [Cooldown] Sleeping {COOLDOWN_SEC}s for GPU/RAM reset...")
                time.sleep(COOLDOWN_SEC)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  SWEEP COMPLETE in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 65}")
    for k, v in summary.items():
        print(f"  {k:25s} : {v}")


if __name__ == "__main__":
    main()
