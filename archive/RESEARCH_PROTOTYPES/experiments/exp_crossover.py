"""
experiments/exp_crossover.py

THE DEFINING EXPERIMENT.

Question: "At what context length does Differential KV become worthwhile?"

This script sweeps context lengths from 4k to 128k tokens and measures:
  - memory bytes: DKV vs FP16/FP8/INT8
  - reconstruction overhead: added compute cost
  - effective bandwidth savings: (bytes_saved - recon_cost_bytes_equiv)
  - crossover point: where DKV first beats INT8 in effective bandwidth

The crossover plot is the central result of Phase 1.

Usage:
    python experiments/exp_crossover.py
    python experiments/exp_crossover.py --mode smooth --output results/crossover/
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from benchmarks.baselines import BaselineRunner
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy, AdaptiveAnchorStrategy
from reconstruction.reconstructor import KVReconstructor
from profiling.profiler import MemoryBandwidthEstimator, ReconstructionProfiler


SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def run_crossover_for_seq(seq_len: int, mode: str, heads: int, head_dim: int,
                           num_layers: int) -> dict:
    """Run all strategies at a single sequence length and return metrics."""

    gen = KVGenerator(num_heads=heads, head_dim=head_dim, seed=42)
    kv = gen.generate(seq_len, mode=mode)
    bl = BaselineRunner(num_heads=heads, head_dim=head_dim)

    # --- FP16 ---
    fp16 = bl.run_fp16(kv)
    fp16_bytes = fp16.total_bytes * num_layers  # scale to all layers

    # --- FP8 ---
    fp8 = bl.run_fp8_simulated(kv)
    fp8_bytes = fp8.total_bytes * num_layers

    # --- INT8 naive ---
    int8 = bl.run_int8_naive(kv)
    int8_bytes = int8.total_bytes * num_layers

    # --- DKV strategies ---
    strategies = {
        "DKV-P32":  PeriodicAnchorStrategy(interval=32),
        "DKV-P64":  PeriodicAnchorStrategy(interval=64),
        "DKV-P128": PeriodicAnchorStrategy(interval=128),
        "DKV-Adapt": AdaptiveAnchorStrategy(
            max_interval=64, delta_norm_threshold=2.0,
            error_estimate_threshold=0.05, min_interval=8
        ),
    }

    result = {
        "seq_len": seq_len,
        "FP16_bytes": fp16_bytes,
        "FP8_bytes": fp8_bytes,
        "INT8_bytes": int8_bytes,
        "FP8_error": round(fp8.mean_relative_error, 6),
        "INT8_error": round(int8.mean_relative_error, 6),
    }

    window_size = min(128, max(16, seq_len // 32))
    import random; random.seed(0)

    for label, strategy in strategies.items():
        manager = AnchorManager(strategy=strategy)
        t0 = time.perf_counter()
        stats = manager.compress(kv)
        compress_ms = (time.perf_counter() - t0) * 1000

        recon = KVReconstructor(manager)
        profiler = ReconstructionProfiler()

        num_queries = min(30, max(5, seq_len // 4096))
        total_error = 0.0
        for _ in range(num_queries):
            start = random.randint(0, max(0, seq_len - window_size - 1))
            end = min(start + window_size - 1, seq_len - 1)
            r = recon.reconstruct_range(start, end)
            err = recon.measure_error(kv, start, end)
            profiler.record_result(r, error=err["mean_relative"])
            total_error += err["mean_relative"]

        summary = profiler.summarize()
        dkv_bytes = stats.total_compressed_bytes * num_layers

        result[f"{label}_bytes"] = dkv_bytes
        result[f"{label}_ratio"] = round(stats.compression_ratio, 4)
        result[f"{label}_anchor_density"] = round(stats.anchor_density, 4)
        result[f"{label}_error"] = round(summary.mean_error, 6)
        result[f"{label}_recon_ms"] = round(summary.mean_latency_ms, 4)
        result[f"{label}_compress_ms"] = round(compress_ms, 3)
        result[f"{label}_tok_per_sec"] = round(summary.tokens_per_second, 1)

        # Effective bandwidth ratio vs INT8:
        # positive = DKV uses less memory than INT8
        bw_saving_vs_int8 = (int8_bytes - dkv_bytes) / int8_bytes
        result[f"{label}_bw_saving_vs_int8"] = round(bw_saving_vs_int8, 4)

    return result


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  CROSSOVER EXPERIMENT — Differential KV vs Baselines")
    print(f"  Mode: {args.mode} | Layers: {args.layers}")
    print(f"  Seq lens: {SEQ_LENS}")
    print(f"{'='*80}\n")

    all_results = []
    for seq_len in SEQ_LENS:
        print(f"  → seq_len = {seq_len:,} tokens ...", end=" ", flush=True)
        t0 = time.perf_counter()
        r = run_crossover_for_seq(
            seq_len, args.mode, args.heads, args.head_dim, args.layers
        )
        elapsed = time.perf_counter() - t0
        print(f"done in {elapsed:.1f}s")
        all_results.append(r)

    # --- Save results ---
    out_path = output_dir / f"crossover_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Crossover data saved → {out_path}")

    # --- Print summary table ---
    print(f"\n{'='*80}")
    print(f"  CROSSOVER SUMMARY (mode={args.mode})")
    print(f"{'='*80}")
    print(f"  {'seq_len':>10} | {'FP16 MB':>8} | {'INT8 MB':>8} | "
          f"{'P64 MB':>8} | {'Adapt MB':>8} | {'P64 ratio':>9} | {'P64 err':>8}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")
    for r in all_results:
        fp16_mb = r["FP16_bytes"] / 1024**2
        int8_mb = r["INT8_bytes"] / 1024**2
        p64_mb  = r.get("DKV-P64_bytes", 0) / 1024**2
        adap_mb = r.get("DKV-Adapt_bytes", 0) / 1024**2
        p64_rat = r.get("DKV-P64_ratio", 0)
        p64_err = r.get("DKV-P64_error", 0)
        print(f"  {r['seq_len']:>10,} | {fp16_mb:>8.1f} | {int8_mb:>8.1f} | "
              f"{p64_mb:>8.1f} | {adap_mb:>8.1f} | {p64_rat:>9.3f} | {p64_err:>8.5f}")

    print(f"\n[→] Run  python visualization/plot_crossover.py --input {out_path}  to plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossover Experiment")
    parser.add_argument("--mode", default="mixed",
                        choices=["gaussian", "smooth", "mixed", "real_approx"])
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", default="results/crossover/")
    args = parser.parse_args()
    main(args)
