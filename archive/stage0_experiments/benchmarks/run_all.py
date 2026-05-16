"""
benchmarks/run_all.py

Master benchmark runner.

Runs the full suite of comparisons across all sequence lengths,
KV modes, and anchor strategies.

Usage:
    python benchmarks/run_all.py [--seq-lens 4096 16384] [--mode mixed]
                                 [--layers 32] [--heads 32] [--head-dim 128]
                                 [--output results/]
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
from tabulate import tabulate

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator, KVMode
from benchmarks.baselines import BaselineRunner
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy, AdaptiveAnchorStrategy
from reconstruction.reconstructor import KVReconstructor
from profiling.profiler import ReconstructionProfiler, MemoryBandwidthEstimator


DEFAULT_SEQ_LENS = [4096, 16384, 65536, 131072]
DEFAULT_LAYERS   = 32
DEFAULT_HEADS    = 32
DEFAULT_HEAD_DIM = 128


def run_diff_kv_benchmark(
    kv: torch.Tensor,
    strategy,
    label: str,
    num_reconstruction_queries: int = 20,
):
    """
    Compress kv with strategy, then simulate random reconstruction queries.
    Returns a dict of metrics.
    """
    seq_len = kv.shape[0]
    manager = AnchorManager(strategy=strategy)

    # --- Compress ---
    t0 = time.perf_counter()
    stats = manager.compress(kv)
    compress_ms = (time.perf_counter() - t0) * 1000

    reconstructor = KVReconstructor(manager)
    profiler = ReconstructionProfiler()

    # --- Simulate reconstruction queries (attention-style windows) ---
    window_size = min(128, seq_len // 4)
    import random
    random.seed(0)

    total_error_sum = 0.0
    for _ in range(num_reconstruction_queries):
        start = random.randint(0, max(0, seq_len - window_size - 1))
        end = min(start + window_size - 1, seq_len - 1)

        result = reconstructor.reconstruct_range(start, end)
        err_metrics = reconstructor.measure_error(kv, start, end)
        profiler.record_result(result, error=err_metrics["mean_relative"])
        total_error_sum += err_metrics["mean_relative"]

    summary = profiler.summarize()

    return {
        "label": label,
        "seq_len": seq_len,
        "compress_ms": round(compress_ms, 3),
        "compressed_bytes": stats.total_compressed_bytes,
        "fp16_bytes": stats.original_fp16_bytes,
        "compression_ratio": round(stats.compression_ratio, 4),
        "anchor_density": round(stats.anchor_density, 4),
        "mean_recon_error": round(summary.mean_error, 6),
        "max_recon_error": round(summary.max_error, 6),
        "mean_recon_latency_ms": round(summary.mean_latency_ms, 4),
        "p95_recon_latency_ms": round(summary.p95_latency_ms, 4),
        "tokens_per_sec": round(summary.tokens_per_second, 1),
        "total_bytes_read": summary.total_bytes_read,
    }


def print_table(rows: List[dict], title: str):
    if not rows:
        return
    headers = list(rows[0].keys())
    table = [[r[h] for h in headers] for r in rows]
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(tabulate(table, headers=headers, tablefmt="rounded_outline", floatfmt=".4f"))


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lens: List[int] = args.seq_lens
    mode: KVMode = args.mode
    num_layers = args.layers

    print(f"\n{'#'*80}")
    print(f"  Differential KV — Full Benchmark Suite")
    print(f"  Mode: {mode} | Layers: {num_layers} | Heads: {args.heads} | HeadDim: {args.head_dim}")
    print(f"  Sequence lengths: {seq_lens}")
    print(f"{'#'*80}")

    generator = KVGenerator(
        num_heads=args.heads,
        head_dim=args.head_dim,
        dtype=torch.float16,
        seed=42,
    )
    baseline_runner = BaselineRunner(num_heads=args.heads, head_dim=args.head_dim)
    bw_estimator = MemoryBandwidthEstimator(
        num_layers=num_layers, num_heads=args.heads, head_dim=args.head_dim
    )

    all_results = []

    strategies = [
        ("Periodic-32",  PeriodicAnchorStrategy(interval=32)),
        ("Periodic-64",  PeriodicAnchorStrategy(interval=64)),
        ("Periodic-128", PeriodicAnchorStrategy(interval=128)),
        ("Adaptive",     AdaptiveAnchorStrategy(
            max_interval=64, delta_norm_threshold=2.0,
            error_estimate_threshold=0.05, min_interval=8
        )),
    ]

    for seq_len in seq_lens:
        print(f"\n[SEQ LEN = {seq_len:,}] Generating KV tensors...")
        kv = generator.generate(seq_len, mode=mode)

        row_group = []

        # --- Baselines ---
        bl = baseline_runner.run_all(kv)
        for key, result in bl.items():
            row_group.append({
                "strategy": result.label,
                "seq_len": seq_len,
                "bytes_KB": round(result.total_bytes / 1024, 1),
                "ratio": round(result.compression_ratio, 3),
                "mean_err": round(result.mean_relative_error, 6),
                "enc_ms": round(result.encode_ms, 3),
                "dec_ms": round(result.decode_ms, 3),
                "recon_ms": "-",
                "tok/sec": "-",
            })

        # --- Differential KV strategies ---
        for strat_label, strategy in strategies:
            r = run_diff_kv_benchmark(kv, strategy, strat_label,
                                      num_reconstruction_queries=args.queries)
            row_group.append({
                "strategy": f"DiffKV-{strat_label}",
                "seq_len": seq_len,
                "bytes_KB": round(r["compressed_bytes"] / 1024, 1),
                "ratio": r["compression_ratio"],
                "mean_err": r["mean_recon_error"],
                "enc_ms": r["compress_ms"],
                "dec_ms": "-",
                "recon_ms": r["mean_recon_latency_ms"],
                "tok/sec": r["tokens_per_sec"],
            })
            all_results.append(r)

        print_table(row_group, f"Results @ seq_len={seq_len:,} [{mode}]")

    # --- Save JSON ---
    out_file = output_dir / f"benchmark_{mode}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")

    # --- Bandwidth summary ---
    print(f"\n{'='*80}")
    print("  Memory Bandwidth Estimates (all seq lengths, theoretical)")
    print(f"{'='*80}")
    bw_rows = []
    for sl in seq_lens:
        ests = bw_estimator.compare_all(sl, anchor_density=0.02)
        for key, est in ests.items():
            bw_rows.append({
                "config": est.label,
                "seq_len": sl,
                "total_MB": round(est.mb, 2),
                "ratio": round(est.compression_ratio, 3),
            })
    print_table(bw_rows, "Bandwidth Model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential KV Full Benchmark")
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=DEFAULT_SEQ_LENS)
    parser.add_argument("--mode", type=str, default="mixed",
                        choices=["gaussian", "smooth", "mixed", "real_approx"])
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--queries", type=int, default=20,
                        help="Number of random reconstruction queries per seq_len")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()
    main(args)
