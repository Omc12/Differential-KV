"""
quick_start.py

One-command end-to-end validation of the Phase 1 prototype.

Runs a lightweight version of all 3 experiments then prints a
summary table so you can verify the system is working in < 60 seconds.

Usage:
    python quick_start.py
    python quick_start.py --seq-lens 1024 4096 8192
"""

import sys
import argparse
import time
import random
from pathlib import Path

import torch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve()))

from benchmarks.kv_generator import KVGenerator
from benchmarks.baselines import BaselineRunner
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy, AdaptiveAnchorStrategy
from reconstruction.reconstructor import KVReconstructor
from profiling.profiler import ReconstructionProfiler, MemoryBandwidthEstimator


BANNER = """
+------------------------------------------------------------------+
|          Differential KV Cache -- Phase 1 Quick Start            |
|          Isolated Research Prototype * PyTorch only             |
+------------------------------------------------------------------+
"""

STRATEGIES = {
    "Periodic-64":  PeriodicAnchorStrategy(interval=64),
    "Periodic-128": PeriodicAnchorStrategy(interval=128),
    "Adaptive":     AdaptiveAnchorStrategy(
        max_interval=64, delta_norm_threshold=2.0,
        error_estimate_threshold=0.05, min_interval=8,
    ),
}


def run_quick(seq_len: int, mode: str, heads: int, head_dim: int):
    gen = KVGenerator(num_heads=heads, head_dim=head_dim, seed=42)
    bl  = BaselineRunner(num_heads=heads, head_dim=head_dim)
    kv  = gen.generate(seq_len, mode=mode)

    rows = []

    # ── Baselines ──────────────────────────────────────────────────────────────
    for key, result in bl.run_all(kv).items():
        rows.append({
            "strategy":    result.label,
            "seq_len":     seq_len,
            "bytes_KB":    f"{result.total_bytes/1024:.0f}",
            "ratio":       f"{result.compression_ratio:.2f}x",
            "mean_err":    f"{result.mean_relative_error:.5f}",
            "recon_ms":    "—",
            "tok/sec":     "—",
        })

    # ── Differential KV ────────────────────────────────────────────────────────
    random.seed(0)
    window = min(128, seq_len // 8)

    for label, strategy in STRATEGIES.items():
        manager = AnchorManager(strategy=strategy)
        t0 = time.perf_counter()
        stats = manager.compress(kv)
        compress_ms = (time.perf_counter() - t0) * 1000

        recon    = KVReconstructor(manager)
        profiler = ReconstructionProfiler()

        for _ in range(10):
            s = random.randint(0, max(0, seq_len - window - 1))
            e = min(s + window - 1, seq_len - 1)
            r = recon.reconstruct_range(s, e)
            err = recon.measure_error(kv, s, e)
            profiler.record_result(r, error=err["mean_relative"])

        summary = profiler.summarize()

        rows.append({
            "strategy":  f"DiffKV-{label}",
            "seq_len":   seq_len,
            "bytes_KB":  f"{stats.total_compressed_bytes/1024:.0f}",
            "ratio":     f"{stats.compression_ratio:.2f}x",
            "mean_err":  f"{summary.mean_error:.5f}",
            "recon_ms":  f"{summary.mean_latency_ms:.3f}",
            "tok/sec":   f"{summary.tokens_per_second:,.0f}",
        })

    return rows


def main(args):
    print(BANNER)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")
    print(f"  Mode   : {args.mode}")
    print(f"  Heads  : {args.heads} x head_dim={args.head_dim}")
    print()

    all_rows = []
    for seq_len in args.seq_lens:
        print(f"  Running seq_len = {seq_len:,} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        rows = run_quick(seq_len, args.mode, args.heads, args.head_dim)
        print(f"done ({time.perf_counter()-t0:.1f}s)")
        all_rows.extend(rows)

    print()
    print(tabulate(
        [[r[k] for k in ["strategy", "seq_len", "bytes_KB", "ratio",
                          "mean_err", "recon_ms", "tok/sec"]]
         for r in all_rows],
        headers=["Strategy", "Seq Len", "Bytes KB", "Ratio",
                 "Mean Err", "Recon ms", "Tok/sec"],
        tablefmt="grid",
    ))

    print("""
Next steps:
  python experiments/exp_crossover.py        <- The defining experiment
  python experiments/exp_compression.py      <- Compression ratios
  python experiments/exp_anchor_density.py   <- Sweet-spot analysis
  python visualization/plot_all.py           <- Generate all plots
  pytest tests/ -v                           <- Run all unit tests
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential KV Quick Start")
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[4096, 16384, 65536])
    parser.add_argument("--mode",     default="mixed",
                        choices=["gaussian", "smooth", "mixed", "real_approx"])
    parser.add_argument("--heads",    type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    main(args)
