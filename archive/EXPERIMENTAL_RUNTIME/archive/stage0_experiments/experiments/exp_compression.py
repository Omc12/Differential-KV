"""
experiments/exp_compression.py

Experiment: What compression ratios are achievable across KV modes and anchor strategies?

Sweeps:
  - KV mode: gaussian, smooth, mixed, real_approx
  - Anchor strategy: Periodic (32/64/128) + Adaptive
  - Sequence length: 16k tokens (fixed — compression ratio is length-stable)

Outputs:
  - Compression ratio per (mode, strategy)
  - Anchor density per (mode, strategy)
  - Delta norm statistics (mean, std, max)
  - Error after reconstruction

Key question:
  "Does smooth KV actually compress better than random?"
  (It should. If not, the whole delta idea is wrong.)
"""

import sys
import json
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy, AdaptiveAnchorStrategy
from reconstruction.reconstructor import KVReconstructor


SEQ_LEN   = 16384
NUM_HEADS = 32
HEAD_DIM  = 128
MODES     = ["gaussian", "smooth", "mixed", "real_approx"]

STRATEGIES = {
    "Periodic-32":   PeriodicAnchorStrategy(interval=32),
    "Periodic-64":   PeriodicAnchorStrategy(interval=64),
    "Periodic-128":  PeriodicAnchorStrategy(interval=128),
    "Adaptive":      AdaptiveAnchorStrategy(
        max_interval=64, delta_norm_threshold=2.0,
        error_estimate_threshold=0.05, min_interval=8
    ),
}


def run_compression_experiment():
    output_dir = Path("results/compression")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = KVGenerator(num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42)
    results = []

    print(f"\n{'='*70}")
    print("  COMPRESSION EXPERIMENT")
    print(f"  seq_len={SEQ_LEN:,} | heads={NUM_HEADS} | head_dim={HEAD_DIM}")
    print(f"{'='*70}\n")

    header = f"{'Mode':<14} {'Strategy':<14} {'Ratio':>7} {'Density':>9} {'MeanErr':>9} {'ΔNorm μ':>9} {'ΔNorm σ':>9}"
    print(header)
    print("-" * len(header))

    for mode in MODES:
        kv = gen.generate(SEQ_LEN, mode=mode)

        for strat_label, strategy in STRATEGIES.items():
            manager = AnchorManager(strategy=strategy)
            stats = manager.compress(kv)

            # Measure reconstruction error over a sample of windows
            recon = KVReconstructor(manager)
            import random; random.seed(1)
            errors = []
            window = 128
            for _ in range(30):
                start = random.randint(0, SEQ_LEN - window - 1)
                end   = start + window - 1
                err   = recon.measure_error(kv, start, end)
                errors.append(err["mean_relative"])

            mean_err  = sum(errors) / len(errors)

            # Delta norm statistics
            delta_norms = stats.delta_norms
            if delta_norms:
                import statistics
                dn_mean = statistics.mean(delta_norms)
                dn_std  = statistics.stdev(delta_norms) if len(delta_norms) > 1 else 0.0
            else:
                dn_mean = dn_std = 0.0

            row = {
                "mode":           mode,
                "strategy":       strat_label,
                "compression_ratio": round(stats.compression_ratio, 4),
                "anchor_density":    round(stats.anchor_density, 4),
                "mean_recon_error":  round(mean_err, 6),
                "delta_norm_mean":   round(dn_mean, 4),
                "delta_norm_std":    round(dn_std, 4),
                "num_anchors":       stats.num_anchors,
                "num_deltas":        stats.num_deltas,
                "anchor_reasons":    stats.anchor_reasons,
            }
            results.append(row)

            print(
                f"{mode:<14} {strat_label:<14} "
                f"{stats.compression_ratio:>7.3f} "
                f"{stats.anchor_density:>9.4f} "
                f"{mean_err:>9.5f} "
                f"{dn_mean:>9.3f} "
                f"{dn_std:>9.3f}"
            )

        print()  # blank line between modes

    out_path = output_dir / "compression_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Compression results saved → {out_path}")
    return results


if __name__ == "__main__":
    run_compression_experiment()
