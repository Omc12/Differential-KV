"""
experiments/exp_adaptive_policies.py — Task 1

Compares all 5 Phase 2 adaptive anchor policies across KV modes.

Key questions:
  1. Which policy achieves lowest anchor density at acceptable error?
  2. What is the compression-vs-error Pareto frontier per policy?
  3. How does each policy behave differently on smooth vs noisy KV?

Root cause of Phase 1 failure (documented here):
  Raw L2 norm of [2, 32, 128] FP16 tensor ~ 90 for unit-normal.
  Any threshold < 90 triggers every token.
  Fix: all Phase 2 policies normalize by sqrt(numel) or use relative measures.
"""

import sys
import json
import random
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy
from anchor_logic.adaptive_policies import (
    AbsoluteNormalizedPolicy, RelativeChangePolicy,
    RollingVariancePolicy, EMAPolicy, LayerNormalizedPolicy,
    make_policy, POLICY_PRESETS,
)
from reconstruction.reconstructor import KVReconstructor
from profiling.profiler import ReconstructionProfiler


SEQ_LEN   = 16384
NUM_HEADS = 32
HEAD_DIM  = 128
MODES     = ["gaussian", "smooth", "mixed", "real_approx"]


def policy_sweep():
    """
    Sweep all policies across multiple hyperparameter configs and KV modes.
    Returns list of result dicts.
    """
    policies_to_test = [
        # Periodic baselines
        ("Periodic-32",   PeriodicAnchorStrategy(interval=32)),
        ("Periodic-64",   PeriodicAnchorStrategy(interval=64)),
        ("Periodic-128",  PeriodicAnchorStrategy(interval=128)),
        ("Periodic-256",  PeriodicAnchorStrategy(interval=256)),

        # Phase 2 adaptive policies
        ("AbsNorm-0.2",   AbsoluteNormalizedPolicy(threshold=0.2,  max_interval=256, min_interval=8)),
        ("AbsNorm-0.4",   AbsoluteNormalizedPolicy(threshold=0.4,  max_interval=256, min_interval=8)),
        ("AbsNorm-0.8",   AbsoluteNormalizedPolicy(threshold=0.8,  max_interval=512, min_interval=8)),

        ("RelChange-0.10", RelativeChangePolicy(threshold=0.10, max_interval=256, min_interval=8)),
        ("RelChange-0.20", RelativeChangePolicy(threshold=0.20, max_interval=256, min_interval=8)),
        ("RelChange-0.40", RelativeChangePolicy(threshold=0.40, max_interval=512, min_interval=16)),

        ("Rolling-k2",    RollingVariancePolicy(k=2.0, window_size=64, max_interval=256, min_interval=8)),
        ("Rolling-k3",    RollingVariancePolicy(k=3.0, window_size=64, max_interval=512, min_interval=8)),
        ("Rolling-k4",    RollingVariancePolicy(k=4.0, window_size=128, max_interval=512, min_interval=16)),

        ("EMA-2.0",       EMAPolicy(alpha=0.1, sensitivity_factor=2.0, max_interval=256, min_interval=8)),
        ("EMA-2.5",       EMAPolicy(alpha=0.1, sensitivity_factor=2.5, max_interval=256, min_interval=8)),
        ("EMA-3.5",       EMAPolicy(alpha=0.1, sensitivity_factor=3.5, max_interval=512, min_interval=8)),

        ("LayerNorm-0.3", LayerNormalizedPolicy(threshold=0.3, max_interval=256, min_interval=8)),
        ("LayerNorm-0.5", LayerNormalizedPolicy(threshold=0.5, max_interval=256, min_interval=8)),
        ("LayerNorm-0.8", LayerNormalizedPolicy(threshold=0.8, max_interval=512, min_interval=16)),
    ]

    gen = KVGenerator(num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42)
    results = []
    random.seed(0)

    print(f"\n{'='*75}")
    print("  ADAPTIVE POLICY SWEEP — Phase 2")
    print(f"  seq_len={SEQ_LEN:,} | heads={NUM_HEADS} | head_dim={HEAD_DIM}")
    print(f"{'='*75}\n")

    header = f"{'Mode':<12} {'Policy':<16} {'Ratio':>7} {'Density':>9} {'MeanErr':>9} {'ReconMs':>8}"
    print(header)
    print("-" * len(header))

    window = 128
    n_queries = 20

    for mode in MODES:
        kv = gen.generate(SEQ_LEN, mode=mode)

        for policy_label, policy in policies_to_test:
            manager = AnchorManager(strategy=policy)
            t0 = time.perf_counter()
            stats = manager.compress(kv)
            compress_ms = (time.perf_counter() - t0) * 1000

            recon    = KVReconstructor(manager)
            profiler = ReconstructionProfiler()
            errors   = []

            for _ in range(n_queries):
                s = random.randint(0, max(0, SEQ_LEN - window - 1))
                e = min(s + window - 1, SEQ_LEN - 1)
                r = recon.reconstruct_range(s, e)
                err = recon.measure_error(kv, s, e)
                profiler.record_result(r, error=err["mean_relative"])
                errors.append(err["mean_relative"])

            summary = profiler.summarize()

            # Get policy-specific stats if available
            policy_stats = {}
            if hasattr(policy, "get_stats"):
                policy_stats = policy.get_stats()

            row = {
                "mode":              mode,
                "policy":            policy_label,
                "compression_ratio": round(stats.compression_ratio, 4),
                "anchor_density":    round(stats.anchor_density, 4),
                "mean_recon_error":  round(summary.mean_error, 6),
                "max_recon_error":   round(summary.max_error, 6),
                "mean_recon_ms":     round(summary.mean_latency_ms, 4),
                "compress_ms":       round(compress_ms, 3),
                "num_anchors":       stats.num_anchors,
                "anchor_reasons":    stats.anchor_reasons,
                "policy_stats":      policy_stats,
            }
            results.append(row)

            print(
                f"{mode:<12} {policy_label:<16} "
                f"{stats.compression_ratio:>7.3f} "
                f"{stats.anchor_density:>9.4f} "
                f"{summary.mean_error:>9.5f} "
                f"{summary.mean_latency_ms:>8.3f}"
            )

        print()  # blank between modes

    return results


def find_pareto_frontier(results, mode: str):
    """
    Find the Pareto frontier of (compression_ratio, -mean_recon_error)
    for a given mode. Returns policies that are not dominated.
    """
    mode_results = [r for r in results if r["mode"] == mode]
    pareto = []
    for r in mode_results:
        dominated = False
        for other in mode_results:
            if (other["compression_ratio"] >= r["compression_ratio"] and
                other["mean_recon_error"]  <= r["mean_recon_error"] and
                other != r):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return sorted(pareto, key=lambda x: x["compression_ratio"], reverse=True)


def main():
    output_dir = Path("results/adaptive_policies")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = policy_sweep()

    # Save full results
    out_path = output_dir / "policy_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Sweep results -> {out_path}")

    # Print Pareto analysis per mode
    print(f"\n{'='*75}")
    print("  PARETO FRONTIER (best compression at given error budget)")
    print(f"{'='*75}")

    pareto_all = {}
    for mode in MODES:
        pareto = find_pareto_frontier(results, mode)
        pareto_all[mode] = [
            {"policy": p["policy"], "ratio": p["compression_ratio"],
             "error": p["mean_recon_error"], "density": p["anchor_density"]}
            for p in pareto
        ]
        print(f"\n  [{mode}]")
        for p in pareto:
            print(f"    {p['policy']:<18} ratio={p['compression_ratio']:.3f}  "
                  f"error={p['mean_recon_error']:.5f}  density={p['anchor_density']:.4f}")

    with open(output_dir / "pareto_frontier.json", "w") as f:
        json.dump(pareto_all, f, indent=2)

    print(f"\n[OK] Pareto analysis -> {output_dir}/pareto_frontier.json")
    print(f"\n[->] Run visualization/plot_adaptive.py to generate plots")


if __name__ == "__main__":
    main()
