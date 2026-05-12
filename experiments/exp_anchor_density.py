"""
experiments/exp_anchor_density.py

Experiment: How does anchor frequency affect compression vs quality?

Sweeps anchor interval from 8 to 512 tokens across all KV modes.

Key tradeoffs:
  - Fewer anchors → smaller storage → longer delta chains → more error
  - More anchors  → larger storage → shorter chains   → less error

Find the "sweet spot" where compression is strong but error stays bounded.

Adaptive strategy also analyzed: how often does it trigger early anchors?
"""

import sys
import json
import random
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy, AdaptiveAnchorStrategy
from reconstruction.reconstructor import KVReconstructor


NUM_HEADS  = 32
HEAD_DIM   = 128
SEQ_LEN    = 32768
MODES      = ["gaussian", "smooth", "mixed", "real_approx"]
INTERVALS  = [8, 16, 32, 64, 128, 256, 512]


def sweep_anchor_interval(kv, mode_label):
    rows = []
    random.seed(0)

    for interval in INTERVALS:
        strategy = PeriodicAnchorStrategy(interval=interval)
        manager  = AnchorManager(strategy=strategy)
        stats    = manager.compress(kv)
        recon    = KVReconstructor(manager)

        # Measure error over 30 random windows
        errors = []
        window = min(64, interval)
        for _ in range(30):
            s = random.randint(0, SEQ_LEN - window - 1)
            e = s + window - 1
            err = recon.measure_error(kv, s, e)
            errors.append(err["mean_relative"])

        mean_err = sum(errors) / len(errors)
        max_err  = max(errors)

        rows.append({
            "mode":              mode_label,
            "interval":          interval,
            "compression_ratio": round(stats.compression_ratio, 4),
            "anchor_density":    round(stats.anchor_density, 4),
            "mean_error":        round(mean_err, 6),
            "max_error":         round(max_err, 6),
            "compressed_KB":     round(stats.total_compressed_bytes / 1024, 1),
        })

    return rows


def analyze_adaptive(kv, mode_label):
    """Analyze what triggers adaptive anchors and their frequency."""
    thresholds = [
        (1.0, 0.02),
        (2.0, 0.05),
        (4.0, 0.10),
    ]
    rows = []
    random.seed(0)

    for dn_thresh, err_thresh in thresholds:
        strategy = AdaptiveAnchorStrategy(
            max_interval=64,
            delta_norm_threshold=dn_thresh,
            error_estimate_threshold=err_thresh,
            min_interval=8,
        )
        manager = AnchorManager(strategy=strategy)
        stats   = manager.compress(kv)

        reasons = stats.anchor_reasons
        errors  = []
        window  = 64
        recon   = KVReconstructor(manager)
        for _ in range(30):
            s = random.randint(0, SEQ_LEN - window - 1)
            e = s + window - 1
            err = recon.measure_error(kv, s, e)
            errors.append(err["mean_relative"])

        rows.append({
            "mode":               mode_label,
            "dn_threshold":       dn_thresh,
            "err_threshold":      err_thresh,
            "compression_ratio":  round(stats.compression_ratio, 4),
            "anchor_density":     round(stats.anchor_density, 4),
            "periodic_triggers":  reasons.get("periodic", 0),
            "magnitude_triggers": reasons.get("adaptive_magnitude", 0),
            "error_triggers":     reasons.get("adaptive_error", 0),
            "mean_error":         round(sum(errors)/len(errors), 6),
        })

    return rows


def main():
    output_dir = Path("results/anchor_density")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = KVGenerator(num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42)
    all_periodic  = []
    all_adaptive  = []

    print(f"\n{'='*70}")
    print("  ANCHOR DENSITY EXPERIMENT")
    print(f"  seq_len={SEQ_LEN:,} | heads={NUM_HEADS} | head_dim={HEAD_DIM}")
    print(f"{'='*70}\n")

    for mode in MODES:
        kv = gen.generate(SEQ_LEN, mode=mode)

        print(f"[Mode: {mode}]")
        print(f"  {'interval':>8} | {'ratio':>7} | {'density':>9} | {'mean_err':>9} | {'max_err':>9}")
        print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

        periodic_rows = sweep_anchor_interval(kv, mode)
        for r in periodic_rows:
            print(
                f"  {r['interval']:>8} | {r['compression_ratio']:>7.3f} | "
                f"{r['anchor_density']:>9.4f} | {r['mean_error']:>9.5f} | "
                f"{r['max_error']:>9.5f}"
            )
        all_periodic.extend(periodic_rows)

        print(f"\n  [Adaptive triggers — mode={mode}]")
        print(f"  {'dn_thresh':>10} | {'err_thresh':>10} | {'ratio':>7} | "
              f"{'periodic':>8} | {'magnitude':>9} | {'err_trig':>8} | {'mean_err':>9}")
        adapt_rows = analyze_adaptive(kv, mode)
        for r in adapt_rows:
            print(
                f"  {r['dn_threshold']:>10.1f} | {r['err_threshold']:>10.3f} | "
                f"{r['compression_ratio']:>7.3f} | {r['periodic_triggers']:>8} | "
                f"{r['magnitude_triggers']:>9} | {r['error_triggers']:>8} | "
                f"{r['mean_error']:>9.5f}"
            )
        all_adaptive.extend(adapt_rows)
        print()

    # Save
    with open(output_dir / "periodic_sweep.json", "w") as f:
        json.dump(all_periodic, f, indent=2)
    with open(output_dir / "adaptive_analysis.json", "w") as f:
        json.dump(all_adaptive, f, indent=2)

    print(f"[OK] Anchor density results saved → {output_dir}/")


if __name__ == "__main__":
    main()
