"""
experiments/exp_lowrank_feasibility.py — Phase 2.5 Objective 3

Low-rank feasibility analysis across KV modes and transformer layers.

Key questions:
  1. Are deltas consistently low-rank or is this mode-specific?
  2. Do dominant subspace directions drift significantly over sequence?
  3. Are some heads substantially more compressible than others?
  4. At what rank does LoRA-style storage become smaller than INT8?

IMPORTANT: This is purely exploratory. No LoRA implementation.
"""

import sys
import json
import math
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy
from analysis.lowrank_analyzer import LowRankAnalyzer
from analysis.layer_analyzer import LayerAnalyzer


SEQ_LEN   = 4096
NUM_HEADS = 32
HEAD_DIM  = 128
MODES     = ["gaussian", "smooth", "mixed", "real_approx"]
INTERVAL  = 64


def main():
    output_dir = Path("results/lowrank_feasibility")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen      = KVGenerator(num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42)
    analyzer = LowRankAnalyzer()
    strategy = PeriodicAnchorStrategy(interval=INTERVAL)

    print(f"\n{'='*70}")
    print("  LOW-RANK FEASIBILITY ANALYSIS")
    print(f"  seq_len={SEQ_LEN:,} | anchor_interval={INTERVAL}")
    print(f"{'='*70}\n")

    all_results = {}

    for mode in MODES:
        kv = gen.generate(SEQ_LEN, mode=mode)

        manager = AnchorManager(strategy=strategy)
        manager.compress(kv)
        anchor_positions = manager.index_list

        profile  = analyzer.analyze(kv, anchor_positions, window_size=128)
        summary  = profile.summary()

        # Compute costs at various ranks
        cost_analysis = {}
        for rank in [1, 2, 4, 8, 16, 32]:
            cost = analyzer.estimate_lowrank_compute_cost(
                SEQ_LEN, NUM_HEADS, HEAD_DIM, rank
            )
            cost_analysis[f"rank_{rank}"] = cost

        all_results[mode] = {
            "profile":      summary,
            "compute_costs": cost_analysis,
        }

        # Print
        print(f"[{mode}]")
        print(f"  Rank for 90% energy: {summary['rank_for_90pct']}")
        print(f"  Rank for 99% energy: {summary['rank_for_99pct']}")
        print(f"  Mean subspace drift: {summary['mean_subspace_drift']:.4f}  "
              f"(std={summary['std_subspace_drift']:.4f})")
        print(f"  Mean head rank-1 energy: {summary['mean_head_rank1']:.4f}")

        print(f"  SVD energy retention:")
        for rank_str, energy in summary["energy_at_rank"].items():
            err_val = summary["error_at_rank"].get(rank_str, 0)
            print(f"    Rank {rank_str:>3}: energy={energy:.4f}  recon_err={err_val:.4f}")

        print(f"  Compute comparison (vs INT8):")
        for rk, cost in list(cost_analysis.items())[:4]:
            print(f"    {rk}: ratio={cost['ratio']}x  "
                  f"lr_smaller={cost['lr_smaller']}  "
                  f"breakeven={cost['breakeven_seq']}")

        # Verdict
        drift_ok = summary["mean_subspace_drift"] < 0.3
        rank_ok  = summary["rank_for_90pct"] <= 16
        verdict = []
        if rank_ok:
            verdict.append(f"LOW-RANK (90%@rank-{summary['rank_for_90pct']})")
        else:
            verdict.append(f"NOT LOW-RANK (needs rank-{summary['rank_for_90pct']} for 90%)")
        if drift_ok:
            verdict.append("STABLE subspace (low drift)")
        else:
            verdict.append("UNSTABLE subspace (high drift — LoRA risky)")
        print(f"  => {' | '.join(verdict)}\n")

    # ── Layer-wise low-rank analysis ─────────────────────────────────────────
    print("[Layer-wise low-rank structure (mixed mode)]")
    layer_analyzer = LayerAnalyzer()
    kv_by_layer    = layer_analyzer.generate_synthetic_layers(
        num_layers=16, seq_len=2048, num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42
    )
    layer_lr = {}
    for layer_idx, kv in kv_by_layer.items():
        mgr = AnchorManager(strategy=PeriodicAnchorStrategy(interval=INTERVAL))
        mgr.compress(kv)
        prof = analyzer.analyze(kv, mgr.index_list, window_size=64)
        layer_lr[layer_idx] = {
            "rank_for_90pct": prof.rank_for_90pct_energy,
            "rank_for_99pct": prof.rank_for_99pct_energy,
            "mean_drift":     round(float(np.mean(prof.subspace_drift)), 4)
                              if prof.subspace_drift else 0.0,
            "mean_head_rank1": round(float(np.mean(prof.per_head_rank1_energy)), 4)
                               if prof.per_head_rank1_energy else 0.0,
        }
        print(f"  Layer {layer_idx:>2}: 90%@rank={prof.rank_for_90pct_energy:<3}  "
              f"drift={layer_lr[layer_idx]['mean_drift']:.4f}  "
              f"head_rank1={layer_lr[layer_idx]['mean_head_rank1']:.4f}")

    all_results["layer_analysis"] = layer_lr

    # Save
    out_path = output_dir / "lowrank_feasibility.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results -> {out_path}")
    print("[->] Run visualization/plot_lowrank.py to visualize")


if __name__ == "__main__":
    main()
