"""
experiments/exp_delta_analysis.py — Task 4: Delta Distribution Analysis
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
from analysis.delta_analyzer import DeltaAnalyzer


SEQ_LEN   = 8192
NUM_HEADS = 32
HEAD_DIM  = 128
MODES     = ["gaussian", "smooth", "mixed", "real_approx"]
INTERVAL  = 64


def _interpret(summary, temporal, lowrank):
    hints = []
    if summary.get("sparsity_001", 0) > 0.5:
        hints.append("SPARSE: INT4/threshold quantization promising")
    elif summary.get("sparsity_01", 0) > 0.7:
        hints.append("MODERATELY SPARSE: INT8 good, INT4 possible")
    else:
        hints.append("DENSE: INT4 unlikely to help much")

    r8 = summary.get("svd_rank8_energy", 0)
    if r8 > 0.9:
        hints.append("LOW-RANK (rank-8 > 90%): LoRA-style very promising")
    elif r8 > 0.7:
        hints.append("PARTIALLY LOW-RANK (rank-8 > 70%): worth exploring")
    else:
        hints.append("FULL-RANK: low-rank compression expensive")

    if summary.get("kurtosis", 0) > 3.0:
        hints.append("HEAVY-TAILED: outlier-aware quantization recommended")
    if temporal.get("autocorrelation_lag1", 0) > 0.5:
        hints.append("TEMPORALLY CORRELATED: predictive coding may help")
    return " | ".join(hints) if hints else "No clear signal"


def main():
    output_dir = Path("results/delta_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen      = KVGenerator(num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42)
    analyzer = DeltaAnalyzer()
    strategy = PeriodicAnchorStrategy(interval=INTERVAL)

    print(f"\n{'='*70}")
    print("  DELTA DISTRIBUTION ANALYSIS")
    print(f"  seq_len={SEQ_LEN:,} | interval={INTERVAL} | heads={NUM_HEADS}")
    print(f"{'='*70}\n")

    all_results = {}

    for mode in MODES:
        kv      = gen.generate(SEQ_LEN, mode=mode)
        manager = AnchorManager(strategy=strategy)
        manager.compress(kv)
        anchor_positions = manager.index_list

        stats    = analyzer.analyze(kv, anchor_positions)
        temporal = analyzer.analyze_temporal_smoothness(kv)
        summary  = stats.summary()

        # Low-rank error sweep
        deltas = []
        last_anchor_kv = None
        for i in range(SEQ_LEN):
            kv_i = kv[i].float()
            if manager.is_anchor(i):
                last_anchor_kv = kv_i
            else:
                if last_anchor_kv is not None:
                    deltas.append((kv_i - last_anchor_kv).flatten())
                    if len(deltas) >= 512:
                        break

        lowrank_errors = {}
        if len(deltas) >= 10:
            delta_mat = torch.stack(deltas[:512])
            try:
                U, S, Vh = torch.linalg.svd(delta_mat, full_matrices=False)
                total_e  = (S**2).sum().item()
                for r in [1, 2, 4, 8, 16, 32]:
                    if r <= len(S):
                        approx   = (U[:, :r] * S[:r]) @ Vh[:r, :]
                        err      = (delta_mat - approx).norm() / (delta_mat.norm() + 1e-9)
                        retained = (S[:r]**2).sum().item() / (total_e + 1e-9)
                        lowrank_errors[f"rank_{r}"] = {
                            "recon_error":     round(err.item(), 5),
                            "energy_retained": round(retained, 4),
                        }
            except Exception as e:
                lowrank_errors["error"] = str(e)

        result = {
            "mode": mode, "seq_len": SEQ_LEN, "interval": INTERVAL,
            "num_deltas": stats.num_deltas,
            "distribution": summary, "temporal": temporal,
            "lowrank": lowrank_errors,
            "interpretation": _interpret(summary, temporal, lowrank_errors),
        }
        all_results[mode] = result

        print(f"[{mode}]")
        print(f"  RMS mean/p95 : {summary['rms_mean']:.4f} / {summary['rms_p95']:.4f}")
        print(f"  Sparsity 0.01: {summary['sparsity_001']:.3f}  Kurtosis: {summary['kurtosis']:.3f}")
        print(f"  SVD rank-1/8 : {summary['svd_rank1_energy']:.4f} / {summary['svd_rank8_energy']:.4f}")
        print(f"  Temporal corr: {temporal.get('autocorrelation_lag1', 0):.4f}")
        if lowrank_errors:
            for rk, v in list(lowrank_errors.items())[:4]:
                if isinstance(v, dict):
                    print(f"  {rk:<12}: err={v['recon_error']:.4f}  "
                          f"energy={v['energy_retained']:.4f}")
        print(f"  => {result['interpretation']}\n")

    out_path = output_dir / "delta_statistics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Delta analysis -> {out_path}")
    print("[->] Run visualization/plot_delta_analysis.py to visualize")


if __name__ == "__main__":
    main()
