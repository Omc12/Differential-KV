"""
experiments/exp_layer_analysis.py — Task 2

Analyzes compression compressibility per transformer layer.

Uses synthetic layer-varying KV tensors (smooth early, mixed mid, noisy late)
to mimic realistic transformer layer behavior.

Key outputs:
  - Per-layer compression profiles
  - Layer ranking by compressibility
  - Strategy recommendation per layer
  - Heatmaps of delta norms per layer × token position
"""

import sys
import json
import math
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.layer_analyzer import LayerAnalyzer
from anchor_logic.strategies import PeriodicAnchorStrategy
from anchor_logic.adaptive_policies import EMAPolicy, RollingVariancePolicy


NUM_LAYERS = 32
SEQ_LEN    = 8192
NUM_HEADS  = 32
HEAD_DIM   = 128


def build_delta_heatmap(
    kv_by_layer: dict,
    strategy,
    sample_tokens: int = 256,
) -> dict:
    """
    Build heatmap data: for each (layer, token_position) → delta RMS.
    Returns dict usable for plotting.
    """
    from anchor_logic.anchor_manager import AnchorManager

    heatmap = {}  # layer_idx -> list of (token_idx, rms)

    for layer_idx, kv in kv_by_layer.items():
        manager = AnchorManager(strategy=strategy)
        manager.compress(kv)

        row = []
        step = max(1, kv.shape[0] // sample_tokens)
        for t in range(0, kv.shape[0], step):
            if manager.is_anchor(t):
                row.append({"token": t, "rms": 0.0, "is_anchor": True})
            else:
                q_delta = manager.get_delta(t)
                if q_delta is not None:
                    anchor_idx, anchor_kv = manager.get_preceding_anchor(t)
                    from compression.quantization import dequantize_int8
                    delta = dequantize_int8(q_delta, target_dtype=torch.float32)
                    rms = (delta.norm() / math.sqrt(delta.numel())).item()
                    row.append({"token": t, "rms": round(rms, 5), "is_anchor": False})

        heatmap[layer_idx] = row

    return heatmap


def main():
    output_dir = Path("results/layer_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  LAYER-WISE COMPRESSIBILITY ANALYSIS")
    print(f"  layers={NUM_LAYERS} | seq_len={SEQ_LEN:,} | heads={NUM_HEADS} | dim={HEAD_DIM}")
    print(f"{'='*65}\n")

    # Strategies to test
    strategies_to_test = {
        "Periodic-64": PeriodicAnchorStrategy(interval=64),
        "EMA-balanced": EMAPolicy(alpha=0.1, sensitivity_factor=2.5,
                                   max_interval=256, min_interval=8),
        "Rolling-k3":   RollingVariancePolicy(k=3.0, window_size=64,
                                               max_interval=256, min_interval=8),
    }

    all_results = {}

    for strat_label, strategy in strategies_to_test.items():
        print(f"[Strategy: {strat_label}]")
        analyzer = LayerAnalyzer(strategy=strategy)

        # Generate synthetic layer-varying KV
        kv_by_layer = analyzer.generate_synthetic_layers(
            num_layers=NUM_LAYERS,
            seq_len=SEQ_LEN,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            seed=42,
        )

        # Analyze all layers
        profiles = analyzer.analyze_all_layers(kv_by_layer, num_recon_queries=15)
        recommendations = analyzer.recommend_strategies(profiles)

        print(f"  {'Rank':>4} {'Layer':>6} {'Ratio':>7} {'Density':>9} "
              f"{'MeanErr':>9} {'Smooth':>9} {'Rec.':>12}")
        print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*12}")
        for p in profiles[:10]:   # show top 10
            print(
                f"  {p.compressibility_rank:>4} {p.layer_idx:>6} "
                f"{p.compression_ratio:>7.3f} "
                f"{p.anchor_density:>9.4f} "
                f"{p.mean_recon_error:>9.5f} "
                f"{p.smoothness_score:>9.5f} "
                f"{recommendations.get(p.layer_idx, '?'):>12}"
            )
        print()

        # Build delta heatmap for this strategy
        heatmap = build_delta_heatmap(kv_by_layer, strategy)

        all_results[strat_label] = {
            "profiles":        [p.to_dict() for p in profiles],
            "recommendations": {str(k): v for k, v in recommendations.items()},
            "heatmap_sample":  {str(k): v[:20] for k, v in heatmap.items()},  # truncate
        }

    # Save
    out_path = output_dir / "layer_profiles.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Layer profiles -> {out_path}")

    # Summary: which layers are most/least compressible?
    print(f"\n{'='*65}")
    print("  LAYER COMPRESSIBILITY RANKING (Periodic-64 strategy)")
    print(f"{'='*65}")

    periodic_profiles = all_results.get("Periodic-64", {}).get("profiles", [])
    best5  = periodic_profiles[:5]
    worst5 = periodic_profiles[-5:]
    print(f"\n  Most compressible layers:")
    for p in best5:
        print(f"    Layer {p['layer_idx']:>2}: ratio={p['compression_ratio']:.3f}  "
              f"err={p['mean_recon_error']:.5f}  smooth={p['smoothness_score']:.5f}")
    print(f"\n  Least compressible layers:")
    for p in worst5:
        print(f"    Layer {p['layer_idx']:>2}: ratio={p['compression_ratio']:.3f}  "
              f"err={p['mean_recon_error']:.5f}  smooth={p['smoothness_score']:.5f}")

    print(f"\n[->] Run visualization/plot_layer_analysis.py to generate heatmaps")


if __name__ == "__main__":
    main()
