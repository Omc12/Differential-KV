"""
experiments/exp_temporal_stability.py — Phase 3 Stage C

Measures temporal subspace drift, turnover, and cross-domain similarity.
Includes Layer-wise and Head-wise drift rates.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from analysis.subspace_tracker import SubspaceTracker
from analysis.layer_analyzer import LayerAnalyzer

# Config
SEQ_LEN = 4096
WINDOW_SIZE = 128
RANK = 8
DOMAINS = ["prose", "code", "reasoning", "repetitive"]

def run_stability_experiment():
    output_dir = Path("results/stability")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gen = KVGenerator(num_heads=32, head_dim=128, seed=42)
    tracker = SubspaceTracker(rank=RANK)
    
    print(f"\n{'='*80}")
    print(f"  STAGE C: TEMPORAL SUBSPACE STABILITY")
    print(f"  seq_len={SEQ_LEN} | window={WINDOW_SIZE} | rank={RANK}")
    print(f"{'='*80}\n")

    domain_deltas = {}
    stability_results = {}

    for domain in DOMAINS:
        print(f"--- Analyzing {domain} ---")
        mode_map = {"prose": "smooth", "code": "mixed", "reasoning": "mixed", "repetitive": "mixed"}
        kv = gen.generate(SEQ_LEN, mode=mode_map.get(domain, "mixed"))
        
        windows = []
        for i in range(0, SEQ_LEN, WINDOW_SIZE):
            end = min(i + WINDOW_SIZE, SEQ_LEN)
            if end - i < 10: continue
            anc = kv[i].float()
            win_deltas = [(kv[t].float() - anc).reshape(-1) for t in range(i + 1, end)]
            windows.append(torch.stack(win_deltas))
        
        metrics = tracker.analyze_temporal_drift(windows)
        stability_results[domain] = {
            "drift_rate": round(metrics.drift_rate, 6),
            "turnover_rate": round(metrics.turnover_rate, 6),
            "persistence": [round(p, 4) for p in metrics.persistence_curve]
        }
        
        # Collect sample deltas for cross-domain
        domain_deltas[domain] = torch.stack([(kv[t].float() - kv[t-1].float()).reshape(-1) for t in range(1, 100)])

    # 1. Layer-wise Drift Rates
    print(f"\n--- Layer-wise Drift Rates ---")
    layer_analyzer = LayerAnalyzer()
    kv_layers = layer_analyzer.generate_synthetic_layers(num_layers=12, seq_len=1024)
    layer_drifts = {}
    for li, lkv in kv_layers.items():
        wins = []
        for i in range(0, 1024, WINDOW_SIZE):
            anc = lkv[i].float()
            wins.append(torch.stack([(lkv[t].float() - anc).reshape(-1) for t in range(i+1, min(i+WINDOW_SIZE, 1024))]))
        m = tracker.analyze_temporal_drift(wins)
        layer_drifts[li] = round(m.drift_rate, 6)
        if li % 4 == 0: print(f"  Layer {li:>2}: drift={m.drift_rate:.4f}")

    # 2. Head-wise Drift Rates
    print(f"\n--- Head-wise Drift Rates (mixed mode) ---")
    kv_mixed = gen.generate(2048, mode="mixed")
    head_drifts = {}
    for h in range(16): # Analyze 16 heads
        wins = []
        for i in range(0, 2048, WINDOW_SIZE):
            anc = kv_mixed[i, :, h, :].float()
            wins.append(torch.stack([(kv_mixed[t, :, h, :].float() - anc).flatten() for t in range(i+1, min(i+WINDOW_SIZE, 2048))]))
        m = tracker.analyze_temporal_drift(wins)
        head_drifts[h] = round(m.drift_rate, 6)
        if h % 4 == 0: print(f"  Head {h:>2}: drift={m.drift_rate:.4f}")

    # Cross-domain similarity
    cross_sim = tracker.cross_domain_similarity(domain_deltas, rank=RANK)

    # Save
    final_data = {
        "stability": stability_results,
        "cross_domain": cross_sim,
        "layer_drifts": layer_drifts,
        "head_drifts": head_drifts
    }
    with open(output_dir / "temporal_stability.json", "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_dir / 'temporal_stability.json'}")

if __name__ == "__main__":
    run_stability_experiment()
