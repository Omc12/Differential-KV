"""
experiments/exp_layer_selective.py — Phase 3 Stage B

Implements and tests heterogeneous rank schedules.
Compares global uniform rank vs layer-selective policies.
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
from analysis.layer_analyzer import LayerAnalyzer
from anchor_logic.layer_selector import LowRankScheduleSelector
from evaluation.generation_drift import GenerationDriftEvaluator

# Config
NUM_LAYERS = 24
SEQ_LEN = 2048
HEADS = 32
DIM = 128

def generate_schedules(num_layers: int):
    # Strategy A: early=rank-4, mid=rank-8, late=dense/fp16
    early_cut = int(num_layers * 0.25)
    late_cut = int(num_layers * 0.75)
    
    strat_a = []
    for i in range(num_layers):
        if i < early_cut: strat_a.append(4)
        elif i < late_cut: strat_a.append(8)
        else: strat_a.append("fp16")
    
    # Strategy B: early=low-rank (16), mid=standard DiffKV (rank-8), late=conservative periodic (dense)
    strat_b = []
    for i in range(num_layers):
        if i < early_cut: strat_b.append(16)
        elif i < late_cut: strat_b.append(8)
        else: strat_b.append("dense")
        
    # Strategy C: dynamic based on spectral decay (placeholder logic)
    strat_c = [8 if i % 2 == 0 else 4 for i in range(num_layers)]

    return {
        "strategy_a": strat_a,
        "strategy_b": strat_b,
        "strategy_c": strat_c
    }

def run_layer_selective_experiment():
    output_dir = Path("results/layer_selective")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = LayerAnalyzer()
    kv_layers = analyzer.generate_synthetic_layers(
        num_layers=NUM_LAYERS, seq_len=SEQ_LEN, num_heads=HEADS, head_dim=DIM
    )
    
    schedules = generate_schedules(NUM_LAYERS)
    results = {}

    print(f"\n{'='*80}")
    print(f"  STAGE B: LAYER-SELECTIVE LOW-RANK")
    print(f"  num_layers={NUM_LAYERS} | seq_len={SEQ_LEN}")
    print(f"{'='*80}\n")

    for name, sched in schedules.items():
        print(f"--- Running {name} ---")
        selector = LowRankScheduleSelector(NUM_LAYERS, rank_schedule=sched)
        
        total_error = 0
        layer_stats = {}
        
        for i in range(NUM_LAYERS):
            kv = kv_layers[i]
            strat = selector.get_strategy(i)
            
            if isinstance(strat, str) and strat.startswith("lowrank_"):
                from compression.lowrank import compress_kv_sequence_lowrank, decompress_kv_sequence_lowrank
                rank = int(strat.split("_")[1])
                anchor_positions = list(range(0, SEQ_LEN, 64))
                blocks, anchors = compress_kv_sequence_lowrank(kv, anchor_positions, rank)
                recon_kv = decompress_kv_sequence_lowrank(blocks, anchors, kv.shape)
                err = torch.norm(kv.float() - recon_kv.float()) / torch.norm(kv.float())
                total_error += err.item()
                layer_stats[i] = {"type": f"lowrank_{rank}", "error": round(err.item(), 6)}
            else:
                # dense/fp16
                layer_stats[i] = {"type": "dense", "error": 0.0}

        avg_err = total_error / NUM_LAYERS
        results[name] = {
            "avg_error": round(avg_err, 6),
            "layers": layer_stats,
            "schedule": sched
        }
        print(f"  {name}: Avg Error = {avg_err:.6f}")

    # Save
    with open(output_dir / "layer_selective_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_dir / 'layer_selective_stats.json'}")

if __name__ == "__main__":
    run_layer_selective_experiment()
