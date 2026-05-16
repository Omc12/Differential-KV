"""
experiments/exp_quant_ablation.py — Phase 3 Stage F

Quantization research for KV deltas.
Compares NF4, BlockWise, OutlierAware, etc. on KV deltas.
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
from compression.quantization_advanced import compare_schemes

# Config
SEQ_LEN = 1024
HEADS = 32
DIM = 128
MODE = "mixed"

def run_quant_ablation():
    output_dir = Path("results/quantization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gen = KVGenerator(num_heads=HEADS, head_dim=DIM, seed=42)
    kv = gen.generate(SEQ_LEN, mode=MODE)
    
    print(f"\n{'='*80}")
    print(f"  STAGE F: QUANTIZATION RESEARCH")
    print(f"  seq_len={SEQ_LEN} | mode={MODE}")
    print(f"{'='*80}\n")

    # 1. Collect deltas from a periodic anchor strategy (interval=64)
    interval = 64
    deltas = []
    for i in range(0, SEQ_LEN, interval):
        anchor_kv = kv[i].float()
        for t in range(i + 1, min(i + interval, SEQ_LEN)):
            deltas.append((kv[t].float() - anchor_kv).flatten())
    
    D = torch.stack(deltas)
    
    # 2. Compare schemes
    print(f"Comparing quantization schemes on {D.shape[0]} deltas (dim={D.shape[1]})...")
    results = compare_schemes(D)
    
    print(f"\n{'Scheme':<20} | {'Error (RMS)':<12} | {'Ratio':<8} | {'Bytes':<10}")
    print("-" * 60)
    for name, stats in results.items():
        if "error_msg" in stats:
            print(f"{name:<20} | ERROR: {stats['error_msg']}")
        else:
            print(f"{name:<20} | {stats['error']:<12.6f} | {stats['ratio']:<8.2f}x | {stats['nbytes']:<10}")

    # 3. Low-Rank + Quantization interaction
    print(f"\n--- Low-Rank + Quantization Interaction ---")
    # We take rank-8 approximation and then quantize the U matrix (coords)
    from compression.lowrank import compress_lowrank, decompress_lowrank
    rank = 8
    lr = compress_lowrank(D, rank)
    
    # Quantize U (currently float16) to INT8 or NF4
    u_tensor = lr.U.float()
    u_results = compare_schemes(u_tensor)
    
    print(f"Quantizing U matrix (rank={rank}):")
    for name, stats in u_results.items():
        if "error" in stats and stats["error"] != -1:
            print(f"  {name:<20}: err={stats['error']:.6f}")

    # Save
    with open(output_dir / "quant_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_dir / 'quant_ablation_results.json'}")

if __name__ == "__main__":
    run_quant_ablation()
