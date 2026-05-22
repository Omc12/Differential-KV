"""
experiments/exp_hybrid_repair.py — Phase 3 Stage E

Hybrid architecture combining low-rank deltas with sparse adaptive repair.
Tests whether sparse repair can stabilize low-rank reconstruction.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Add root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from compression.lowrank import compress_lowrank, decompress_lowrank

# Config
SEQ_LEN = 2048
RANK = 4
THRESHOLD = 0.1 # Relative error threshold for repair
INTERVAL = 128

def compress_hybrid(kv: torch.Tensor, interval: int, rank: int, threshold: float):
    """
    Compress KV sequence using low-rank deltas + sparse repair.
    """
    seq_len, _, heads, dim = kv.shape
    feat_dim = 2 * heads * dim
    
    anchors = list(range(0, seq_len, interval))
    blocks = {}
    repair_count = 0
    
    for i, start in enumerate(anchors):
        end = anchors[i+1] if i+1 < len(anchors) else seq_len
        anchor_kv = kv[start].float()
        
        # 1. Standard Low-Rank Compression for the block
        rows = []
        tok_indices = []
        for t in range(start + 1, end):
            rows.append((kv[t].float() - anchor_kv).reshape(-1))
            tok_indices.append(t)
        
        if not rows:
            blocks[start] = None
            continue
            
        D = torch.stack(rows)
        lr = compress_lowrank(D, rank)
        
        # 2. Identify tokens needing repair
        recon_D = decompress_lowrank(lr, dtype=torch.float32)
        errors = torch.norm(D - recon_D, dim=1) / (torch.norm(D, dim=1) + 1e-9)
        
        repair_tokens = {} # tok_idx -> full_precision_delta
        for j, err in enumerate(errors):
            if err > threshold:
                tok_idx = tok_indices[j]
                repair_tokens[tok_idx] = D[j].to(torch.float16)
                repair_count += 1
        
        blocks[start] = {
            "lowrank": lr,
            "repairs": repair_tokens,
            "tokens": tok_indices
        }
        
    return blocks, repair_count

def run_hybrid_experiment():
    output_dir = Path("results/hybrid")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gen = KVGenerator(num_heads=32, head_dim=128, seed=42)
    kv = gen.generate(SEQ_LEN, mode="mixed")
    
    print(f"\n{'='*80}")
    print(f"  STAGE E: HYBRID LOW-RANK + ADAPTIVE REPAIR")
    print(f"  rank={RANK} | interval={INTERVAL} | threshold={THRESHOLD}")
    print(f"{'='*80}\n")

    # Baseline: Low-Rank Only
    blocks_lr, _ = compress_hybrid(kv, INTERVAL, RANK, threshold=1e9) # never repair
    
    # Hybrid: Low-Rank + Repair
    blocks_hybrid, repair_count = compress_hybrid(kv, INTERVAL, RANK, THRESHOLD)
    
    # Calculate total error for both
    def calc_error(blocks):
        total_sq_err = 0
        total_norm = 0
        for start, data in blocks.items():
            if data is None: continue
            lr = data["lowrank"]
            repairs = data["repairs"]
            toks = data["tokens"]
            
            anchor_kv = kv[start].float()
            recon_D = decompress_lowrank(lr, dtype=torch.float32)
            
            for j, t in enumerate(toks):
                original = kv[t].float()
                if t in repairs:
                    recon = anchor_kv + repairs[t].float().reshape(2, 32, 128)
                else:
                    recon = anchor_kv + recon_D[j].reshape(2, 32, 128)
                
                total_sq_err += torch.norm(original - recon)**2
                total_norm += torch.norm(original)**2
        return torch.sqrt(total_sq_err / total_norm).item()

    err_lr = calc_error(blocks_lr)
    err_hybrid = calc_error(blocks_hybrid)
    
    print(f"  Low-Rank Only Error : {err_lr:.6f}")
    print(f"  Hybrid Error        : {err_hybrid:.6f}")
    print(f"  Repair tokens       : {repair_count} ({repair_count/SEQ_LEN:.2%})")

    # Memory estimate
    # LR: anchor + U + V
    # Hybrid: LR + repairs (FP16 deltas)
    # Each repair is 2 * heads * dim * 2 bytes
    repair_bytes = repair_count * (2 * 32 * 128 * 2)
    
    results = {
        "lowrank_only_error": err_lr,
        "hybrid_error": err_hybrid,
        "repair_count": repair_count,
        "repair_rate": repair_count / SEQ_LEN,
        "estimated_repair_overhead_bytes": repair_bytes
    }

    # Save
    with open(output_dir / "hybrid_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_dir / 'hybrid_stats.json'}")

if __name__ == "__main__":
    run_hybrid_experiment()
