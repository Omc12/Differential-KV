"""
experiments/exp_headwise_analysis.py — Phase 3 Stage D

Investigates head-wise low-rank structure.
Determines whether compression should be head-selective.
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
from analysis.lowrank_analyzer import LowRankAnalyzer

# Config
SEQ_LEN = 2048
HEADS = 32
DIM = 128
MODE = "mixed"

def run_headwise_experiment():
    output_dir = Path("results/headwise")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gen = KVGenerator(num_heads=HEADS, head_dim=DIM, seed=42)
    kv = gen.generate(SEQ_LEN, mode=MODE)
    
    analyzer = LowRankAnalyzer()
    
    print(f"\n{'='*80}")
    print(f"  STAGE D: HEAD-WISE LOW-RANK STRUCTURE")
    print(f"  heads={HEADS} | seq_len={SEQ_LEN} | mode={MODE}")
    print(f"{'='*80}\n")

    # We'll use a fixed anchor interval of 64
    anchor_positions = list(range(0, SEQ_LEN, 64))
    
    # We'll modify LowRankAnalyzer slightly to get more detailed per-head stats if needed,
    # but the existing one already does some per-head analysis.
    # Let's perform a more thorough sweep here.
    
    head_results = []
    for h in range(HEADS):
        # Extract head deltas
        head_deltas = []
        for i in range(len(anchor_positions)):
            start = anchor_positions[i]
            next_anc = anchor_positions[i+1] if i+1 < len(anchor_positions) else SEQ_LEN
            anchor_kv = kv[start, :, h, :].float()
            for t in range(start + 1, next_anc):
                delta = (kv[t, :, h, :].float() - anchor_kv).flatten()
                head_deltas.append(delta)
        
        D_h = torch.stack(head_deltas)
        U, S, Vh = torch.linalg.svd(D_h, full_matrices=False)
        total_energy = (S**2).sum().item()
        
        # Rank for 90% energy
        cumsum = (S**2).cumsum(0)
        r90 = torch.where(cumsum / total_energy >= 0.90)[0][0].item() + 1
        
        # Smoothness (mean delta norm)
        smoothness = D_h.norm(dim=1).mean().item()
        
        # Entropy of singular values (spectral entropy)
        probs = (S**2) / total_energy
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        
        head_results.append({
            "head_idx": h,
            "rank_90": int(r90),
            "smoothness": round(smoothness, 6),
            "spectral_entropy": round(entropy, 6),
            "energy_top1": round((S[0]**2 / total_energy).item(), 4)
        })
        
        if h < 8: # Print first few
            print(f"  Head {h:>2}: r90={r90:<2} | entropy={entropy:.4f} | top1_energy={head_results[-1]['energy_top1']:.4f}")

        head_results.sort(key=lambda x: x["rank_90"])
        
    # --- New: Head-Selective Compression Test ---
    print(f"\n[Selective Head Compression Test]")
    # We'll compress only the top 50% most compressible heads (lowest r90)
    # and keep the other 50% as standard deltas (FP16).
    compressible_heads = [h["head_idx"] for h in head_results[:HEADS//2]]
    
    total_selective_sq_err = 0
    total_global_sq_err = 0
    total_elements = 0
    
    avg_rank = int(np.mean([h["rank_90"] for h in head_results]))
    
    for h in range(HEADS):
        head_deltas = []
        for i in range(0, SEQ_LEN, 64):
            anc = kv[i, :, h, :].float()
            for t in range(i + 1, min(i + 64, SEQ_LEN)):
                head_deltas.append((kv[t, :, h, :].float() - anc).flatten())
        
        if not head_deltas: continue
        D_h = torch.stack(head_deltas)
        total_elements += D_h.numel()
        
        # 1. Global Rank Baseline
        lr_global = compress_lowrank(D_h, avg_rank)
        recon_global = decompress_lowrank(lr_global, dtype=torch.float32)
        total_global_sq_err += torch.norm(D_h - recon_global)**2
        
        # 2. Selective
        if h in compressible_heads:
            lr_sel = compress_lowrank(D_h, 2) # aggressive
            recon_sel = decompress_lowrank(lr_sel, dtype=torch.float32)
        else:
            recon_sel = D_h # preserve
        total_selective_sq_err += torch.norm(D_h - recon_sel)**2
        
    rms_global = torch.sqrt(total_global_sq_err / (total_elements + 1e-9)).item()
    rms_selective = torch.sqrt(total_selective_sq_err / (total_elements + 1e-9)).item()
    
    print(f"  Global (avg_rank={avg_rank}) RMS Error: {rms_global:.6f}")
    print(f"  Selective Head RMS Error: {rms_selective:.6f}")
    
    # Update results
    final_output = {
        "head_stats": head_results,
        "selective_test": {
            "compressible_heads": compressible_heads,
            "selective_error": rms_selective,
            "global_error": rms_global,
            "avg_rank_baseline": avg_rank
        }
    }

    # Save
    with open(output_dir / "headwise_stats.json", "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_dir / 'headwise_stats.json'}")

if __name__ == "__main__":
    run_headwise_experiment()
