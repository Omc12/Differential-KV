"""
experiments/exp_lowrank_static.py — Phase 3 Stage A

Research prototype for static low-rank delta reconstruction.
Compares ΔKV ≈ U @ V.T across multiple ranks vs benchmarks.
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
from compression.lowrank import compress_kv_sequence_lowrank, decompress_kv_sequence_lowrank, estimate_memory
from compression.quantization import quantize_int8, dequantize_int8
from evaluation.generation_drift import GenerationDriftEvaluator

# Config
RANKS = [1, 2, 4, 8, 16, 32]
MODES = ["mixed", "smooth", "real_approx"]
SEQ_LEN = 2048
HEADS = 32
DIM = 128
INTERVAL = 64

def run_static_recon_experiment():
    output_dir = Path("results/lowrank_static")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gen = KVGenerator(num_heads=HEADS, head_dim=DIM, seed=42)
    results = {}

    print(f"\n{'='*80}")
    print(f"  STAGE A: STATIC LOW-RANK DELTA RECONSTRUCTION")
    print(f"  seq_len={SEQ_LEN} | interval={INTERVAL} | ranks={RANKS}")
    print(f"{'='*80}\n")

    for mode in MODES:
        print(f"--- Mode: {mode} ---")
        kv = gen.generate(SEQ_LEN, mode=mode)
        anchor_positions = list(range(0, SEQ_LEN, INTERVAL))
        
        mode_results = {"ranks": {}, "benchmarks": {}}
        
        # 1. Test Ranks
        for r in RANKS:
            # Low-Rank
            blocks, anchors = compress_kv_sequence_lowrank(kv, anchor_positions, r)
            recon_kv = decompress_kv_sequence_lowrank(blocks, anchors, kv.shape)
            
            error = torch.norm(kv.float() - recon_kv.float()) / torch.norm(kv.float())
            mem = estimate_memory(SEQ_LEN, HEADS, DIM, r, INTERVAL)
            
            mode_results["ranks"][r] = {
                "error": round(error.item(), 6),
                "ratio_fp16": mem["ratio_fp16"],
                "ratio_int8": mem["ratio_int8"],
                "bytes": mem["lowrank_bytes"]
            }
            print(f"  Rank {r:>2}: err={error.item():.6f} | ratio={mem['ratio_fp16']:.2f}x")

        # 2. Benchmarks
        # INT8 Delta (simplified comparison)
        # We simulate INT8 by quantizing the deltas from the same anchors
        feat_dim = 2 * HEADS * DIM
        total_int8_err = 0
        n_deltas = 0
        for i in range(len(anchor_positions)):
            start = anchor_positions[i]
            end = anchor_positions[i+1] if i+1 < len(anchor_positions) else SEQ_LEN
            anchor_kv = kv[start].float()
            for t in range(start + 1, end):
                delta = (kv[t].float() - anchor_kv).reshape(-1)
                q = quantize_int8(delta)
                dq = dequantize_int8(q, target_dtype=torch.float32)
                total_int8_err += torch.norm(delta - dq)**2
                n_deltas += 1
        
        avg_int8_err = torch.sqrt(total_int8_err / (n_deltas * feat_dim + 1e-9)).item()
        
        mode_results["benchmarks"]["int8"] = {
            "error": round(avg_int8_err, 6),
            "ratio_fp16": 2.0 # approx for symmetric int8 vs fp16
        }
        print(f"  INT8   : err={avg_int8_err:.6f} | ratio=~2.0x")
        
        results[mode] = mode_results

    # Save Results
    with open(output_dir / "static_recon_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_dir / 'static_recon_stats.json'}")

def run_behavioral_validation():
    print(f"\n{'='*80}")
    print(f"  BEHAVIORAL VALIDATION (Token Agreement / Drift)")
    print(f"{'='*80}\n")
    
    evaluator = GenerationDriftEvaluator(model_name="gpt2", device="cpu")
    evaluator.load_model()
    
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Differential KV cache compression is a technique to",
        "import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):",
        "The capital of France is Paris. The capital of Germany is"
    ]
    
    strategies = ["periodic_64"] + [f"lowrank_{r}" for r in [1, 4, 16]]
    
    results = evaluator.evaluate(prompts, strategies=strategies)
    
    output_dir = Path("results/lowrank_static")
    with open(output_dir / "behavioral_validation.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"\n[OK] Behavioral results saved.")

if __name__ == "__main__":
    run_static_recon_experiment()
    # Note: run_behavioral_validation() is slower and requires model weights.
    # We'll keep it in the script but user might want to run it selectively.
    if "--behavior" in sys.argv:
        run_behavioral_validation()
