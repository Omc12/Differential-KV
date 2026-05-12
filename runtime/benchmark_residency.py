"""
runtime/benchmark_residency.py

High-fidelity residency benchmarking for Differential KV.
Uses RealKVProfiler to capture allocator-level deltas.
"""

import os
import torch
import json
import time
from typing import Dict, List, Any
from tqdm import tqdm

from runtime.hf_diffkv_wrapper import DiffKVHFWrapper
from runtime.real_kv_profiler import RealKVProfiler

def run_residency_benchmark(
    model_id: str,
    context_lengths: List[int],
    modes: List[Dict[str, Any]],
    device: str = "cuda"
):
    results = {}
    profiler = RealKVProfiler(device=device)
    
    for mode_cfg in modes:
        mode_name = mode_cfg["name"]
        print(f"\n>>> Benchmarking Residency: {mode_name}")
        results[mode_name] = []
        
        # Load model wrapper (baseline)
        try:
            # We load the model once per mode to ensure a clean state
            wrapper = DiffKVHFWrapper(model_id, mode_cfg, device=device)
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
            
        for ctx_len in context_lengths:
            print(f"  Context Length: {ctx_len}")
            
            # 1. Start profiling (capture baseline with model loaded)
            profiler.capture_baseline()
            
            # 2. Prefill to fill KV cache
            prompt = "The " * (ctx_len - 1)
            inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(device)
            
            try:
                with torch.no_grad():
                    # We must use a method that updates the manager
                    # For prefill, we can do it in one shot then push to manager
                    outputs = wrapper.model(inputs.input_ids, use_cache=True)
                    wrapper._update_manager(outputs.past_key_values)
                    
                    # CRITICAL: Delete standard HF cache to measure ONLY our manager's residency
                    del outputs
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"    Failed at {ctx_len}: {e}")
                # Don't break the whole benchmark, just skip this context
                break
                
            # 3. Capture residency after prefill (Compressed state)
            stats = profiler.get_residency()
            
            # 4. Sanity check: Tensor-level summing
            tensor_stats = RealKVProfiler.get_tensor_residency(wrapper.manager)
            
            # 5. Measure Peak during reconstruction (Decode 1 step)
            with torch.no_grad():
                _ = wrapper.generate("Once", max_new_tokens=1)
            peak_stats = profiler.get_residency()
            
            results[mode_name].append({
                "context_length": ctx_len,
                "active_mb": stats.active_mb,
                "reserved_mb": stats.reserved_mb,
                "fragmentation_mb": stats.fragmentation_mb,
                "peak_recon_mb": peak_stats.peak_mb,
                "tensor_sum_mb": tensor_stats["total"],
                "breakdown": tensor_stats
            })
            
            # Clear manager for next context
            wrapper.manager.clear()
            torch.cuda.empty_cache()
            
        # Cleanup
        del wrapper
        torch.cuda.empty_cache()
        
    return results

if __name__ == "__main__":
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    CONTEXTS = [1024, 2048, 4096, 8192]
    
    MODES = [
        {"name": "fp16", "mode": "fp16", "block_size": 64},
        {"name": "int8", "mode": "int8", "block_size": 64},
        {"name": "lowrank_r16", "mode": "lowrank", "rank": 16, "block_size": 64},
        {"name": "lowrank_r8", "mode": "lowrank", "rank": 8, "block_size": 64},
        {"name": "lowrank_sparse", "mode": "lowrank_sparse", "rank": 16, "sparse_ratio": 0.01, "block_size": 64},
    ]
    
    results = run_residency_benchmark(MODEL, CONTEXTS, MODES)
    
    os.makedirs("results/phase5c", exist_ok=True)
    with open("results/phase5c/real_kv_residency.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResidency results saved to results/phase5c/real_kv_residency.json")
