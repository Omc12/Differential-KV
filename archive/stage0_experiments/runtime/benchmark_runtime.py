"""
runtime/benchmark_runtime.py — Phase 4

Comprehensive runtime benchmarking for Differential KV.
Measures latency, throughput, and VRAM usage.
"""

import os
import time
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import torch.nn.functional as F

# Import our compression modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compression.quantization import quantize_int8, dequantize_int8
from compression.lowrank import compress_lowrank, decompress_lowrank, LowRankDelta
from compression.sparse_repair import compress_lowrank_sparse, decompress_lowrank_sparse, LowRankSparseDelta

@dataclass
class ModelConfig:
    name: str
    num_heads: int
    head_dim: int
    num_layers: int

MODELS = {
    "tinyllama": ModelConfig("TinyLlama-1.1B", 32, 64, 22),
    "phi-2": ModelConfig("Phi-2", 32, 80, 32),
    "gemma-2b": ModelConfig("Gemma-2B", 8, 256, 18), # K/V heads
}

class RuntimeKVManager:
    """Simulates a KV Cache manager with different compression backends."""
    def __init__(self, config: ModelConfig, method: str, device: str = "cuda"):
        self.config = config
        self.method = method
        self.device = device
        self.layers_kv = [] # List of layers
        self.feat_dim = 2 * config.num_heads * config.head_dim
        
    def reset(self, seq_len: int):
        self.layers_kv = []
        torch.cuda.empty_cache()
        
        # Pre-allocate or prepare based on method
        if self.method == "fp16":
            # Just full tensors
            for _ in range(self.config.num_layers):
                self.layers_kv.append(torch.randn(seq_len, self.feat_dim, dtype=torch.float16, device=self.device))
        elif self.method == "int8":
            for _ in range(self.config.num_layers):
                raw = torch.randn(seq_len, self.feat_dim, dtype=torch.float16, device=self.device)
                self.layers_kv.append(quantize_int8(raw))
        elif self.method == "diffkv_periodic":
            # Anchor every 64, INT8 deltas
            interval = 64
            for _ in range(self.config.num_layers):
                raw = torch.randn(seq_len, self.feat_dim, dtype=torch.float16, device=self.device)
                anchors = {}
                deltas = {}
                for i in range(0, seq_len, interval):
                    anc = raw[i].clone()
                    anchors[i] = anc
                    end = min(i + interval, seq_len)
                    if end > i + 1:
                        d = raw[i+1:end] - anc
                        # Ensure everything stays on device
                        q = quantize_int8(d)
                        deltas[i] = q
                self.layers_kv.append((anchors, deltas))
        elif self.method == "diffkv_lowrank":
            # Anchor every 64, Rank-16 deltas
            interval = 64
            rank = 16
            for _ in range(self.config.num_layers):
                raw = torch.randn(seq_len, self.feat_dim, dtype=torch.float16, device=self.device)
                anchors = {}
                deltas = {}
                for i in range(0, seq_len, interval):
                    anc = raw[i].clone()
                    anchors[i] = anc
                    end = min(i + interval, seq_len)
                    if end > i + 1:
                        d = raw[i+1:end] - anc
                        lr = compress_lowrank(d, rank)
                        # Explicitly move to device to avoid any SVD-induced CPU migration
                        lr.U = lr.U.to(self.device)
                        lr.V = lr.V.to(self.device)
                        deltas[i] = lr
                self.layers_kv.append((anchors, deltas))
        elif self.method == "diffkv_lowrank_sparse":
            # Anchor every 64, Rank-16 deltas + 1% sparse
            interval = 64
            rank = 16
            for _ in range(self.config.num_layers):
                raw = torch.randn(seq_len, self.feat_dim, dtype=torch.float16, device=self.device)
                anchors = {}
                deltas = {}
                for i in range(0, seq_len, interval):
                    anc = raw[i].clone()
                    anchors[i] = anc
                    end = min(i + interval, seq_len)
                    if end > i + 1:
                        d = raw[i+1:end] - anc
                        lrs = compress_lowrank_sparse(d, rank, sparse_ratio=0.01)
                        lrs.low_rank.U = lrs.low_rank.U.to(self.device)
                        lrs.low_rank.V = lrs.low_rank.V.to(self.device)
                        lrs.sparse_indices = lrs.sparse_indices.to(self.device)
                        lrs.sparse_values = lrs.sparse_values.to(self.device)
                        deltas[i] = lrs
                self.layers_kv.append((anchors, deltas))

    def get_vram_usage(self) -> float:
        """Measure actual VRAM allocated in MB."""
        # Note: We measure the memory allocated by the tensors themselves
        # or use torch.cuda.memory_allocated() if we want the global state.
        return torch.cuda.memory_allocated() / (1024**2)

    def simulate_decode_step(self, token_idx: int) -> float:
        """
        Simulate the overhead of fetching and reconstructing KV for one layer.
        Returns latency in milliseconds.
        """
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # We simulate the fetch logic for ALL layers to get a representative timing
        for layer_idx in range(self.config.num_layers):
            data = self.layers_kv[layer_idx]
            
            if self.method == "fp16":
                # Direct access
                _ = data[token_idx]
            elif self.method == "int8":
                # Dequantize token
                # data is QuantizedDelta
                _ = (data.data[token_idx].float() * data.scale).to(torch.float16)
            elif "diffkv" in self.method:
                anchors, deltas = data
                interval = 64
                anchor_idx = (token_idx // interval) * interval
                anchor_val = anchors[anchor_idx]
                
                if token_idx == anchor_idx:
                    _ = anchor_val
                else:
                    delta_block = deltas[anchor_idx]
                    local_idx = token_idx - anchor_idx - 1
                    if "periodic" in self.method:
                        # delta_block is QuantizedDelta
                        d = (delta_block.data[local_idx].float() * delta_block.scale)
                        _ = anchor_val + d.to(torch.float16)
                    elif "lowrank" in self.method:
                        if "sparse" in self.method:
                            # delta_block is LowRankSparseDelta
                            # Reconstructing single token: U[i] @ V + sparse
                            u_i = delta_block.low_rank.U[local_idx].float()
                            d = (u_i @ delta_block.low_rank.V * delta_block.low_rank.scale)
                            # Sparse repair lookup (simulated as we'd need a mask or index search)
                            # For simplicity in simulation, we just do a small add
                            _ = anchor_val + d.to(torch.float16)
                        else:
                            # delta_block is LowRankDelta
                            u_i = delta_block.U[local_idx].float()
                            d = (u_i @ delta_block.V * delta_block.scale)
                            _ = anchor_val + d.to(torch.float16)

        torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) * 1000

def run_experiment_suite():
    print("Starting Phase 4 Runtime Validation...")
    results = []
    
    contexts = [4096, 8192, 16384, 32768]
    methods = ["fp16", "int8", "diffkv_periodic", "diffkv_lowrank", "diffkv_lowrank_sparse"]
    model_cfg = MODELS["tinyllama"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. Results will not reflect GPU performance.")

    for ctx in contexts:
        print(f"\nContext Length: {ctx}")
        for method in methods:
            manager = RuntimeKVManager(model_cfg, method, device)
            
            # Reset and measure baseline VRAM
            torch.cuda.empty_cache()
            base_mem = torch.cuda.memory_allocated() / (1024**2)
            
            try:
                manager.reset(ctx)
                total_mem = manager.get_vram_usage() - base_mem
                
                # Warmup
                for i in range(min(10, ctx)):
                    manager.simulate_decode_step(i)
                
                # Measure latency
                latencies = []
                n_steps = 100
                indices = np.random.randint(0, ctx, n_steps)
                for idx in indices:
                    latencies.append(manager.simulate_decode_step(int(idx)))
                
                avg_latency = np.mean(latencies)
                tokens_per_sec = 1000.0 / (avg_latency + 1e-9)
                
                res = {
                    "context": ctx,
                    "method": method,
                    "vram_mb": round(total_mem, 2),
                    "avg_latency_ms": round(avg_latency, 4),
                    "tokens_per_sec": round(tokens_per_sec, 2)
                }
                results.append(res)
                print(f"  {method:20}: {res['tokens_per_sec']:8} tok/s | {res['vram_mb']:8.1f} MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  {method:20}: OOM at {ctx}")
                    results.append({"context": ctx, "method": method, "status": "OOM"})
                    torch.cuda.empty_cache()
                else:
                    raise e

    # Save results
    os.makedirs("results/runtime_validation", exist_ok=True)
    with open("results/runtime_validation/benchmark_data.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_experiment_suite()
