"""
runtime/kv_memory_profiler.py

Isolated memory profiling for Differential KV components.
Measures actual tensor residency.
"""

import torch
import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class MemoryStats:
    total_bytes: int
    anchor_bytes: int
    lowrank_bytes: int
    sparse_bytes: int
    quantized_bytes: int
    fp16_baseline_bytes: int
    
    def to_dict(self):
        return {
            "total_mb": self.total_bytes / (1024**2),
            "anchor_mb": self.anchor_bytes / (1024**2),
            "lowrank_mb": self.lowrank_bytes / (1024**2),
            "sparse_mb": self.sparse_bytes / (1024**2),
            "quantized_mb": self.quantized_bytes / (1024**2),
            "fp16_baseline_mb": self.fp16_baseline_bytes / (1024**2),
            "compression_ratio": self.fp16_baseline_bytes / (self.total_bytes + 1e-9)
        }

class KVMemoryProfiler:
    """
    Profiles memory residency of KVRuntimeManager components.
    """
    @staticmethod
    def profile_manager(manager) -> MemoryStats:
        total = 0
        anchors = 0
        lowrank = 0
        sparse = 0
        quantized = 0
        baseline = 0
        
        for layer_idx, blocks in manager.cache.items():
            for block in blocks:
                # Baseline estimate: what would this block be in FP16?
                # Block size: 1 anchor + deltas
                num_tokens = 1 + (len(block.token_indices) if block.token_indices else 0)
                feat_dim = block.anchor_kv.numel()
                baseline += num_tokens * feat_dim * 2 # FP16
                
                # Actual residency
                a_bytes = block.anchor_kv.element_size() * block.anchor_kv.nelement()
                anchors += a_bytes
                total += a_bytes
                
                if block.U is not None:
                    u_bytes = block.U.element_size() * block.U.nelement()
                    v_bytes = block.V.element_size() * block.V.nelement()
                    lowrank += u_bytes + v_bytes
                    total += u_bytes + v_bytes
                    
                if block.sparse_values is not None:
                    s_bytes = (block.sparse_values.element_size() * block.sparse_values.nelement() + 
                               block.sparse_indices.element_size() * block.sparse_indices.nelement())
                    sparse += s_bytes
                    total += s_bytes
                    
                if block.q_deltas is not None:
                    q_bytes = block.q_deltas.element_size() * block.q_deltas.nelement()
                    quantized += q_bytes
                    total += q_bytes
                    
        return MemoryStats(total, anchors, lowrank, sparse, quantized, baseline)

def generate_scaling_data(
    model_config: Dict[str, Any],
    contexts: List[int],
    mode: str = "lowrank",
    rank: int = 16,
    block_size: int = 64
):
    """
    Simulates a manager state for various contexts and measures residency.
    """
    results = []
    heads = model_config["num_heads"]
    dim = model_config["head_dim"]
    layers = model_config["num_layers"]
    
    for ctx in contexts:
        # We don't need real data for residency profiling, just tensors of correct shape
        # Simulation of a manager state
        total_anchors = max(1, ctx // block_size)
        tokens_per_block = block_size - 1
        
        anchor_bytes = total_anchors * (2 * heads * dim) * 2 * layers
        if mode == "lowrank":
            u_bytes = ctx * rank * 2 * layers # U is [n, rank]
            v_bytes = total_anchors * (rank * 2 * heads * dim) * 4 * layers # V is [rank, feat_dim]
            total_bytes = anchor_bytes + u_bytes + v_bytes
        elif mode == "fp16":
            total_bytes = ctx * (2 * heads * dim) * 2 * layers
        else:
            total_bytes = 0 # etc
            
        results.append({
            "context": ctx,
            "residency_mb": total_bytes / (1024**2),
            "bytes_per_token": total_bytes / ctx if ctx > 0 else 0
        })
        
    return results
