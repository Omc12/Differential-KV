"""
runtime/real_kv_profiler.py

True GPU residency profiler for Differential KV.
Uses torch.cuda memory stats and delta accounting.
"""

import torch
import gc
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ResidencySnapshot:
    active_mb: float
    reserved_mb: float
    fragmentation_mb: float
    peak_mb: float

class RealKVProfiler:
    """
    Measures real GPU residency by tracking allocator deltas.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.baseline_active = 0
        self.baseline_reserved = 0
        
    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            gc.collect()
            torch.cuda.empty_cache()

    def capture_baseline(self):
        """Capture memory state before KV cache allocation."""
        self.synchronize()
        self.baseline_active = torch.cuda.memory_allocated(self.device)
        self.baseline_reserved = torch.cuda.memory_reserved(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def get_residency(self) -> ResidencySnapshot:
        """Calculate residency relative to baseline."""
        self.synchronize()
        current_active = torch.cuda.memory_allocated(self.device)
        current_reserved = torch.cuda.memory_reserved(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        
        active_delta = current_active - self.baseline_active
        reserved_delta = current_reserved - self.baseline_reserved
        peak_delta = peak - self.baseline_active
        
        return ResidencySnapshot(
            active_mb=max(0.0, active_delta / 1024**2),
            reserved_mb=max(0.0, reserved_delta / 1024**2),
            fragmentation_mb=max(0.0, (reserved_delta - active_delta) / 1024**2),
            peak_mb=max(0.0, peak_delta / 1024**2)
        )

    @staticmethod
    def get_tensor_residency(manager) -> Dict[str, float]:
        """
        Calculates residency by summing up tensor bytes.
        This serves as a sanity check against allocator-level deltas.
        """
        stats = {
            "anchors": 0,
            "lowrank_u": 0,
            "lowrank_v": 0,
            "sparse": 0,
            "quantized": 0,
            "total": 0
        }
        
        for layer_idx, blocks in manager.cache.items():
            for block in blocks:
                # Anchors
                a = block.anchor_kv.element_size() * block.anchor_kv.nelement()
                stats["anchors"] += a
                stats["total"] += a
                
                # LowRank
                if block.U is not None:
                    u = block.U.element_size() * block.U.nelement()
                    v = block.V.element_size() * block.V.nelement()
                    stats["lowrank_u"] += u
                    stats["lowrank_v"] += v
                    stats["total"] += u + v
                
                # Sparse
                if block.sparse_values is not None:
                    s = (block.sparse_values.element_size() * block.sparse_values.nelement() + 
                         block.sparse_indices.element_size() * block.sparse_indices.nelement())
                    stats["sparse"] += s
                    stats["total"] += s
                    
                # Quantized or Raw FP16
                if block.q_deltas is not None:
                    if block.mode == "fp16":
                        q = block.q_deltas.element_size() * block.q_deltas.nelement()
                    else:
                        # Our QuantizedDelta has .data (tensor) and .scale (float)
                        q = block.q_deltas.data.element_size() * block.q_deltas.data.nelement()
                        q += 4 # float scale
                    stats["quantized"] += q
                    stats["total"] += q
                    
        # Convert to MB
        for k in stats:
            stats[k] /= (1024**2)
            
        return stats
