import time
import torch
from typing import Dict

class CapsuleOverheadMapper:
    """
    PHASE 18.7E: Capsule Overhead Mapper.
    Measures the compute and memory cost of maintaining hierarchical capsules.
    """
    def __init__(self):
        self.traces = []

    def measure_lookup_latency(self, registry, num_lookups: int = 1000):
        """Measures the overhead of capsule-aware KV routing."""
        start = time.perf_counter()
        for i in range(num_lookups):
            _ = registry.get_capsule_for_token(i % 16384)
        end = time.perf_counter()
        return (end - start) / num_lookups

    def estimate_vram_overhead(self, registry) -> Dict[str, float]:
        """Calculates VRAM consumed by capsule metadata and pinned KV states."""
        metadata_bytes = 0
        kv_bytes = 0
        
        for capsule in registry.capsules.values():
            # Estimate metadata size (very rough)
            metadata_bytes += 512 # approx size of MemoryCapsule object
            
            if capsule.kv_states:
                for layer_idx, tensor in capsule.kv_states.items():
                    kv_bytes += tensor.element_size() * tensor.nelement()
                    
        return {
            "metadata_mb": metadata_bytes / (1024 * 1024),
            "kv_pinned_mb": kv_bytes / (1024 * 1024),
            "total_mb": (metadata_bytes + kv_bytes) / (1024 * 1024)
        }

    def log_trace(self, tps: float, vram_usage: float, fidelity_gain: float):
        self.traces.append({
            "timestamp": time.time(),
            "tps": tps,
            "vram_gb": vram_usage,
            "fidelity_gain": fidelity_gain
        })
