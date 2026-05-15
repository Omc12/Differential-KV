
import torch
from typing import Dict, Any, List

class KVPressureProfiler:
    """
    PHASE 24.3: KV Pressure Profiler (LCS).
    Tracks KV growth and residency pressure under long-context scaling.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history = []
        
    def record_pressure(self, 
                        context_len: int, 
                        total_kv_bytes: int, 
                        sparse_kv_bytes: int):
        """
        Records KV residency metrics at a specific context length.
        """
        reduction_ratio = 1.0 - (sparse_kv_bytes / total_kv_bytes) if total_kv_bytes > 0 else 0.0
        
        entry = {
            "context_len": context_len,
            "total_kv_mb": total_kv_bytes / 1e6,
            "sparse_kv_mb": sparse_kv_bytes / 1e6,
            "reduction_ratio": reduction_ratio,
            "vram_pressure": torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0.0
        }
        self.history.append(entry)
        return entry

    def get_scaling_metrics(self) -> Dict[str, Any]:
        if not self.history:
            return {}
            
        avg_reduction = sum(e["reduction_ratio"] for e in self.history) / len(self.history)
        max_context = max(e["context_len"] for e in self.history)
        
        return {
            "avg_kv_reduction": avg_reduction,
            "max_context_benchmarked": max_context,
            "pressure_trend": "increasing" if self.history[-1]["vram_pressure"] > self.history[0]["vram_pressure"] else "stable"
        }
