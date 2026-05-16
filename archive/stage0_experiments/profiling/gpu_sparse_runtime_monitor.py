import time
from typing import Dict, Any
from .cuda_sparse_timer import CUDASparseTimer
from .real_gpu_memory_map import RealGPUMemoryMap

class GPUSparseRuntimeMonitor:
    """
    PHASE 7.5B: GPU Sparse Runtime Monitor
    A unified monitoring system that aggregates hardware-level 
    telemetry for sparse operations in real-time.
    """
    def __init__(self):
        self.timer = CUDASparseTimer()
        self.memory_map = RealGPUMemoryMap()
        self.metrics_history = []

    def capture_runtime_snapshot(self) -> Dict[str, Any]:
        """
        Captures a comprehensive snapshot of current GPU state.
        """
        mem_stats = self.memory_map.get_memory_layout()
        timing_stats = self.timer.get_all_metrics()
        
        snapshot = {
            "timestamp": time.time(),
            "memory": mem_stats,
            "timing": timing_stats,
            "status": "HEALTHY" if mem_stats["fragmentation_mb"] < 500 else "FRAGMENTED"
        }
        
        self.metrics_history.append(snapshot)
        return snapshot

    def get_aggregate_stats(self) -> Dict[str, float]:
        """Computes average metrics over recent history."""
        if not self.metrics_history:
            return {}
            
        avg_vram = sum(s["memory"]["total_allocated_mb"] for s in self.metrics_history) / len(self.metrics_history)
        return {
            "avg_vram_mb": avg_vram,
            "snapshot_count": len(self.metrics_history)
        }
