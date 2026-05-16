import torch
import time
from typing import Dict, Any

class PersistentTritonDispatcher:
    """
    Forces Triton kernels into dominant execution position via persistent dispatch.
    """
    def __init__(self):
        self.stats = {
            "triton_launch_count": 0,
            "dense_fallback_count": 0,
            "total_triton_runtime": 0.0,
            "sustained_kernel_uptime": 0.0
        }
        self.start_time = time.perf_counter()

    def dispatch_kernel(self, kernel_fn, *args, **kwargs):
        """
        Dispatches a Triton kernel and tracks occupancy.
        """
        t0 = time.perf_counter()
        
        # In SHM, we suppress dense fallbacks
        try:
            result = kernel_fn(*args, **kwargs)
            self.stats["triton_launch_count"] += 1
        except Exception as e:
            # Fallback is only allowed if absolutely necessary, but counted as failure
            self.stats["dense_fallback_count"] += 1
            print(f"[SHM] WARNING: Persistent dispatch failed, falling back: {e}")
            raise e
            
        duration = time.perf_counter() - t0
        self.stats["total_triton_runtime"] += duration
        return result

    def get_telemetry(self) -> Dict[str, Any]:
        total_uptime = time.perf_counter() - self.start_time
        runtime_percent = (self.stats["total_triton_runtime"] / total_uptime) * 100 if total_uptime > 0 else 0
        
        return {
            "triton_kernel_runtime_percent": runtime_percent,
            "triton_launch_count": self.stats["triton_launch_count"],
            "dense_fallback_count": self.stats["dense_fallback_count"],
            "sustained_kernel_uptime": total_uptime
        }

# Global singleton
dispatcher = PersistentTritonDispatcher()
