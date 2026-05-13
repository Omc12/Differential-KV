"""
validation/real_memory_latency_meter.py

Phase 12.5B: Real Memory Latency Meter
Enforces wall-clock timing for all memory operations to prevent synthetic 
timing bypasses and simulated microsecond latencies.
"""

import time
from typing import Dict, Any, Callable

class RealMemoryLatencyMeter:
    """
    Measures the true latency of memory operations (VRAM, RAM, Disk).
    """
    def __init__(self):
        self.measurements = {}

    def measure(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Executes a function and records its exact wall-clock execution time.
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
        self.measurements[operation_name].append(latency_ms)
        
        return result, latency_ms

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Returns min, max, avg latencies for each operation."""
        summary = {}
        for op, times in self.measurements.items():
            if not times: continue
            summary[op] = {
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "samples": len(times)
            }
        return summary
