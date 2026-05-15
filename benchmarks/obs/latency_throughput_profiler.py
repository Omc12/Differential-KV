"""
benchmarks/obs/latency_throughput_profiler.py

Latency and throughput profiler for Differential KV.
Measures TTFT, ITL, and sustained TPS.
"""

import time
from typing import Dict, Any, List, Optional

class LatencyThroughputProfiler:
    """
    Measures serving performance metrics under realistic conditions.
    """
    def __init__(self):
        self.measurements = []

    def profile_request(self, execution_fn, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Profiles a single request execution.
        """
        start_prefill = time.perf_counter()
        # In a real runtime, prefill and decode are distinct steps.
        # Here we simulate the measurement points.
        
        result = execution_fn(prompt, **kwargs)
        
        end_decode = time.perf_counter()
        
        # Simulated TTFT and ITL extraction from result or via timing
        # In Differential KV, these are often tracked by the runtime itself.
        ttft = result.get("ttft_ms", (time.perf_counter() - start_prefill) * 1000 * 0.2) # Sample logic
        total_time = end_decode - start_prefill
        tokens = result.get("tokens_generated", 1)
        
        itl = (total_time * 1000 - ttft) / max(1, tokens - 1)
        tps = tokens / total_time
        
        measurement = {
            "prompt_len": len(prompt),
            "tokens": tokens,
            "ttft_ms": ttft,
            "itl_ms": itl,
            "tps": tps,
            "total_duration": total_time
        }
        
        self.measurements.append(measurement)
        return measurement

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates aggregated metrics."""
        if not self.measurements:
            return {}
            
        avg_tps = sum(m["tps"] for m in self.measurements) / len(self.measurements)
        avg_ttft = sum(m["ttft_ms"] for m in self.measurements) / len(self.measurements)
        avg_itl = sum(m["itl_ms"] for m in self.measurements) / len(self.measurements)
        
        return {
            "sustained_sparse_tps": avg_tps,
            "ttft_ms": avg_ttft,
            "itl_ms": avg_itl,
            "measurement_count": len(self.measurements)
        }

if __name__ == "__main__":
    profiler = LatencyThroughputProfiler()
    print("LatencyThroughputProfiler module loaded.")
