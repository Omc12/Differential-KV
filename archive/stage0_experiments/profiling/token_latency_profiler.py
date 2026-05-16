"""
profiling/token_latency_profiler.py

Measures token-to-token latency, throughput (tok/sec), and routing overhead.
"""

import time
import torch
from typing import List, Dict, Any

class TokenLatencyProfiler:
    def __init__(self):
        self.latencies = []
        
    def start_token(self):
        self.start_time = time.time()
        
    def end_token(self):
        latency = (time.time() - self.start_time) * 1000 # ms
        self.latencies.append(latency)
        
    def get_statistics(self) -> Dict[str, float]:
        if not self.latencies: return {}
        
        avg_latency = sum(self.latencies) / len(self.latencies)
        throughput = 1000.0 / avg_latency
        
        return {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": sorted(self.latencies)[int(len(self.latencies) * 0.95)],
            "throughput_tok_sec": throughput,
            "total_tokens": len(self.latencies)
        }
