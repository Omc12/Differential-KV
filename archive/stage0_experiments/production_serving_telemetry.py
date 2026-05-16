import time
import numpy as np
from typing import List, Dict, Any

class ProductionServingTelemetry:
    """
    PSR System 5: Production Serving Telemetry.
    Tracks end-to-end serving metrics, including latency percentiles and overhead ratios.
    """
    def __init__(self):
        self.ttfts: List[float] = []
        self.itls: List[float] = []
        self.total_latencies: List[float] = []
        self.queue_delays: List[float] = []
        self.token_counts: List[int] = []
        self.start_time = time.time()
        
        self.sparse_runtime_ms = 0.0
        self.serving_overhead_ms = 0.0

    def record_request_metrics(self, ttft: float, total_time: float, tokens: int, queue_delay: float):
        self.ttfts.append(ttft)
        self.total_latencies.append(total_time)
        self.token_counts.append(tokens)
        self.queue_delays.append(queue_delay)
        
        # Calculate ITL if tokens > 1
        if tokens > 1:
            self.itls.append((total_time - ttft) / (tokens - 1))

    def record_overhead(self, runtime_ms: float, overhead_ms: float):
        self.sparse_runtime_ms += runtime_ms
        self.serving_overhead_ms += overhead_ms

    def get_full_report(self) -> Dict[str, Any]:
        duration = time.time() - self.start_time
        total_tokens = sum(self.token_counts)
        
        return {
            "p50_ttft_ms": np.percentile(self.ttfts, 50) * 1000 if self.ttfts else 0,
            "p95_ttft_ms": np.percentile(self.ttfts, 95) * 1000 if self.ttfts else 0,
            "p99_ttft_ms": np.percentile(self.ttfts, 99) * 1000 if self.ttfts else 0,
            "p50_itl_ms": np.percentile(self.itls, 50) * 1000 if self.itls else 0,
            "p99_itl_ms": np.percentile(self.itls, 99) * 1000 if self.itls else 0,
            "avg_queue_delay_ms": np.mean(self.queue_delays) * 1000 if self.queue_delays else 0,
            "system_tps": total_tokens / duration if duration > 0 else 0,
            "tokens_per_sec_per_user": np.mean([1.0/itl if itl > 0 else 0 for itl in self.itls]) if self.itls else 0,
            "sparse_runtime_ratio": self.sparse_runtime_ms / (self.sparse_runtime_ms + self.serving_overhead_ms) if (self.sparse_runtime_ms + self.serving_overhead_ms) > 0 else 0,
            "serving_overhead_ratio": self.serving_overhead_ms / (self.sparse_runtime_ms + self.serving_overhead_ms) if (self.sparse_runtime_ms + self.serving_overhead_ms) > 0 else 0,
            "total_requests": len(self.ttfts)
        }
