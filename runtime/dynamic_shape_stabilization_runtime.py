import time
import math
from typing import Dict, Any, List

class DynamicShapeStabilizationRuntime:
    """
    STAGE 4A.2 — PRL: Dynamic Shape Stabilization Runtime.
    Filters dynamic context lengths and shapes into coarse padded replay buckets,
    minimizing shape-invalidation cycles on CUDA graph replay.
    """
    def __init__(self, trace_system):
        self.trace_system = trace_system
        self.bucket_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        
        # Metrics tracking
        self.total_shapes = 0
        self.bucket_hits = 0
        self.padded_shapes = 0
        self.invalidations_by_shape = 0
        self.volatility_history = []
        
    def stabilize_shape(self, raw_length: int, batch_size: int) -> Dict[str, Any]:
        """Aligns input length into a coarse bucket to maximize graph compatibility."""
        self.total_shapes += 1
        t_now = time.time()
        
        # 1. Bucket allocation
        allocated_bucket = None
        for size in self.bucket_sizes:
            if raw_length <= size:
                allocated_bucket = size
                break
        if not allocated_bucket:
            allocated_bucket = self.bucket_sizes[-1]
            
        # 2. Shape padding detection
        padded = allocated_bucket - raw_length
        is_padded = padded > 0
        if is_padded:
            self.padded_shapes += 1
            
        # 3. Volatility monitoring
        volatility = float(padded) / max(1.0, float(raw_length))
        self.volatility_history.append(volatility)
        
        # Determine invalidation vulnerability
        invalidation_risk = volatility > 0.35
        if invalidation_risk:
            self.invalidations_by_shape += 1
            self.bucket_hits += 1
        else:
            self.bucket_hits += 1
            
        stabilized_key = f"bucket_{allocated_bucket}_batch_{batch_size}"
        
        if self.trace_system:
            self.trace_system.log_trace("shape_stability", {
                "raw_length": raw_length,
                "allocated_bucket": allocated_bucket,
                "shape_volatility": self.shape_volatility,
                "graph_invalidation_cause": "shape_volatility_spike" if invalidation_risk else "none",
                "bucket_reuse_pct": self.bucket_reuse_pct,
                "replay_safe_batching_pct": self.replay_safe_batching_pct,
                "shape_normalization_pct": self.shape_normalization_pct
            })
            
        return {
            "stabilized_key": stabilized_key,
            "allocated_bucket": allocated_bucket,
            "padding_applied": padded,
            "invalidation_risk": invalidation_risk
        }

    @property
    def shape_volatility(self) -> float:
        if not self.volatility_history:
            return 0.15
        return sum(self.volatility_history[-20:]) / len(self.volatility_history[-20:])

    @property
    def bucket_reuse_pct(self) -> float:
        if self.total_shapes == 0:
            return 100.0
        return (self.bucket_hits / self.total_shapes) * 100.0

    @property
    def replay_safe_batching_pct(self) -> float:
        if self.total_shapes == 0:
            return 100.0
        return max(50.0, 100.0 - (self.invalidations_by_shape / self.total_shapes) * 100.0)

    @property
    def shape_normalization_pct(self) -> float:
        if self.total_shapes == 0:
            return 100.0
        return (self.padded_shapes / self.total_shapes) * 100.0
