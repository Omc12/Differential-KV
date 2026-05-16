import torch
import numpy as np
from typing import Dict, Optional
from empirical.runtime_truth_logger import RuntimeTruthLogger

class LiveSparseTelemetry:
    """
    Tracks real sparse density and retrieval metrics during execution.
    """
    def __init__(self, logger: RuntimeTruthLogger):
        self.logger = logger
        self.step_count = 0

    def track_step(self, 
                   active_tokens: int, 
                   total_capacity: int, 
                   retrieval_success: bool,
                   retrieval_latency: float,
                   anchor_count: int):
        """Tracks a single inference step's sparse metrics."""
        density = active_tokens / total_capacity if total_capacity > 0 else 0
        
        metrics = {
            "step": self.step_count,
            "density": float(density),
            "active_tokens": active_tokens,
            "total_capacity": total_capacity,
            "retrieval_success": retrieval_success,
            "retrieval_latency_ms": retrieval_latency * 1000,
            "anchor_count": anchor_count
        }
        
        self.logger.log("sparse_telemetry", metrics)
        self.step_count += 1

    def log_density_drift(self, target_density: float, actual_density: float):
        """Logs drift between intended and actual sparsity."""
        self.logger.log("density_drift", {
            "target": target_density,
            "actual": actual_density,
            "drift": actual_density - target_density
        })

if __name__ == "__main__":
    from empirical.runtime_truth_logger import RuntimeTruthLogger
    logger = RuntimeTruthLogger("telemetry_test")
    telemetry = LiveSparseTelemetry(logger)
    telemetry.track_step(1024, 8192, True, 0.005, 128)
