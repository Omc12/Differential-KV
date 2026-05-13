from typing import Dict, List
from empirical.runtime_truth_logger import RuntimeTruthLogger

class QueuePressureTelemetry:
    """Tracks queue depth, wait times, and starvation in multi-user serving."""
    def __init__(self, logger: RuntimeTruthLogger):
        self.logger = logger

    def track_queue(self, queue_depth: int, active_users: int, pending_requests: int):
        self.logger.log("queue_pressure", {
            "queue_depth": queue_depth,
            "active_users": active_users,
            "pending_requests": pending_requests,
            "saturation_ratio": queue_depth / active_users if active_users > 0 else 0
        })

class RetrievalInterferenceTrace:
    """Measures how concurrent users degrade individual retrieval stability."""
    def __init__(self, logger: RuntimeTruthLogger):
        self.logger = logger

    def log_interference(self, user_id: int, base_retrieval_latency: float, concurrent_latency: float):
        interference = (concurrent_latency - base_retrieval_latency) / base_retrieval_latency if base_retrieval_latency > 0 else 0
        self.logger.log("retrieval_interference", {
            "user_id": user_id,
            "base_latency_ms": base_retrieval_latency * 1000,
            "concurrent_latency_ms": concurrent_latency * 1000,
            "interference_penalty": float(interference)
        })

class MultiUserLatencyTracker:
    """Tracks per-user latency statistics under concurrent load."""
    def __init__(self, logger: RuntimeTruthLogger):
        self.logger = logger

    def log_user_latency(self, user_id: int, latency: float, batch_size: int):
        self.logger.log("user_latency", {
            "user_id": user_id,
            "latency_ms": latency * 1000,
            "batch_size": batch_size,
            "ms_per_token": (latency * 1000) / batch_size if batch_size > 0 else 0
        })
