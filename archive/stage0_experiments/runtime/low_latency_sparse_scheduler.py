"""
runtime/low_latency_sparse_scheduler.py

High-efficiency scheduler for sparse execution in Differential KV.
Focuses on minimizing scheduling overhead and maximizing GPU occupancy.
"""

import time
import torch
from typing import List, Dict, Any

class LowLatencySparseScheduler:
    def __init__(self, target_tps: int = 400, max_concurrency: int = 8):
        self.target_tps = target_tps
        self.max_concurrency = max_concurrency
        self.execution_queue = []
        self.active_jobs = 0
        self.latency_log = []

    def schedule_sparse_batch(self, batch_id: str, priority: float = 1.0):
        """
        Schedules a sparse computation batch with latency awareness.
        """
        start_time = time.perf_counter()
        
        # Simple priority-based scheduling
        job = {
            "id": batch_id,
            "priority": priority,
            "timestamp": start_time,
            "status": "pending"
        }
        
        self.execution_queue.append(job)
        self.execution_queue.sort(key=lambda x: x["priority"], reverse=True)
        
        # Wait if too many active jobs (simulated low-latency throttling)
        while self.active_jobs >= self.max_concurrency:
            time.sleep(0.0001) # 100 microseconds
            
        self.active_jobs += 1
        job["status"] = "executing"
        
        return job

    def complete_job(self, batch_id: str):
        """Marks a job as complete and updates stats."""
        for job in self.execution_queue:
            if job["id"] == batch_id:
                latency = (time.perf_counter() - job["timestamp"]) * 1000
                self.latency_log.append(latency)
                self.execution_queue.remove(job)
                self.active_jobs -= 1
                return latency
        return None

    def get_performance_metrics(self):
        """Returns P95 latency and average scheduling overhead."""
        if not self.latency_log:
            return {"p95_ms": 0, "avg_ms": 0}
            
        sorted_latencies = sorted(self.latency_log)
        p95_idx = int(len(sorted_latencies) * 0.95)
        
        return {
            "p95_ms": sorted_latencies[p95_idx],
            "avg_ms": sum(self.latency_log) / len(self.latency_log),
            "throughput_tps": len(self.latency_log) / (time.perf_counter() - self.latency_log[0] / 1000 if self.latency_log else 1)
        }
