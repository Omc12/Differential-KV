"""
profiling/sparse_scheduler_overhead.py

Quantifies sparse execution scheduling costs in Differential KV.
Focus: sparse scheduling, memory migration latency.
"""

import torch
import time
import numpy as np
from typing import Dict, Any

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_sparse_scheduler_overhead(config: Dict[str, Any], num_steps: int = 100):
    print("--- STARTING SPARSE SCHEDULER OVERHEAD PROFILING ---")
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    scheduling_latencies = []
    
    for step in range(num_steps):
        # Prepare mock input for scheduler
        health_info = {"cognitive_health_score": 0.8, "collapse_probability": 0.1}
        
        start = time.perf_counter()
        # In actual system, scheduler.decide_rank() or similar
        # Here we use the DynamicRankScheduler from anchor_logic
        rank_params = runtime.scheduler.schedule_step(step, health_info)
        scheduling_latencies.append(time.perf_counter() - start)
        
    avg_latency = np.mean(scheduling_latencies) * 1000
    p95_latency = np.percentile(scheduling_latencies, 95) * 1000
    
    print(f"\nAverage Scheduling Latency: {avg_latency:.4f} ms")
    print(f"P95 Scheduling Latency:     {p95_latency:.4f} ms")
    print(f"Total steps profiled:      {num_steps}")

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1,
        "repair_threshold": 0.3
    }
    run_sparse_scheduler_overhead(config)
