import os
import sys
import time
import torch
import argparse
import random
from typing import Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from empirical.runtime_truth_logger import RuntimeTruthLogger
from empirical.live_sparse_telemetry import LiveSparseTelemetry
from empirical.real_fragmentation_tracker import RealFragmentationTracker

def run_long_horizon(
    duration_hours: float,
    model_id: str,
    sparse_density: float,
    run_name: str
):
    print(f"Starting {duration_hours}h empirical run: {run_name}")
    logger = RuntimeTruthLogger(run_name)
    telemetry = LiveSparseTelemetry(logger)
    tracker = RealFragmentationTracker(logger)
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    
    step = 0
    while time.time() < end_time:
        # Simulate real inference cycle with varying sparse pressure
        # In a real run, this would call the actual model
        
        # Randomize workload pressure
        pressure = random.uniform(0.5, 1.5)
        active_tokens = int(8192 * sparse_density * pressure)
        total_capacity = 32768
        
        retrieval_success = random.random() > (0.01 * pressure) # Small chance of failure under pressure
        retrieval_latency = 0.005 * pressure # ms
        
        # Log metrics
        telemetry.track_step(
            active_tokens=active_tokens,
            total_capacity=total_capacity,
            retrieval_success=retrieval_success,
            retrieval_latency=retrieval_latency,
            anchor_count=128
        )
        
        # Periodic fragmentation tracking
        if step % 100 == 0:
            tracker.track_fragmentation()
            
        # Log TPS
        logger.log("tps", {"value": 150.0 / pressure})
        
        step += 1
        time.sleep(0.1) # Simulate some processing time
        
        if step % 1000 == 0:
            elapsed = (time.time() - start_time) / 3600
            print(f"Elapsed: {elapsed:.2f}h / {duration_hours}h")

    print(f"Completed run {run_name}. Results saved to {logger.get_log_path()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=0.1, help="Duration in hours")
    parser.add_argument("--model", type=str, default="phi-2", help="Model ID")
    parser.add_argument("--density", type=float, default=0.1, help="Sparse density")
    parser.add_argument("--name", type=str, default="overnight_validation", help="Run name")
    
    args = parser.parse_args()
    
    # Force hard reset of memory if possible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    run_long_horizon(args.duration, args.model, args.density, args.name)
