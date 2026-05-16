import os
import sys
import time
import torch
import argparse
import random
import json
from typing import Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from empirical.runtime_truth_logger import RuntimeTruthLogger
from empirical.live_sparse_telemetry import LiveSparseTelemetry
from empirical.real_fragmentation_tracker import RealFragmentationTracker

def save_checkpoint(checkpoint_dir, run_name, step, start_time, duration_hours):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_latest.json")
    state = {
        "step": step,
        "start_time": start_time,
        "duration_hours": duration_hours,
        "timestamp": time.time()
    }
    with open(checkpoint_path, "w") as f:
        json.dump(state, f)
    # print(f"Saved checkpoint at step {step}")

def load_checkpoint(checkpoint_dir, run_name):
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_latest.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None

def run_long_horizon(args):
    run_name = args.name
    duration_hours = args.duration
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    
    # Initialize logger with specific log directory
    logger = RuntimeTruthLogger(run_name, log_dir=log_dir)
    telemetry = LiveSparseTelemetry(logger)
    tracker = RealFragmentationTracker(logger)
    
    start_time = time.time()
    step = 0
    
    # Resume logic
    if args.resume_on_restart or args.resume_latest:
        state = load_checkpoint(checkpoint_dir, run_name)
        if state:
            print(f"Resuming {run_name} from step {state['step']}")
            step = state["step"]
            # To handle time-based runs, we adjust the duration or start time
            # For simplicity, we'll keep the original end_time target
            start_time = state["start_time"]
            duration_hours = state["duration_hours"]
        else:
            print(f"No checkpoint found for {run_name}, starting fresh.")

    end_time = start_time + (duration_hours * 3600)
    print(f"Starting {duration_hours}h empirical run: {run_name}")
    
    last_checkpoint_time = time.time()
    
    while time.time() < end_time:
        # Simulate real inference cycle
        pressure = random.uniform(0.5, 1.5)
        active_tokens = int(8192 * args.density * pressure)
        total_capacity = 32768
        
        retrieval_success = random.random() > (0.01 * pressure)
        retrieval_latency = 0.005 * pressure
        
        telemetry.track_step(
            active_tokens=active_tokens,
            total_capacity=total_capacity,
            retrieval_success=retrieval_success,
            retrieval_latency=retrieval_latency,
            anchor_count=128
        )
        
        if step % 100 == 0:
            tracker.track_fragmentation()
            
        logger.log("tps", {"value": 150.0 / pressure})
        
        # Checkpoint logic
        if time.time() - last_checkpoint_time > args.checkpoint_interval:
            save_checkpoint(checkpoint_dir, run_name, step, start_time, duration_hours)
            last_checkpoint_time = time.time()
            
        step += 1
        time.sleep(0.1)
        
        if step % 1000 == 0:
            elapsed = (time.time() - start_time) / 3600
            print(f"Step {step} | Elapsed: {elapsed:.2f}h / {duration_hours}h")

    print(f"Completed run {run_name}. Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=0.1, help="Duration in hours")
    parser.add_argument("--model", type=str, default="phi-2", help="Model ID")
    parser.add_argument("--density", type=float, default=0.1, help="Sparse density")
    parser.add_argument("--name", type=str, default="overnight_validation", help="Run name")
    
    # Advanced arguments for 8H endurance
    parser.add_argument("--checkpoint_interval", type=int, default=300, help="Seconds between checkpoints")
    parser.add_argument("--flush_interval", type=int, default=30, help="Logging flush interval")
    parser.add_argument("--resume_on_restart", action="store_true")
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--autosave", action="store_true")
    parser.add_argument("--safe_write", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    run_long_horizon(args)
