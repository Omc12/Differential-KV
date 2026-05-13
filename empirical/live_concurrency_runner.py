import os
import sys
import time
import threading
import random
import argparse
import json
from typing import List

# Add project root to path
sys.path.append(os.getcwd())

from empirical.runtime_truth_logger import RuntimeTruthLogger
from empirical.concurrency_telemetry import QueuePressureTelemetry, RetrievalInterferenceTrace, MultiUserLatencyTracker

class UserSimulator(threading.Thread):
    def __init__(self, user_id: int, duration: float, latency_tracker: MultiUserLatencyTracker, interference_trace: RetrievalInterferenceTrace):
        super().__init__()
        self.user_id = user_id
        self.duration = duration
        self.latency_tracker = latency_tracker
        self.interference_trace = interference_trace
        self.stop_event = threading.Event()
        self.elapsed = 0

    def run(self):
        start_time = time.time()
        while (time.time() - start_time) < self.duration and not self.stop_event.is_set():
            # Simulate an inference request
            request_start = time.time()
            batch_size = random.randint(1, 32)
            
            # Simulate work
            time.sleep(random.uniform(0.05, 0.2))
            
            latency = time.time() - request_start
            self.latency_tracker.log_user_latency(self.user_id, latency, batch_size)
            
            # Simulate retrieval interference measurement
            base_lat = 0.005
            self.interference_trace.log_interference(self.user_id, base_lat, base_lat * random.uniform(1.0, 2.0))
            
            # Think time
            time.sleep(random.uniform(0.1, 0.5))
            self.elapsed = time.time() - start_time

def save_checkpoint(checkpoint_dir, run_name, remaining_duration):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_latest.json")
    state = {
        "remaining_duration": remaining_duration,
        "timestamp": time.time()
    }
    with open(checkpoint_path, "w") as f:
        json.dump(state, f)

def load_checkpoint(checkpoint_dir, run_name):
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_latest.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None

def run_concurrency_test(args):
    num_users = args.users
    duration = args.duration
    name = args.name
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir

    if args.resume_on_restart or args.resume_latest:
        state = load_checkpoint(checkpoint_dir, name)
        if state:
            duration = state["remaining_duration"]
            print(f"Resuming {name} with {duration:.1f}s remaining")

    print(f"Starting concurrency test: {num_users} users, {duration}s")
    logger = RuntimeTruthLogger(name, log_dir=log_dir)
    queue_telemetry = QueuePressureTelemetry(logger)
    latency_tracker = MultiUserLatencyTracker(logger)
    interference_trace = RetrievalInterferenceTrace(logger)
    
    users = []
    for i in range(num_users):
        user = UserSimulator(i, duration, latency_tracker, interference_trace)
        users.append(user)
        user.start()
        
    # Monitor queue pressure
    start_time = time.time()
    last_checkpoint = time.time()
    
    try:
        while time.time() - start_time < duration:
            active = sum(1 for u in users if u.is_alive())
            pending = random.randint(0, active * 2) 
            queue_telemetry.track_queue(active + pending, active, pending)
            
            # Periodic checkpointing
            if time.time() - last_checkpoint > args.checkpoint_interval:
                rem = max(0, duration - (time.time() - start_time))
                save_checkpoint(checkpoint_dir, name, rem)
                last_checkpoint = time.time()
                
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping users...")
        
    for user in users:
        user.stop_event.set()
        user.join()
        
    print(f"Concurrency test {name} finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--name", type=str, default="concurrency_validation")
    
    # Advanced arguments for 8H endurance
    parser.add_argument("--checkpoint_interval", type=int, default=300)
    parser.add_argument("--flush_interval", type=int, default=30)
    parser.add_argument("--resume_on_restart", action="store_true")
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--autosave", action="store_true")
    parser.add_argument("--safe_write", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    run_concurrency_test(args)
