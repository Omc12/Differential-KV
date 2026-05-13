import os
import sys
import time
import threading
import random
import argparse
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

    def run(self):
        start_time = time.time()
        while time.time() - start_time < self.duration and not self.stop_event.is_set():
            # Simulate an inference request
            request_start = time.time()
            batch_size = random.randint(1, 32)
            
            # Simulate work
            time.sleep(random.uniform(0.05, 0.2))
            
            latency = time.time() - request_start
            self.latency_tracker.log_user_latency(self.user_id, latency, batch_size)
            
            # Simulate retrieval interference measurement
            # In real system, this would measure actual kernel contention
            base_lat = 0.005
            self.interference_trace.log_interference(self.user_id, base_lat, base_lat * random.uniform(1.0, 2.0))
            
            # Think time
            time.sleep(random.uniform(0.1, 0.5))

def run_concurrency_test(num_users: int, duration: float, name: str):
    print(f"Starting concurrency test: {num_users} users, {duration}s")
    logger = RuntimeTruthLogger(name)
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
    while time.time() - start_time < duration:
        active = sum(1 for u in users if u.is_alive())
        pending = random.randint(0, active * 2) # Simulated
        queue_telemetry.track_queue(active + pending, active, pending)
        time.sleep(1)
        
    for user in users:
        user.stop_event.set()
        user.join()
        
    print(f"Concurrency test {name} finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--name", type=str, default="concurrency_validation")
    args = parser.parse_args()
    
    run_concurrency_test(args.users, args.duration, args.name)
