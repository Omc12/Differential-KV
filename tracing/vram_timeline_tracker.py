"""
tracing/vram_timeline_tracker.py

Tracks VRAM allocation and fragmentation timelines during Differential KV execution.
Generates data for memory efficiency visualization.
"""

import torch
import time
import json
import os
import threading
from typing import List, Dict

class VRAMTimelineTracker:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.timeline = []
        self.running = False
        self.thread = None

    def _track(self):
        while self.running:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3) # GB
                reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                self.timeline.append({
                    "timestamp": time.time(),
                    "allocated_gb": allocated,
                    "reserved_gb": reserved
                })
            time.sleep(self.interval)

    def start(self):
        print("Starting VRAM Timeline Tracking...")
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print("VRAM Tracking stopped.")

    def save_timeline(self, output_path: str = "results/phase38/vram_timeline.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.timeline, f, indent=4)
        print(f"Timeline saved to {output_path}")

if __name__ == "__main__":
    tracker = VRAMTimelineTracker(interval=0.05)
    tracker.start()
    
    # Simulate workload
    if torch.cuda.is_available():
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(1024, 1024, 10, device="cuda"))
            time.sleep(0.1)
        del tensors
        torch.cuda.empty_cache()
        time.sleep(0.2)
        
    tracker.stop()
    tracker.save_timeline()
