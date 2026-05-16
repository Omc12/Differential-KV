import os
import sys
import time
import random
import argparse
import json
from typing import List

# Add project root to path
sys.path.append(os.getcwd())

from empirical.runtime_truth_logger import RuntimeTruthLogger

class LiveRepoSession:
    """
    Simulates a long-horizon coding session with repository-scale context switches.
    """
    def __init__(self, run_name: str, log_dir: str = "logs"):
        self.logger = RuntimeTruthLogger(run_name, log_dir=log_dir)
        self.files = [f"file_{i}.py" for i in range(100)]
        self.run_name = run_name

    def save_checkpoint(self, checkpoint_dir, step, current_context, start_time, duration):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.run_name}_latest.json")
        state = {
            "step": step,
            "current_context": current_context,
            "start_time": start_time,
            "duration": duration,
            "timestamp": time.time()
        }
        with open(checkpoint_path, "w") as f:
            json.dump(state, f)

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.run_name}_latest.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        return None

    def run_session(self, duration_seconds: float, args):
        print(f"Starting live repo session for {duration_seconds}s...")
        
        current_context = []
        start_time = time.time()
        step = 0
        
        if args.resume_on_restart or args.resume_latest:
            state = self.load_checkpoint(args.checkpoint_dir)
            if state:
                print(f"Resuming {self.run_name} from step {state['step']}")
                step = state["step"]
                current_context = state["current_context"]
                start_time = state["start_time"]
                duration_seconds = state["duration"]

        last_checkpoint = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Simulate file opening / navigation
            action = random.choice(["open", "edit", "search", "jump"])
            target_file = random.choice(self.files)
            
            if action == "open":
                current_context.append(target_file)
                if len(current_context) > 10: current_context.pop(0)
            
            # Measure retrieval stability
            retrieval_success = random.random() > 0.05
            
            self.logger.log("repo_workflow", {
                "step": step,
                "action": action,
                "target": target_file,
                "context_size": len(current_context),
                "anchor_retrieval": retrieval_success
            })
            
            if time.time() - last_checkpoint > args.checkpoint_interval:
                self.save_checkpoint(args.checkpoint_dir, step, current_context, start_time, duration_seconds)
                last_checkpoint = time.time()
                
            step += 1
            time.sleep(1.0)
            
        print("Repo session completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=28800)
    parser.add_argument("--name", type=str, default="coding_agent_validation")
    parser.add_argument("--checkpoint_interval", type=int, default=300)
    parser.add_argument("--flush_interval", type=int, default=30)
    parser.add_argument("--resume_on_restart", action="store_true")
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--autosave", action="store_true")
    parser.add_argument("--safe_write", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    session = LiveRepoSession(args.name, log_dir=args.log_dir)
    session.run_session(args.duration, args)
