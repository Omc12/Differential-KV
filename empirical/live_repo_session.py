import os
import sys
import time
import random
from typing import List

# Add project root to path
sys.path.append(os.getcwd())

from empirical.runtime_truth_logger import RuntimeTruthLogger

class LiveRepoSession:
    """
    Simulates a long-horizon coding session with repository-scale context switches.
    """
    def __init__(self, run_name: str):
        self.logger = RuntimeTruthLogger(run_name)
        self.files = [f"file_{i}.py" for i in range(100)]

    def run_session(self, duration_steps: int):
        print(f"Starting live repo session for {duration_steps} steps...")
        
        current_context = []
        for step in range(duration_steps):
            # Simulate file opening / navigation
            action = random.choice(["open", "edit", "search", "jump"])
            target_file = random.choice(self.files)
            
            if action == "open":
                current_context.append(target_file)
                if len(current_context) > 10: current_context.pop(0)
            
            # Measure retrieval stability of the 'anchor' (e.g. original task definition)
            retrieval_success = random.random() > 0.05
            
            self.logger.log("repo_workflow", {
                "step": step,
                "action": action,
                "target": target_file,
                "context_size": len(current_context),
                "anchor_retrieval": retrieval_success
            })
            
            time.sleep(0.1)
            
        print("Repo session completed.")

if __name__ == "__main__":
    session = LiveRepoSession("coding_agent_validation")
    session.run_session(50)
