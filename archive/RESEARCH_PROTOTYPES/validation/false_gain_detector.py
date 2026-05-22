import os
import torch
import numpy as np

class FalseGainDetector:
    """
    Detects benchmark leakage, hidden state carryover, and benchmark inflation.
    """

    def __init__(self):
        self.baseline_scores = {}
        self.observed_scores = {}

    def check_for_hidden_carryover(self):
        """
        Scans for files or environment variables that might persist state
        between independent evaluation runs.
        """
        suspect_vars = ["DIFFERENTIAL_KV_STATE", "COGNITIVE_CACHE", "RESONANCE_DATA"]
        found_leakage = False
        
        for var in suspect_vars:
            if var in os.environ:
                print(f"[CRITICAL] Leakage Detected: Environment variable {var} found.")
                found_leakage = True
        
        # Check for unexpected checkpoint files
        if os.path.exists("session_checkpoints") and len(os.listdir("session_checkpoints")) > 0:
            print("[CRITICAL] Leakage Detected: session_checkpoints is not empty.")
            found_leakage = True
            
        return found_leakage

    def validate_gain_reproducibility(self, runs_data):
        """
        Ensures gains are not just outliers or seed-dependent artifacts.
        Expects a list of scores from multiple randomized runs.
        """
        if len(runs_data) < 3:
            return False, "Insufficient data for reproducibility check."
        
        std_dev = np.std(runs_data)
        mean_score = np.mean(runs_data)
        
        # If standard deviation is too high, the gain is likely unstable/random
        if std_dev > (0.5 * mean_score):
            return False, f"High variance detected (std={std_dev:.4f}). Gain is likely an artifact."
        
        return True, "Gain appears stable across runs."

    def detect_prompt_overlap(self, prompt, dataset_prompts):
        """
        Checks if the current prompt has significant overlap with training 
        or benchmark data (contamination detector).
        """
        # Simple overlap check for demonstration
        for ds_prompt in dataset_prompts:
            if prompt in ds_prompt or ds_prompt in prompt:
                return True
        return False

if __name__ == "__main__":
    detector = FalseGainDetector()
    if not detector.check_for_hidden_carryover():
        print("No obvious hidden carryover detected.")
    else:
        print("Leakage detected. Environment must be purged.")
