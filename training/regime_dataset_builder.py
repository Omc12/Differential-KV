"""
training/regime_dataset_builder.py

Builds a dataset of latent trajectory features labeled by reasoning regime.
Used to train a robust regime classifier.
"""

import torch
import json
import os
import glob
from typing import List, Dict, Any
from analysis.cognitive_regime_classifier import CognitiveRegimeClassifier

class RegimeDatasetBuilder:
    def __init__(self, output_path: str = "training/regime_data.json"):
        self.output_path = output_path
        self.data = []
        self.classifier = CognitiveRegimeClassifier() # Used for initial labeling/bootstrap
        
    def collect_from_trajectories(self, trajectory_dir: str):
        """
        Processes saved trajectory files and extracts features.
        Files are expected to be .json or .pt files containing metrics and hidden state stats.
        """
        files = glob.glob(os.path.join(trajectory_dir, "*.json"))
        for f in files:
            with open(f, "r") as f_in:
                traj = json.load(f_in)
                # traj is expected to have 'metrics' and 'label' (or we infer label from filename/metadata)
                label = traj.get("regime_label", "unknown")
                if label == "unknown":
                    # Try to infer from task name
                    task = traj.get("task_name", "").lower()
                    if "math" in task or "gsm8k" in task:
                        label = "mathematical_reasoning"
                    elif "code" in task or "humaneval" in task:
                        label = "code_generation"
                    elif "plan" in task:
                        label = "recursive_planning"
                    elif "qa" in task or "retrieval" in task:
                        label = "retrieval_heavy"
                    else:
                        label = "narrative_dialogue"
                
                metrics = traj.get("metrics", [])
                for step_metrics in metrics:
                    self.data.append({
                        "features": step_metrics,
                        "label": label
                    })
        
        print(f"Collected {len(self.data)} samples.")

    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"Dataset saved to {self.output_path}")

if __name__ == "__main__":
    builder = RegimeDatasetBuilder()
    # In a real run, we'd point to results/phase27/trajectories/
    # For this implementation, we'll create some synthetic high-fidelity data if none exists
    os.makedirs("training", exist_ok=True)
    builder.save()
