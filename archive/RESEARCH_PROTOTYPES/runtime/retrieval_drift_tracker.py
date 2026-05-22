import json
import os
from typing import Dict, Any, List

class RetrievalDriftTracker:
    """
    Logs and analyzes retrieval quality degradation over time.
    """
    def __init__(self, log_dir: str = "results/reconstruction_5b"):
        self.log_dir = log_dir
        self.scores = []
        os.makedirs(log_dir, exist_ok=True)

    def log_score(self, step: int, score: float):
        self.scores.append({"step": step, "score": score})
        
    def calculate_drift(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        initial = self.scores[0]["score"]
        current = self.scores[-1]["score"]
        return (current - initial) / initial

    def save_logs(self):
        path = os.path.join(self.log_dir, "retrieval_drift.json")
        with open(path, "w") as f:
            json.dump({
                "drift": self.calculate_drift(),
                "history": self.scores
            }, f, indent=2)
