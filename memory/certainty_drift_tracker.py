import torch

class CertaintyDriftTracker:
    def __init__(self):
        self.drift_score = 0.0
    def track_drift(self, importance: torch.Tensor):
        self.drift_score = (importance < 1000.0).float().mean().item()
