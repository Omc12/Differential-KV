import torch
import numpy as np
from typing import List, Dict

class LocalTokenCoherenceTracker:
    """
    SPS Phase 20.6: Local Token Coherence Tracker.
    Tracks symbolic sequence consistency and detects early drift.
    """
    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.history = []
        self.match_history = []
        
    def record(self, generated_id: int, expected_id: int):
        self.history.append((generated_id, expected_id))
        self.match_history.append(1.0 if generated_id == expected_id else 0.0)
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
            self.match_history.pop(0)
            
    def get_coherence_score(self) -> float:
        """Returns 1.0 for perfect local matching, 0.0 for total drift."""
        if not self.match_history:
            return 1.0
        return sum(self.match_history) / len(self.match_history)

class SymbolicDriftPredictor:
    """
    SPS Phase 20.6: Symbolic Drift Predictor.
    Predicts symbolic corruption BEFORE it happens based on logit distribution.
    """
    def predict_risk(self, logits: torch.Tensor, expected_id: int) -> float:
        """
        Calculates a risk score [0, 1] that the next token will drift.
        High risk if:
        - Expected token is not top-1
        - Expected token prob is low
        - Competitive tokens are in different symbolic categories
        """
        probs = torch.softmax(logits.float().squeeze(0), dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5)
        
        expected_prob = probs[expected_id].item()
        top_id = top_ids[0].item()
        
        risk = 0.0
        
        # Risk 1: Probability Gap
        if expected_prob < 0.5:
            risk += 0.4
        if expected_prob < 0.1:
            risk += 0.4
            
        # Risk 2: Misalignment
        if top_id != expected_id:
            risk += 0.2
            
        return min(1.0, risk)

class PrecisionEntropyAuditor:
    """
    SPS Phase 20.6: Precision Entropy Auditor.
    Ensures symbolic stabilization does not collapse decoder freedom.
    """
    def __init__(self):
        self.entropy_history = []
        
    def audit(self, logits: torch.Tensor) -> Dict[str, float]:
        probs = torch.softmax(logits.float().squeeze(0), dim=-1)
        log_probs = torch.log(probs + 1e-12)
        entropy = -(probs * log_probs).sum().item()
        
        self.entropy_history.append(entropy)
        
        # Detect collapse
        is_collapsed = entropy < 0.05  # threshold for deterministic forcing
        
        return {
            "entropy_nats": entropy,
            "is_collapsed": is_collapsed,
            "mean_entropy": np.mean(self.entropy_history[-20:]) if self.entropy_history else 0.0
        }
