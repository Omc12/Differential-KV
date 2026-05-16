import torch
from typing import Dict, List

class RetrievalPathTracker:
    """
    Monitors retrieval health across long contexts.
    Detects 'retrieval collapse' (when the model stops attending to past context).
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.retrieval_scores: List[float] = []

    def track_step(self, attn_weights: torch.Tensor):
        """
        Calculates a 'retrieval score' for the current step.
        Score = (Sum of attention to past context) / (Total attention)
        """
        # attn_weights: [H, Q, K]
        # In decoding, Q is usually 1. 
        # We want to see how much attention goes to the 'historical' part of K.
        
        k_len = attn_weights.size(-1)
        if k_len < 10:
            return

        # Ratio of attention to the first 90% of the context
        historical_split = int(k_len * 0.9)
        historical_attn = attn_weights[..., :historical_split].sum().item()
        total_attn = attn_weights.sum().item()
        
        score = historical_attn / (total_attn + 1e-9)
        self.retrieval_scores.append(score)
        
        if len(self.retrieval_scores) > self.window_size:
            self.retrieval_scores.pop(0)

    def is_collapsing(self, threshold: float = 0.05) -> bool:
        """
        Returns True if historical retrieval drops below threshold.
        """
        if len(self.retrieval_scores) < self.window_size // 2:
            return False
            
        recent_avg = sum(self.retrieval_scores[-10:]) / 10
        return recent_avg < threshold
