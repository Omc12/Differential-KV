import torch

class RetrievalAnchorEstimator:
    """
    Identifies tokens that act as 'anchors' for future retrieval.
    Uses attention-path analysis (tokens that are frequently attended to).
    """
    def __init__(self, decay_rate: float = 0.95):
        self.anchor_scores = None
        self.decay_rate = decay_rate

    def update_anchors(self, attention_probs: torch.Tensor):
        """
        attention_probs: [batch, heads, q_len, k_len]
        """
        # Sum attention weights over queries to see which keys are most 'popular'
        popularity = attention_probs.sum(dim=-2) # [batch, heads, k_len]
        
        if self.anchor_scores is None:
            self.anchor_scores = popularity
        else:
            # Update cumulative popularity with decay
            # Need to handle expanding k_len
            current_k_len = popularity.size(-1)
            prev_k_len = self.anchor_scores.size(-1)
            
            if current_k_len > prev_k_len:
                # Pad previous scores
                padding = torch.zeros(*self.anchor_scores.shape[:-1], current_k_len - prev_k_len, device=self.anchor_scores.device)
                self.anchor_scores = torch.cat([self.anchor_scores, padding], dim=-1)
                
            self.anchor_scores = self.decay_rate * self.anchor_scores + (1.0 - self.decay_rate) * popularity

    def get_top_anchors(self, k: int = 16) -> torch.Tensor:
        """
        Returns indices of the most important anchor tokens.
        """
        if self.anchor_scores is None:
            return None
        
        # Aggregate over heads
        mean_scores = self.anchor_scores.mean(dim=1) # [batch, k_len]
        _, indices = torch.topk(mean_scores, k, dim=-1)
        return indices
