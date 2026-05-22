import torch

class AnchorSelectionEngine:
    """
    PHASE 18.2A: Selects critical KV anchors from real hidden states.
    Uses a combination of Recent Tokens and Heavy Hitters (Importance).
    """
    def __init__(self, anchor_budget: int = 1024, recent_window: int = 512):
        self.anchor_budget = anchor_budget
        self.recent_window = recent_window

    @torch.no_grad()
    def select_anchors(self, attention_scores: torch.Tensor):
        """
        Input: attention_scores [batch, num_heads, seq_len]
        Output: indices of selected tokens [batch, budget]
        """
        # Average across heads
        avg_scores = attention_scores.mean(dim=1) # [batch, seq_len]
        batch_size, seq_len = avg_scores.shape
        
        if seq_len <= self.anchor_budget + self.recent_window:
            return torch.arange(seq_len, device=avg_scores.device).unsqueeze(0).expand(batch_size, -1)

        # 1. Protect Recent Window
        recent_indices = torch.arange(seq_len - self.recent_window, seq_len, device=avg_scores.device)
        
        # 2. Select Heavy Hitters from the past
        past_scores = avg_scores[:, :seq_len - self.recent_window]
        heavy_hitter_budget = self.anchor_budget - self.recent_window
        
        _, top_indices = torch.topk(past_scores, k=heavy_hitter_budget, dim=-1)
        
        # Combine
        combined_indices = torch.cat([top_indices, recent_indices.expand(batch_size, -1)], dim=-1)
        return torch.sort(combined_indices, dim=-1)[0]
