import torch

class AttentionPathStitcher:
    """
    PHASE 19.0D: Attention Path Stitcher.
    Stitches together disconnected sparse islands by preserving 
    critical traversal paths found in the attention matrix.
    """
    def __init__(self, top_k_paths: int = 4):
        self.top_k_paths = top_k_paths

    def stitch_paths(self, attention_probs: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        """
        Identifies high-probability attention paths from the current query 
        to previous tokens and ensures those paths are preserved.
        
        Args:
            attention_probs: [batch, num_heads, q_len, k_len]
            current_mask: [1, k_len] existing importance mask
        """
        # Average across heads and queries to find globally important keys for THIS step
        avg_attn = attention_probs.mean(dim=(0, 1, 2)) # [k_len]
        
        # Select top-K most attended tokens that aren't already in the mask
        values, indices = torch.topk(avg_attn, min(self.top_k_paths, avg_attn.numel()))
        
        new_mask = current_mask.clone()
        new_mask[0, indices] = True
        
        return new_mask
        
    def get_stitching_efficiency(self, attention_probs: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Measures what percentage of attention mass is captured by the mask.
        """
        avg_attn = attention_probs.mean(dim=(0, 1, 2))
        masked_attn = avg_attn[mask[0]]
        return masked_attn.sum().item()
