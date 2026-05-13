import torch
import torch.nn as nn

class EntropyGuidedEviction(nn.Module):
    """
    Uses attention entropy to guide KV cache eviction.
    High entropy attention distributions suggest many tokens are relevant.
    Low entropy suggests only a few 'heavy hitters' are needed.
    """
    def __init__(self, entropy_threshold: float = 0.8):
        super().__init__()
        self.entropy_threshold = entropy_threshold

    def compute_entropy(self, attention_probs: torch.Tensor) -> torch.Tensor:
        """
        attention_probs: [batch, heads, q_len, k_len]
        """
        # Entropy = -sum(p * log(p))
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-9), dim=-1)
        return entropy.mean(dim=-2) # Average over queries

    def determine_eviction_count(self, entropy: torch.Tensor, current_cache_size: int) -> int:
        """
        Determines how many tokens to evict based on entropy.
        Higher entropy = less eviction (keep more context).
        """
        # Normalize entropy to [0, 1] range (roughly)
        # For a sequence of N, max entropy is log(N)
        max_entropy = torch.log(torch.tensor(float(current_cache_size)))
        normalized_entropy = (entropy / max_entropy).clamp(0, 1)
        
        # If entropy is low, we can evict more.
        eviction_ratio = (1.0 - normalized_entropy).mean().item()
        
        # Guard: don't evict everything
        eviction_ratio = min(eviction_ratio, 0.8)
        
        return int(current_cache_size * eviction_ratio)
