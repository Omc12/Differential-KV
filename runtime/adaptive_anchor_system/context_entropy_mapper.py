import torch
import torch.nn.functional as F

class ContextEntropyMapper:
    """
    Calculates token-level and region-level entropy to guide anchor placement.
    High entropy regions (high information) require denser anchors.
    """
    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing

    def calculate_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculates entropy from attention weights.
        attn_weights: [H, Q, K]
        Returns: [K] entropy per key token
        """
        # Average across heads and queries to get general importance/uncertainty of keys
        avg_attn = attn_weights.mean(dim=(0, 1)) # [K]
        
        # Calculate Shannon entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        p = avg_attn + 1e-9
        entropy = -(p * torch.log(p)).sum() / torch.log(torch.tensor(p.size(0), dtype=torch.float))
        
        return entropy

    def get_token_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        More granular entropy calculation for each token.
        """
        # [H, Q, K] -> [Q, K] (mean over heads)
        mean_attn = attn_weights.mean(dim=0)
        
        # Entropy per query: how 'spread out' is the attention for each query
        # But we want to know which keys are 'hard' to retrieve or 'dense' in info.
        # Actually, let's look at the entropy of the attention distribution *received* by keys.
        
        # Re-normalize across keys for each query if not already
        p = F.softmax(mean_attn, dim=-1)
        
        token_entropy = -(p * torch.log(p + 1e-9)).sum(dim=0) # [K]
        return token_entropy

    def map_sequence_entropy(self, attn_weights: torch.Tensor, bucket_size: int = 128) -> torch.Tensor:
        """
        Maps entropy into buckets to match anchor modes.
        """
        token_entropy = self.get_token_entropy(attn_weights)
        
        # Pool entropy into buckets
        num_buckets = (token_entropy.size(0) + bucket_size - 1) // bucket_size
        buckets = []
        for i in range(num_buckets):
            start = i * bucket_size
            end = min((i + 1) * bucket_size, token_entropy.size(0))
            buckets.append(token_entropy[start:end].mean())
            
        return torch.stack(buckets)
