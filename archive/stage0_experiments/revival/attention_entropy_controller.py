import torch

class AttentionEntropyController:
    """
    Measures the entropy of attention distributions to signal pruning safety.
    """

    def __init__(self):
        pass

    def measure_entropy(self, attention_weights):
        """
        attention_weights: [batch, heads, query_len, key_len] (must be normalized)
        Returns a value between 0 and 1 representing normalized entropy.
        """
        # Calculate Shannon entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
        
        # Max entropy for a distribution of size N is log(N)
        max_entropy = torch.log(torch.tensor(attention_weights.shape[-1], dtype=torch.float32))
        
        normalized_entropy = entropy / (max_entropy + 1e-9)
        
        # Average across heads and batch
        return normalized_entropy.mean().item()

if __name__ == "__main__":
    controller = AttentionEntropyController()
    # Sharp attention (Low entropy)
    sharp = torch.zeros(1, 1, 1, 100)
    sharp[..., 0] = 1.0
    # Flat attention (High entropy)
    flat = torch.ones(1, 1, 1, 100) / 100.0
    
    print(f"Sharp Entropy: {controller.measure_entropy(sharp):.4f}")
    print(f"Flat Entropy: {controller.measure_entropy(flat):.4f}")
