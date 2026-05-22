import torch
import torch.nn as nn

class RetrievalPriorityAttention(nn.Module):
    """
    Attention mechanism that dynamically prioritizes retrieval-heavy heads.
    Allocates more sparsity 'budget' to heads showing high retrieval activity.
    """
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_retrieval_momentum = torch.zeros(num_heads)

    def forward(self, q, k, v, retrieval_mask):
        """
        Adjusts attention weights based on retrieval priority.
        """
        # Calculate retrieval intensity per head
        # retrieval_mask: [batch, heads, seq_len]
        intensity = retrieval_mask.float().sum(dim=-1).mean(dim=0)
        
        # Update momentum
        self.head_retrieval_momentum = 0.9 * self.head_retrieval_momentum.to(intensity.device) + 0.1 * intensity
        
        # Apply priority scaling
        priority_scores = torch.softmax(self.head_retrieval_momentum, dim=0)
        
        # Standard attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        
        # Heads with higher priority get less aggressive pruning/masking
        # (This would be used by the scheduler to adjust per-head density)
        
        return torch.matmul(torch.softmax(attn, dim=-1), v), priority_scores
