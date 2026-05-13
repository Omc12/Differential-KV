import torch
import torch.nn.functional as F

class SparseAttentionRouter:
    """
    Minimal sparse attention router that selects a subset of KV heads/tokens 
    to attend to, reducing FLOPs.
    """
    def __init__(self, sparsity_factor=0.5):
        self.sparsity_factor = sparsity_factor

    def route(self, query_states, key_states, value_states):
        """
        Args:
            query_states: [batch, heads, 1, head_dim] (decoding step)
            key_states: [batch, heads, seq_len, head_dim]
            value_states: [batch, heads, seq_len, head_dim]
        """
        # Calculate attention scores
        # scores: [batch, heads, 1, seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / (key_states.size(-1) ** 0.5)
        
        # Softmax to get importance
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # Select top-k tokens per head
        seq_len = key_states.size(2)
        k = max(1, int(seq_len * (1 - self.sparsity_factor)))
        
        _, top_indices = torch.topk(attn_probs, k, dim=-1)
        
        # Mask out non-selected tokens (for validation of throughput vs accuracy)
        # In a real sparse kernel, we wouldn't even load these.
        mask = torch.zeros_like(attn_probs).scatter_(-1, top_indices, 1.0)
        
        # For actual FLOP reduction in simulation, we can return the indices
        # or a masked attention result.
        return mask, top_indices

    def apply_mask(self, attn_weights, mask):
        return attn_weights + (1.0 - mask) * -10000.0
