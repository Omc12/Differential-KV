import torch
import collections

class DynamicCacheEvictor:
    """
    Policy-based cache eviction (LRU or Importance-based).
    """
    def __init__(self, max_cache_size=1024, policy="lru"):
        self.max_cache_size = max_cache_size
        self.policy = policy
        self.access_history = collections.defaultdict(list) # Track access for LRU

    def evict(self, key_states, value_states, step_idx):
        """
        Args:
            key_states: [batch, heads, seq_len, head_dim]
            value_states: [batch, heads, seq_len, head_dim]
            step_idx: Current generation step
        """
        seq_len = key_states.size(2)
        if seq_len <= self.max_cache_size:
            return key_states, value_states
        
        if self.policy == "lru":
            # In a real model, we'd need to track which tokens were attended to.
            # Here we simulate LRU by keeping the last N tokens.
            return key_states[:, :, -self.max_cache_size:], value_states[:, :, -self.max_cache_size:]
        
        elif self.policy == "importance":
            # Simple importance: L2 norm (magnitude)
            importance = torch.norm(key_states, dim=-1) # [batch, heads, seq_len]
            _, top_indices = torch.topk(importance, self.max_cache_size, dim=-1)
            top_indices = top_indices.sort(dim=-1).values
            
            # Gather (simplified)
            batch_size, num_heads, _, head_dim = key_states.shape
            idx = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            # This gather is slightly different from the pruner one due to shape
            pruned_k = torch.gather(key_states, 2, idx)
            pruned_v = torch.gather(value_states, 2, idx)
            
            return pruned_k, pruned_v

        return key_states, value_states
