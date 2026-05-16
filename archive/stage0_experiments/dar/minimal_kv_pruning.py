import torch

def prune_kv_cache(key_states, value_states, pruning_ratio=0.5, attention_sink_size=4):
    """
    Prune KV cache based on a simple importance metric (e.g., L2 norm of keys or attention scores).
    This is a minimal implementation for validation.
    
    Args:
        key_states: [batch, heads, seq_len, head_dim]
        value_states: [batch, heads, seq_len, head_dim]
        pruning_ratio: Fraction of tokens to remove.
        attention_sink_size: Number of initial tokens to protect.
    """
    seq_len = key_states.size(2)
    if seq_len <= attention_sink_size:
        return key_states, value_states
    
    # Simple importance: L2 norm of keys across head_dim
    # In a real scenario, we might use attention scores from the previous step.
    # For a "minimal" version, we'll use a local importance proxy.
    importance = torch.norm(key_states, dim=-1) # [batch, heads, seq_len]
    
    # Protect attention sinks
    importance[:, :, :attention_sink_size] = float('inf')
    
    num_to_keep = max(attention_sink_size, int(seq_len * (1 - pruning_ratio)))
    
    # Find indices of top-k tokens
    _, top_indices = torch.topk(importance, num_to_keep, dim=-1, sorted=True)
    top_indices = top_indices.sort(dim=-1).values # Keep original order
    
    # Gather tokens
    # Note: This is a simplified gather. Real transformer integration would be more complex.
    batch_size, num_heads, _, head_dim = key_states.shape
    
    # Reshape for gathering
    key_states = key_states.view(batch_size * num_heads, seq_len, head_dim)
    value_states = value_states.view(batch_size * num_heads, seq_len, head_dim)
    top_indices = top_indices.view(batch_size * num_heads, num_to_keep)
    
    # Create index for gather
    idx = top_indices.unsqueeze(-1).expand(-1, -1, head_dim)
    
    pruned_keys = torch.gather(key_states, 1, idx)
    pruned_values = torch.gather(value_states, 1, idx)
    
    # Reshape back
    pruned_keys = pruned_keys.view(batch_size, num_heads, num_to_keep, head_dim)
    pruned_values = pruned_values.view(batch_size, num_heads, num_to_keep, head_dim)
    
    return pruned_keys, pruned_values

class MinimalKVPruner:
    def __init__(self, pruning_ratio=0.5, attention_sink_size=4):
        self.pruning_ratio = pruning_ratio
        self.attention_sink_size = attention_sink_size
        
    def __call__(self, key_states, value_states):
        return prune_kv_cache(key_states, value_states, self.pruning_ratio, self.attention_sink_size)
