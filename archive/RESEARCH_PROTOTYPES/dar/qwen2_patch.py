import torch
from dar.minimal_kv_pruning import prune_kv_cache

def apply_kv_reduction(past_key_values, pruning_ratio=0.3):
    """
    Manually prune the KV cache after a step.
    Handles both tuple and DynamicCache formats.
    """
    if past_key_values is None:
        return None
    
    # Check if it's a DynamicCache (or similar)
    if hasattr(past_key_values, "key_cache"):
        # For DynamicCache, key_cache is a list of tensors [batch, heads, seq, dim]
        for i in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[i]
            v = past_key_values.value_cache[i]
            pk, pv = prune_kv_cache(k, v, pruning_ratio=pruning_ratio)
            past_key_values.key_cache[i] = pk
            past_key_values.value_cache[i] = pv
        return past_key_values

    # Handle legacy tuple format
    new_kv = []
    for k, v in past_key_values:
        pk, pv = prune_kv_cache(k, v, pruning_ratio=pruning_ratio)
        new_kv.append((pk, pv))
    
    return tuple(new_kv)
