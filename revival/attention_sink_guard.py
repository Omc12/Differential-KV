import torch

class AttentionSinkGuard:
    """
    Ensures that critical 'sink' tokens are never evicted or pruned from the KV cache.
    """

    def __init__(self, num_sink_tokens=4):
        self.num_sink_tokens = num_sink_tokens
        self.sink_indices = list(range(num_sink_tokens))

    def protect_cache(self, k, v, mask=None):
        """
        Applies protection to the KV tensors. 
        In practice, this means ensuring these indices are excluded from pruning logic.
        """
        # Return a boolean mask of tokens that MUST be kept
        seq_len = k.shape[-2]
        protection_mask = torch.zeros(seq_len, dtype=torch.bool, device=k.device)
        
        # Ensure sink tokens are always protected
        effective_sinks = min(self.num_sink_tokens, seq_len)
        protection_mask[:effective_sinks] = True
        
        return protection_mask

    def apply_guarded_pruning(self, k, v, pruning_mask):
        """
        Overrides a pruning mask to ensure sinks are preserved.
        pruning_mask: True where we want to KEEP the token.
        """
        seq_len = k.shape[-2]
        sink_mask = torch.zeros(seq_len, dtype=torch.bool, device=k.device)
        effective_sinks = min(self.num_sink_tokens, seq_len)
        sink_mask[:effective_sinks] = True
        
        # Guarded mask: Keep if pruning logic says keep OR if it's a sink
        guarded_mask = pruning_mask | sink_mask
        
        return k[:, :, guarded_mask, :], v[:, :, guarded_mask, :]

if __name__ == "__main__":
    guard = AttentionSinkGuard(num_sink_tokens=2)
    k = torch.randn(1, 8, 10, 64)
    v = torch.randn(1, 8, 10, 64)
    pruning_mask = torch.zeros(10, dtype=torch.bool) # Prune everything
    
    gk, gv = guard.apply_guarded_pruning(k, v, pruning_mask)
    print(f"Original seq_len: 10, Guarded seq_len: {gk.shape[2]}")
    assert gk.shape[2] == 2, "Sink guard failed to protect tokens."
