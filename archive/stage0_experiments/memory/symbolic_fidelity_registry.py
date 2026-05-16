import torch

class SymbolicFidelityRegistry:
    """
    PHASE 18.6A: Symbolic Fidelity Registry.
    Preserves exact symbolic spans (high-entropy tokens) separately from 
    the general sparse KV cache for precision reinforcement.
    """
    def __init__(self, fidelity_budget: int = 512):
        self.budget = fidelity_budget
        self.symbolic_spans = {} # {anchor_id: (tokens, kv_states)}
        self.fidelity_cache = None # [batch, num_heads, fidelity_len, head_dim]
        self.fidelity_indices = None # Absolute indices of symbolic tokens

    def detect_high_entropy_tokens(self, hidden_states, threshold: float = 3.0):
        """
        Identifies symbolic spans based on hidden state L2-norm variance.
        Threshold set to 3.0 for extreme selectivity in 18.8.
        """
        magnitudes = torch.norm(hidden_states, p=2, dim=-1)
        mean = magnitudes.mean(dim=-1, keepdim=True)
        std = magnitudes.std(dim=-1, keepdim=True)
        
        # Add stability floors
        std = torch.clamp(std, min=0.1)
        
        # Z-score based surprise detection
        z_scores = (magnitudes - mean) / std
        
        # Only consider tokens with magnitude significantly above mean
        symbolic_mask = (z_scores > threshold) & (magnitudes > mean * 1.1)
        return symbolic_mask

    def update_fidelity_cache(self, past_key_values, symbolic_mask, global_indices):
        """
        Extracts and stores high-fidelity KV states for symbolic tokens.
        """
        batch_size = past_key_values[0][0].shape[0]
        device = past_key_values[0][0].device
        
        # Find local indices of symbolic tokens
        local_symbolic_idx = torch.where(symbolic_mask)
        if local_symbolic_idx[0].numel() == 0:
            return

        # Get absolute indices
        abs_indices = global_indices[local_symbolic_idx[1]]
        
        if self.fidelity_indices is not None and self.fidelity_indices.numel() >= self.budget:
            return

        for layer_idx, (k, v) in enumerate(past_key_values):
            q_len = symbolic_mask.shape[1]
            chunk_k = k[:, :, -q_len:, :]
            chunk_v = v[:, :, -q_len:, :]
            
            pruned_k = chunk_k[:, :, symbolic_mask[0], :]
            pruned_v = chunk_v[:, :, symbolic_mask[0], :]
            
            if layer_idx == 0:
                self.fidelity_indices = abs_indices if self.fidelity_indices is None else torch.cat([self.fidelity_indices, abs_indices])
        
        return
