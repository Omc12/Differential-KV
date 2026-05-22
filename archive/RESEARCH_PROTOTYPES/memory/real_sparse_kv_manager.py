import torch

class RealSparseKVManager:
    """
    PHASE 18.4H: Unified Persistent Importance Manager.
    Robustly tracks semantic importance across all prefill chunks.
    """
    def __init__(self, anchor_budget: int = 4096, local_window: int = 512):
        self.budget = anchor_budget
        self.local = local_window
        self.accumulated_importance = None

    def reset_accumulation(self):
        self.accumulated_importance = None

    def prune_kv(self, past_key_values, hidden_states=None):
        if past_key_values is None:
            return None, None

        # seq_len is the length of the KV cache AFTER the current chunk was added
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        batch_size = past_key_values[0][0].shape[0]

        # 1. Expand and Update Importance
        if hidden_states is not None:
            # hidden_states: [batch, q_len, hidden_dim]
            q_len = hidden_states.shape[1]
            current_imp = torch.norm(hidden_states, p=2, dim=-1) # [batch, q_len]
            
            if self.accumulated_importance is None:
                # First chunk: set buffer to current importance
                self.accumulated_importance = current_imp
            else:
                # Expand importance buffer to match current seq_len
                # accumulated_importance was pruned to 'budget' or is at previous seq_len
                diff = seq_len - self.accumulated_importance.shape[1]
                if diff > 0:
                    padding = torch.zeros((batch_size, diff), device=device)
                    self.accumulated_importance = torch.cat([self.accumulated_importance, padding], dim=1)
                
                # Update the most recent q_len tokens with their current importance
                # We use max() to preserve the peak activation seen for any token
                self.accumulated_importance[:, -q_len:] = torch.max(self.accumulated_importance[:, -q_len:], current_imp)

        # 2. Check if pruning is required
        if seq_len <= self.budget:
            return past_key_values, None

        # 3. Selection Strategy (Tri-Tier + Temporal Correction)
        importance = self.accumulated_importance.clone()
        
        # Temporal Weighting: Amplify importance of older tokens to counteract decay
        # Older tokens have lower indices. factor(0) = 2.0, factor(seq_len) = 1.0
        indices_range = torch.arange(seq_len, device=device).float()
        distance_factor = 2.0 - (indices_range / seq_len)
        importance = importance * distance_factor
        
        # Tier 1 & 2: Structural Anchors
        importance[:, :128] = float('inf') # Sinks
        importance[:, -self.local:] = float('inf') # Recent window
        
        # Tier 3: Semantic Anchors (Heavy Hitters)
        _, indices = torch.topk(importance, k=self.budget, dim=-1)
        indices, _ = torch.sort(indices, dim=-1)
        
        # 4. Sync State
        self.accumulated_importance = torch.gather(self.accumulated_importance, 1, indices)

        # 5. Physical Pruning
        new_pkv = []
        for layer_k, layer_v in past_key_values:
            num_heads = layer_k.shape[1]
            head_dim = layer_k.shape[3]
            idx_expanded = indices.view(batch_size, 1, -1, 1).expand(-1, num_heads, -1, head_dim)
            pruned_k = torch.gather(layer_k, 2, idx_expanded)
            pruned_v = torch.gather(layer_v, 2, idx_expanded)
            new_pkv.append((pruned_k, pruned_v))
            
        return tuple(new_pkv), indices
