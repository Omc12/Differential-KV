import torch

class SemanticGeometryKVManager:
    """
    PHASE 18.5: Continuity-Aware Sparse KV Manager.
    Preserves semantic neighborhoods to prevent anchor isolation.
    Optimized for Phase 18.9 to favor recent context and symbolic stability.
    """
    def __init__(self, anchor_budget: int = 6144, local_window: int = 512, neighborhood_size: int = 16):
        self.budget = anchor_budget
        self.local = local_window
        self.neighborhood = neighborhood_size
        self.accumulated_importance = None
        self.absolute_indices = None # Tracks absolute sequence positions
        self.instruction_lane_size = 512 # Full system prompt + instruction protection

    def reset_accumulation(self):
        self.accumulated_importance = None
        self.absolute_indices = None

    def update_importance(self, hidden_states, seq_len):
        """
        Manually updates the importance buffer before pruning execution.
        """
        if hidden_states is None:
            return
            
        q_len = hidden_states.shape[1]
        device = hidden_states.device
        batch_size = hidden_states.shape[0]
        # Use L2-norm as base importance
        current_imp = torch.norm(hidden_states, p=2, dim=-1)
        
        if self.accumulated_importance is None:
            self.accumulated_importance = current_imp
            self.absolute_indices = torch.arange(q_len, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            diff = seq_len - self.accumulated_importance.shape[1]
            if diff > 0:
                padding = torch.zeros((batch_size, diff), device=device)
                self.accumulated_importance = torch.cat([self.accumulated_importance, padding], dim=1)
                
                # Update absolute indices
                new_indices = torch.arange(self.absolute_indices.max() + 1, self.absolute_indices.max() + 1 + diff, device=device)
                self.absolute_indices = torch.cat([self.absolute_indices, new_indices.unsqueeze(0).expand(batch_size, -1)], dim=1)
            
            # Update the tail with current importance
            self.accumulated_importance[:, -q_len:] = torch.max(self.accumulated_importance[:, -q_len:], current_imp)

    def prune_kv(self, past_key_values, hidden_states=None):
        if past_key_values is None:
            return None, None

        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        batch_size = past_key_values[0][0].shape[0]
        
        if hidden_states is not None:
            self.update_importance(hidden_states, seq_len)

        if seq_len <= self.budget:
            return past_key_values, None

        # 1. Importance Weighting
        importance = self.accumulated_importance.clone()
        
        # Temporal Weighting: Favor recent tokens
        indices_range = torch.arange(seq_len, device=device).float()
        distance_factor = 1.0 + (indices_range / seq_len)
        importance = importance * distance_factor
        
        # 2. Candidate Selection
        candidate_count = min(seq_len, self.budget * 2)
        _, top_indices = torch.topk(importance, k=candidate_count, dim=-1)
        
        # 3. Neighborhood Expansion & Protection
        mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
        mask[:, :self.instruction_lane_size] = True
        mask[:, -self.local:] = True
        
        for i in range(-self.neighborhood, self.neighborhood + 1):
            neighbor_indices = torch.clamp(top_indices + i, 0, seq_len - 1)
            mask.scatter_(1, neighbor_indices, True)
            
        # 4. Final Budget Enforcement
        importance[~mask] = -1.0
        _, final_indices = torch.topk(importance, k=self.budget, dim=-1)
        final_indices, _ = torch.sort(final_indices, dim=-1)
        
        # 5. Sync State and Prune KV
        self.accumulated_importance = torch.gather(self.accumulated_importance, 1, final_indices)
        self.absolute_indices = torch.gather(self.absolute_indices, 1, final_indices)

        new_pkv = []
        for layer_k, layer_v in past_key_values:
            num_heads = layer_k.shape[1]
            head_dim = layer_k.shape[3]
            idx_expanded = final_indices.view(batch_size, 1, -1, 1).expand(-1, num_heads, -1, head_dim)
            pruned_k = torch.gather(layer_k, 2, idx_expanded)
            pruned_v = torch.gather(layer_v, 2, idx_expanded)
            new_pkv.append((pruned_k, pruned_v))
            
        return tuple(new_pkv), final_indices
