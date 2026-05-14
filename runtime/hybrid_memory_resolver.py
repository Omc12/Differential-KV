import torch
from memory.semantic_geometry_tracker import SemanticGeometryKVManager
from memory.symbolic_fidelity_registry import SymbolicFidelityRegistry

class HybridMemoryResolver:
    """
    PHASE 18.6C: Hybrid Semantic + Symbolic Resolver.
    Orchestrates the interplay between continuity pathways and exact symbolic spans.
    """
    def __init__(self, anchor_budget: int = 6144, fidelity_budget: int = 512):
        self.geometry = SemanticGeometryKVManager(anchor_budget=anchor_budget)
        self.fidelity = SymbolicFidelityRegistry(fidelity_budget=fidelity_budget)
        self.global_offset = 0

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids):
        """
        1. Detect symbolic spans in the current chunk.
        2. Update the fidelity registry.
        3. Prune KV cache while PROTECTING symbolic spans.
        """
        # A. Symbolic Detection
        symbolic_mask = self.fidelity.detect_high_entropy_tokens(hidden_states)
        q_len = hidden_states.shape[1]
        
        # Absolute indices for this chunk
        chunk_indices = torch.arange(self.global_offset, self.global_offset + q_len, device=hidden_states.device)
        self.global_offset += q_len
        
        # B. Sync Geometry Manager (Update importance)
        # We temporarily inject the symbolic mask into the importance scores
        # in the next step to ensure they are protected.
        
        # C. Enhanced Pruning with Structural Symbolic Protection
        if self.geometry.accumulated_importance is not None:
            # Find the absolute indices of symbolic tokens in the CURRENT chunk
            chunk_sym_idx = torch.where(symbolic_mask[0])[0]
            imp_len = self.geometry.accumulated_importance.shape[1]
            
            # Protect the NEIGHBORHOOD of each symbolic token with infinite importance
            # Radius of 16 (32 total tokens) provides a stable structural runway
            for local_idx in chunk_sym_idx:
                start = max(0, imp_len - q_len + local_idx - 16)
                end = min(imp_len, imp_len - q_len + local_idx + 16)
                self.geometry.accumulated_importance[:, start:end] = float('inf')

        pruned_pkv, indices = self.geometry.prune_kv(past_key_values, hidden_states=hidden_states)
        return pruned_pkv, indices
