import torch
from .persistent_relevance_resolver import PersistentRelevanceResolver
from memory.structural_anchor_detector import StructuralAnchorDetector
from memory.anchor_relative_capsules import AnchorRelativeCapsuleManager
from memory.prefix_suffix_guardian import PrefixSuffixGuardian
from memory.microboundary_preservation import MicroboundaryPreserver
from memory.reinforcement_budget_controller import ReinforcementBudgetController
from memory.symbolic_transition_stabilizer import SymbolicTransitionStabilizer

class AnchorReinforcementResolver(PersistentRelevanceResolver):
    """
    PHASE 18.9: Anchor-Relative Reinforcement & Symbolic Boundary Stabilization (ARRSBS).
    Specifically targets lead-in preservation and structural anchoring.
    """
    def __init__(self, tokenizer, anchor_budget=4096, fidelity_token_budget=1024):
        super().__init__(anchor_budget=anchor_budget, fidelity_token_budget=fidelity_token_budget)
        self.tokenizer = tokenizer
        self.anchor_detector = StructuralAnchorDetector(tokenizer)
        self.anchor_manager = AnchorRelativeCapsuleManager(self.anchor_detector)
        self.guardian = PrefixSuffixGuardian(edge_size=12) # Increased edge size
        self.micro_preserver = MicroboundaryPreserver()
        self.stabilizer = SymbolicTransitionStabilizer(transition_window=12) # Smoother transition
        self.budget_controller = ReinforcementBudgetController(max_reinforced_tokens=fidelity_token_budget)

    def resolve_and_prune(self, past_key_values, hidden_states, input_ids):
        """
        Executes ARRSBS resolution and pruning with structural backbone protection.
        """
        seq_len = hidden_states.size(1)
        device = hidden_states.device
        
        # 1. Base Symbolic Detection (from 18.8)
        # We use a very sensitive threshold here because anchors will help filter noise
        symbolic_mask = self.fidelity.detect_high_entropy_tokens(hidden_states, threshold=1.2)
        base_spans = self._mask_to_spans(symbolic_mask)
        
        # 2. Structural Anchor Detection (18.9A)
        anchor_indices = self.anchor_detector.get_anchor_indices(input_ids)
        
        # 3. Anchor-Relative Reinforcement (18.9B)
        # snap to anchors and ensure lead-in buffer
        reinforced_spans = self.anchor_manager.apply_reinforcement(input_ids, base_spans, lead_in_buffer=24)
        
        # 4. Symbolic Boundary Hardening (18.9C)
        edge_mask = self.guardian.protect_edges(reinforced_spans, seq_len)
        micro_mask = self.micro_preserver.protect_microboundaries(input_ids, self.tokenizer, seq_len)
        
        # Combined protection masks
        final_protection_mask = (edge_mask.to(device) | micro_mask.to(device))
        
        # 5. Injection into Geometry Manager
        self.geometry.update_importance(hidden_states, past_key_values[0][0].shape[2])
        
        if self.geometry.accumulated_importance is not None:
            imp_len = self.geometry.accumulated_importance.shape[1]
            self.budget_controller.reset()
            
            # Use a finite but extremely large value relative to current magnitudes
            max_val = torch.max(self.geometry.accumulated_importance).item()
            dtype_max = torch.finfo(self.geometry.accumulated_importance.dtype).max
            pin_value = min(dtype_max, max(1e4, max_val * 10.0))
            
            # A. Protect Reinforced Spans (Capsules)
            for start, end in reinforced_spans:
                g_start = max(0, imp_len - seq_len + start)
                g_end = min(imp_len, imp_len - seq_len + end + 1)
                num_tokens = g_end - g_start
                if num_tokens > 0 and self.budget_controller.request_protection(num_tokens, priority=1):
                    self.geometry.accumulated_importance[:, g_start:g_end] = pin_value
            
            # B. Structural Backbone: Protect regions IMMEDIATELY following anchors
            # often identifiers follow a "Context:" or ":" marker.
            for a_idx in anchor_indices:
                a_idx_item = a_idx.item()
                # Protect the anchor itself and the following 16 tokens
                g_start = max(0, imp_len - seq_len + a_idx_item)
                g_end = min(imp_len, g_start + 16)
                if self.budget_controller.request_protection(g_end - g_start, priority=2):
                    self.geometry.accumulated_importance[:, g_start:g_end] = pin_value
            
            # C. Micro-boundaries and Edges
            combined_indices = torch.where(final_protection_mask)[0]
            for idx in combined_indices:
                g_idx = max(0, imp_len - seq_len + idx.item())
                if g_idx < imp_len:
                    if self.budget_controller.request_protection(1, priority=2):
                        self.geometry.accumulated_importance[:, g_idx] = pin_value

            # 6. Apply Transition Smoothing
            full_symbolic_mask = torch.zeros((1, seq_len), device=device, dtype=torch.bool)
            for start, end in reinforced_spans:
                full_symbolic_mask[0, start:end+1] = True
            
            target_region = self.geometry.accumulated_importance[:, imp_len-seq_len:]
            smoothed_region = self.stabilizer.apply_smoothing(target_region, full_symbolic_mask)
            self.geometry.accumulated_importance[:, imp_len-seq_len:] = smoothed_region

        # 7. Pruning Execution
        pruned_kv, indices = self.geometry.prune_kv(past_key_values, hidden_states=None)
        
        # 8. Metadata and State Update
        self.chunk_idx += 1
        metadata = {
            'indices': indices,
            'anchors_detected': len(anchor_indices),
            'reinforced_spans': reinforced_spans,
            'budget_utilization': self.budget_controller.get_utilization()
        }
        
        return pruned_kv, metadata

    def _mask_to_spans(self, mask):
        spans = []
        if mask.any():
            indices = torch.where(mask[0])[0].tolist()
            if indices:
                start = indices[0]
                for i in range(1, len(indices)):
                    if indices[i] != indices[i-1] + 1:
                        spans.append((start, indices[i-1]))
                        start = indices[i]
                spans.append((start, indices[-1]))
        return spans
