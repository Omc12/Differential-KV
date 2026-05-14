import torch
from .hybrid_memory_resolver import HybridMemoryResolver
from memory.persistent_relevance_tracker import PersistentRelevanceTracker
from memory.anticipatory_capsule_engine import AnticipatoryCapsuleEngine
from memory.multiscale_capsule_hierarchy import MultiScaleCapsuleHierarchy, CapsuleScale
from memory.resolution_sharpening_engine import ResolutionSharpeningEngine

class PersistentRelevanceResolver(HybridMemoryResolver):
    """
    PHASE 18.8: Persistent Relevance & Resolution Sharpening (PRMRS) Resolver.
    Integrates predictive relevance, anticipatory activation, and multi-scale capsules.
    """
    def __init__(self, anchor_budget=4096, fidelity_token_budget=1024):
        super().__init__(anchor_budget=anchor_budget, fidelity_budget=fidelity_token_budget)
        self.relevance_tracker = PersistentRelevanceTracker()
        self.anticipatory_engine = AnticipatoryCapsuleEngine(lookback_window=16, lookahead_window=8)
        self.hierarchy = MultiScaleCapsuleHierarchy()
        self.sharpening_engine = ResolutionSharpeningEngine()
        self.chunk_idx = 0

    def resolve_and_prune(self, past_key_values, hidden_states, input_ids):
        """
        Executes PRMRS resolution and pruning.
        """
        seq_len = hidden_states.size(1)
        
        # 1. Calculate Base Fidelity Signals
        # Selective threshold for symbolic detection
        symbolic_mask = self.fidelity.detect_high_entropy_tokens(hidden_states, threshold=2.2)
        
        # Convert mask to spans
        base_spans = []
        if symbolic_mask.any():
            indices = torch.where(symbolic_mask[0])[0].tolist()
            if indices:
                start = indices[0]
                for i in range(1, len(indices)):
                    if indices[i] != indices[i-1] + 1:
                        base_spans.append((start, indices[i-1]))
                        start = indices[i]
                base_spans.append((start, indices[-1]))
        
        # 2. Anticipatory Expansion & Persistence Tracking (18.8A, 18.8B)
        expanded_spans = self.anticipatory_engine.detect_and_expand(
            hidden_states, input_ids, self.chunk_idx, base_spans
        )
        
        # 3. Multi-Scale Capsule Allocation (18.8C)
        micro_capsules = self.hierarchy.allocate_capsules(
            [(s, e, 1.0) for s, e in expanded_spans], CapsuleScale.MICRO
        )
        
        # 4. Inject Capsule Protection into Geometry Manager
        # MANDATORY: Update importance buffer with current chunk's hidden states first
        # to ensure g_start:g_end indices exist in the buffer.
        self.geometry.update_importance(hidden_states, past_key_values[0][0].shape[2])
        
        if self.geometry.accumulated_importance is not None:
            imp_len = self.geometry.accumulated_importance.shape[1]
            for start, end in expanded_spans:
                g_start = max(0, imp_len - seq_len + start)
                g_end = min(imp_len, imp_len - seq_len + end + 1)
                self.geometry.accumulated_importance[:, g_start:g_end] = float('inf')

        # 5. Pruning Execution
        pruned_kv, indices = self.geometry.prune_kv(past_key_values, hidden_states=hidden_states)
        
        # 6. Metadata and State Update
        self.chunk_idx += 1
        metadata = {
            'indices': indices,
            'capsules': micro_capsules,
            'relevance_utilization': len(micro_capsules)
        }
        
        return pruned_kv, metadata
