import torch
import time
from runtime.calibrated_memory_resolver import CalibratedMemoryResolver
from memory.contextual_salience_model import ContextualSalienceModel
from memory.adaptive_importance_estimator import AdaptiveImportanceEstimator
from memory.symbolic_relevance_tracker import SymbolicRelevanceTracker
from memory.contextual_visibility_router import ContextualVisibilityRouter
from memory.relational_importance_graph import RelationalImportanceGraph
from memory.query_relevance_predictor import QueryRelevancePredictor
from memory.weak_signal_accumulator import WeakSignalAccumulator
from memory.adaptive_threshold_scheduler import AdaptiveThresholdScheduler
from analysis.salience_overhead_tracker import SalienceOverheadTracker

class AdaptiveSalienceResolver(CalibratedMemoryResolver):
    """
    PHASE 20.1: ASSCIM Resolver.
    Adaptive Symbolic Salience & Contextual Importance Modeling.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(anchor_budget, fidelity_budget)
        self.salience_model = ContextualSalienceModel()
        self.importance_estimator = AdaptiveImportanceEstimator(target_budget=anchor_budget)
        self.relevance_tracker = SymbolicRelevanceTracker()
        self.visibility_router = ContextualVisibilityRouter(fidelity_budget=fidelity_budget)
        self.relational_graph = RelationalImportanceGraph()
        self.query_predictor = QueryRelevancePredictor(tokenizer)
        self.weak_signal = WeakSignalAccumulator()
        self.threshold_scheduler = AdaptiveThresholdScheduler()
        
        self.salience_overhead = SalienceOverheadTracker()
        self.step_count = 0

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        start_time = time.perf_counter()
        self.step_count += 1
        q_len = hidden_states.shape[1]
        
        # 1. Salience & Importance (20.1A)
        salience_scores = self.salience_model.estimate_salience(hidden_states)
        
        # 2. Adaptive Thresholding (20.1D)
        threshold = self.threshold_scheduler.adjust_threshold(hidden_states)
        
        # 3. Query Relevance Prediction (20.1B)
        relevance_boost = self.query_predictor.predict_relevance(chunk_input_ids, salience_scores)
        boosted_salience = salience_scores * relevance_boost
        
        # 4. Importance Estimation
        # (Pass noise estimate if available, currently 0.0)
        importance_weights = self.importance_estimator.calculate_importance(boosted_salience)
        
        # 5. Weak Signal Accumulation (20.1C)
        chunk_indices = torch.arange(self.global_offset, self.global_offset + q_len, device=hidden_states.device)
        low_salience_mask = (boosted_salience < threshold) & (boosted_salience > threshold * 0.5)
        self.weak_signal.accumulate(chunk_indices[low_salience_mask[0]], boosted_salience[low_salience_mask])
        
        # Upgrade accumulated signals
        upgrade_mask = self.weak_signal.get_accumulation_mask(chunk_indices)
        importance_weights[0, upgrade_mask] = torch.maximum(importance_weights[0, upgrade_mask], torch.tensor(0.8, device=hidden_states.device))
        
        # 6. Relational Propagation (20.1B)
        # Update graph with sequential token relationships in the chunk
        for i in range(q_len - 1):
            if importance_weights[0, i] > 0.6:
                self.relational_graph.add_relationship(int(chunk_indices[i]), int(chunk_indices[i+1]), weight=0.5)
        
        importance_weights = self.relational_graph.propagate_importance(importance_weights, chunk_indices)
        
        # 7. Routing (20.1A)
        fidelity_mask, final_importance = self.visibility_router.route_tokens(importance_weights, boosted_salience)
        
        # 8. Absolute indices for this chunk
        self.global_offset += q_len
        
        # Detect all symbolic indices (including upgrades)
        symbolic_mask = (boosted_salience > threshold) | (importance_weights > 0.7)
        current_sym_idx = chunk_indices[symbolic_mask[0]]
        
        # Update fidelity registry
        self.fidelity.update_fidelity_cache(past_key_values, symbolic_mask, chunk_indices)
        
        # Update relevance tracker
        self.relevance_tracker.update_relevance(current_sym_idx)
        
        # Boost importance of historically relevant tokens
        relevance_boost_vec = self.relevance_tracker.get_boost_factor(chunk_indices).to(hidden_states.device)
        final_importance *= relevance_boost_vec
        
        # PRUNE
        pruned_pkv, indices = self.geometry.prune_kv(past_key_values, hidden_states=hidden_states)
        
        # Track overhead
        self.salience_overhead.record(time.perf_counter() - start_time)
        
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decoder guidance is inherited from CalibratedMemoryResolver (DTASCC).
        We use the same trust arbitration but with better importance signals.
        """
        return super().guide_decoder(logits)
