import torch
import time
from runtime.adaptive_salience_resolver import AdaptiveSalienceResolver
from memory.symbolic_capture_lock import SymbolicCaptureLock

class LockedSalienceResolver(AdaptiveSalienceResolver):
    """
    PHASE 20.2: SCLCPP Resolver with Generation-Time Locking.
    Ensures contiguous payload preservation during both prefill and generation.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        # Aggressive Lock for 20.2 tuning
        self.capture_lock = SymbolicCaptureLock(lock_duration=64, trigger_threshold=1.2)
        self.generation_lock_counter = 0

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        start_time = time.perf_counter()
        self.step_count += 1
        q_len = hidden_states.shape[1]
        
        # 1. Base ASSCIM Salience
        salience_scores = self.salience_model.estimate_salience(hidden_states)
        
        # 2. Adaptive Thresholding
        threshold = self.threshold_scheduler.adjust_threshold(hidden_states)
        
        # 3. Query Relevance Prediction
        relevance_boost = self.query_predictor.predict_relevance(chunk_input_ids, salience_scores)
        boosted_salience = salience_scores * relevance_boost
        
        # 4. Symbolic Capture Lock (Prefill)
        lock_mask = self.capture_lock.update_locks(boosted_salience)
        
        # 5. Importance Estimation
        importance_weights = self.importance_estimator.calculate_importance(boosted_salience)
        
        # 6. Relational Propagation
        chunk_indices = torch.arange(self.global_offset, self.global_offset + q_len, device=hidden_states.device)
        importance_weights = self.relational_graph.propagate_importance(importance_weights, chunk_indices)
        
        # 7. Routing & Span Expansion
        fidelity_mask, final_importance = self.visibility_router.route_tokens(importance_weights, boosted_salience)
        fidelity_mask |= lock_mask
        
        # 8. Update Fidelity Registry
        symbolic_mask = fidelity_mask | (importance_weights > 0.7)
        self.fidelity.update_fidelity_cache(past_key_values, symbolic_mask, chunk_indices)
        
        # 9. Geometry Reinforcement
        seq_len = past_key_values[0][0].shape[2]
        self.geometry.update_importance(hidden_states, seq_len)
        if self.geometry.accumulated_importance is not None:
             # Force absolute protection for locked tokens in the geometry buffer
             self.geometry.accumulated_importance[:, -q_len:][lock_mask] = 100000.0
        
        # 10. Generation Lock Activation
        if lock_mask[0, -1]:
            self.generation_lock_counter = 64
        
        # Update global offset
        self.global_offset += q_len
        
        # PRUNE
        pruned_pkv, indices = self.geometry.prune_kv(past_key_values, hidden_states=None)
        
        # Track overhead
        self.salience_overhead.record(time.perf_counter() - start_time)
        
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Intensified decoder guidance during generation lock.
        """
        # 1. Base Calibration (DTASCC)
        calibrated_logits = super().guide_decoder(logits)
        
        # 2. Generation Lock Application (Stronger Bias)
        if self.generation_lock_counter > 0:
            self.generation_lock_counter -= 1
            # Intensify calibration strength significantly
            # This forces the model to look at the high-fidelity symbolic cache
            calibrated_logits *= 2.0
            
        return calibrated_logits
