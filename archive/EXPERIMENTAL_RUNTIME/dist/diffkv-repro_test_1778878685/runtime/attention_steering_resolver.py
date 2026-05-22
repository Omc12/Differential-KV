import torch
import time
from typing import Optional, List, Tuple
from runtime.locked_salience_resolver import LockedSalienceResolver
from memory.dynamic_attention_booster import DynamicAttentionBooster

class AttentionSteeringResolver(LockedSalienceResolver):
    """
    PHASE 20.4: MSSPSR - Refined implementation.
    Refines the steering intensity and prevents attention collapse.
    Uses proper head-averaged mass calculation and index tracking.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        self.booster = DynamicAttentionBooster(boost_factor=5.0)
        self.logit_bias_strength = 15.0 

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        print(f"[DEBUG] AttentionSteeringResolver.resolve_and_prune called. Chunk size: {chunk_input_ids.shape[1]}")
        # 1. Base logic (20.2 Locks + 20.1 Salience)
        # This call handles global_offset and pruning.
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # 2. Register Locked Spans for Steering (20.3 - HARDENED)
        q_len = hidden_states.shape[1]
        # global_offset was already incremented by q_len in AdaptiveSalienceResolver
        chunk_indices = torch.arange(self.global_offset - q_len, self.global_offset, device=hidden_states.device)
        
        # Extract symbolic tokens using the salience model logic
        salience_scores = self.salience_model.estimate_salience(hidden_states)
        threshold = self.threshold_scheduler.adjust_threshold(hidden_states)
        relevance_boost = self.query_predictor.predict_relevance(chunk_input_ids, salience_scores)
        
        # PHASE 20.4: Use additive combination to ensure anchors are detected even with weak salience
        combined_score = salience_scores + (relevance_boost - 1.0)
        
        # Safety check for NaNs in combined_score
        if torch.isnan(combined_score).any():
            print(f"[WARNING] NaNs detected in combined_score! Forcing to zero.")
            combined_score = torch.nan_to_num(combined_score)
            
        max_score = combined_score.max().item()
        print(f"[DEBUG] Salience Max: {max_score:.4f}, Threshold: {threshold:.4f}")
        
        symbolic_mask = (combined_score > threshold)
        
        if symbolic_mask.any():
            # Get the indices where the mask is True for the first batch
            mask_indices = symbolic_mask[0].nonzero(as_tuple=True)[0]
            locked_idx = chunk_indices[mask_indices]
            locked_tokens = chunk_input_ids[0, mask_indices] # [num_tokens]
            
            if len(locked_idx) > 0:
                print(f"[DEBUG] Adding Booster Span: {int(locked_idx.min())}-{int(locked_idx.max())}, Tokens: {len(locked_tokens)}")
                self.booster.add_span(
                    int(locked_idx.min()), 
                    int(locked_idx.max()), 
                    tokens=locked_tokens
                )
        
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        PHASE 20.4: MSSPSR - Probabilistic steering based on attention mass.
        """
        # Ensure input logits are not NaN
        if torch.isnan(logits).any():
            print(f"[WARNING] NaNs detected in input logits! Neutralizing.")
            logits = torch.nan_to_num(logits)
            
        print(f"[DEBUG] guide_decoder called. Weights: {attention_weights is not None}")
        # 1. Base Calibration (Standard DTASCC + Generation Lock)
        calibrated_logits = super().guide_decoder(logits)
        
        # 2. Token-Space Steering (20.4 - PROBABILISTIC)
        locked_token_ids = self.booster.get_locked_token_ids()
        
        if locked_token_ids:
            token_indices = torch.tensor(locked_token_ids, device=calibrated_logits.device)
            token_indices = token_indices[token_indices < calibrated_logits.shape[-1]]
            
            if len(token_indices) > 0:
                steering_factor = 1.0 
                span_mass = 0.0
                
                if attention_weights is not None:
                    try:
                        # attention_weights shape: [layers, batch, heads, q_len, kv_len]
                        # We average across heads for the last query token of the last layer
                        # Safety: check weights for NaNs
                        if torch.isnan(attention_weights).any():
                            print(f"[WARNING] NaNs in attention_weights!")
                            attention_weights = torch.nan_to_num(attention_weights)
                            
                        mass = attention_weights[-1, 0].mean(dim=0)[-1] # [kv_len]
                        
                        # Use geometry absolute indices to mask the mass vector
                        current_abs_indices = self.geometry.absolute_indices[0]
                        steering_mask = self.booster.get_steering_bias(current_abs_indices, mass.device)
                        
                        # Fix: kv_len might include the current token's KV which is not in absolute_indices yet
                        if mass.shape[0] > steering_mask.shape[0]:
                            mass = mass[:steering_mask.shape[0]]
                        
                        # Calculate total attention mass on the target span
                        span_mass = mass[steering_mask > 1.0].sum().item()
                        
                        # MSSPSR: Decay steering as model gains attention confidence
                        if span_mass > 0.0:
                            steering_factor = max(0.0, 1.0 - (span_mass / 0.25))
                    except Exception as e:
                        print(f"[DEBUG] Attention Weight processing error: {e}")
                        pass
                
                if steering_factor > 0:
                    print(f"[DEBUG] Steering: {steering_factor:.2f}, Mass: {span_mass:.4f}, Tokens: {len(token_indices)}")
                    calibrated_logits[0, token_indices] += self.logit_bias_strength * steering_factor
        
        # FINAL SAFETY: Ensure calibrated_logits are valid
        calibrated_logits = torch.nan_to_num(calibrated_logits)
        
        return calibrated_logits
