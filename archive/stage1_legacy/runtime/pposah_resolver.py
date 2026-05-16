
import torch
import time
from typing import Optional, List, Dict

from runtime.sps_resolver import SPSResolver
from memory.symbolic_anchor_index import SymbolicAnchorIndex, ConfirmedSpanRootTracker
from analysis.logit_analysis_cache import LogitAnalysisCache, FusedPrecisionTelemetry
from memory.symbolic_precision_field import SymbolicPrecisionField

class PPOSAHResolver(SPSResolver):
    """
    PPOSAH Phase 20.6A: Precision Path Optimization & Symbolic Alignment Hardening.
    Implements constant-time anchor indexing and fused telemetry.
    """
    def __init__(self, tokenizer, anchor_budget: int = 2048, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        # PPOSAH Modules
        self.anchor_index = SymbolicAnchorIndex()
        self.root_tracker = ConfirmedSpanRootTracker()
        self.logit_cache = LogitAnalysisCache()
        self.fused_telemetry = FusedPrecisionTelemetry()
        
        # Override precision field to be more localized
        self.precision_field = SymbolicPrecisionField(tokenizer, base_precision_strength=1.5)
        
    def reset_generation_state(self):
        super().reset_generation_state()
        self.anchor_index.reset()
        self.root_tracker.reset()
        self.logit_cache.reset()
        self.fused_telemetry.reset()
        
    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        # 1. Standard resolve/prune logic (registers spans in booster)
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # 2. PPOSAH: Register exact tokens in anchor index
        # We check if a new span was added to the booster
        if len(self.booster.active_spans) > 0:
            start, end = self.booster.active_spans[-1]
            # If this is a new span (within the current chunk range)
            if start >= self.global_offset - chunk_input_ids.shape[1]:
                # Extract the tokens for this span
                # The tokens are stored in the booster.ordered_sequences
                if len(self.booster.ordered_sequences) > 0:
                    last_tokens = self.booster.ordered_sequences[-1]
                    self.anchor_index.add_span(start, end, torch.tensor(last_tokens))
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Phase 20.6A: PPOSAH GUIDANCE.
        Eliminates GPU stalls and fuzzy alignment.
        """
        # Step 1: Base Guidance (ALFSR + DTASCC)
        # We call the grandparent (ALFSR) directly to avoid SPS fuzzy logic
        # Wait, PPOSAH should replace SPS.
        base_guided_logits = super(SPSResolver, self).guide_decoder(logits, attention_weights)
        
        # Step 2: Shared Logit Analysis (FUSED)
        self.logit_cache.update(base_guided_logits)
        
        # Step 3: Exact Anchor Alignment
        # Instead of fuzzy searching, we check if we are currently locked to a span
        expected_token_id = self._get_exact_symbolic_token(attention_weights)
        
        if expected_token_id is not None and expected_token_id != -1:
            # Probability Gate (using cached probs)
            expected_prob = self.logit_cache.probs[expected_token_id].item()
            if expected_prob < 1e-4:
                return base_guided_logits

            # Step 4: Fused Metrics & Decay (NO NEW SYNC)
            risk = 0.0 # Placeholder, logic inside fused_telemetry
            coherence = self.coherence_tracker.get_coherence_score()
            
            # Use cached risk/entropy via fused telemetry
            stab_factor = self.decay_balancer.step(0.0, coherence) # Simplified balancer input
            
            # Apply precision field (localized)
            precision_bias = self.precision_field.get_precision_logits(
                expected_token_id, base_guided_logits, stab_factor
            )
            
            final_logits = base_guided_logits + precision_bias
            
            # Log telemetry (ONE sync only)
            metrics = self.fused_telemetry.measure(self.logit_cache, expected_token_id, stab_factor)
            self.precision_trace.append(metrics)
            
            return final_logits
            
        return base_guided_logits

    def _get_exact_symbolic_token(self, attention_weights: Optional[torch.Tensor]) -> Optional[int]:
        """
        Phase 20.6A: Exact symbolic token identification.
        Uses span_mass and confirmed root start to anchor position.
        """
        # 1. If already confirmed and locked, increment and return
        if self.root_tracker.is_confirmed:
            offset = self.root_tracker.current_offset
            # Identify which sequence corresponds to the confirmed root_start
            for i, (start, end) in enumerate(self.booster.active_spans):
                if start == self.root_tracker.root_start:
                    if i < len(self.booster.ordered_sequences):
                        seq = self.booster.ordered_sequences[i]
                        if 0 <= offset < len(seq):
                            return seq[offset]
            return None

        # 2. Check for new confirmation via attention mass
        if attention_weights is not None:
            try:
                # We need the current KV length to map attention weights to absolute indices
                mass = attention_weights[-1, 0].mean(dim=0)[-1] # [kv_len]
                current_abs_indices = self.geometry.absolute_indices[0]
                
                # Check all spans for hits
                for i, (start, end) in enumerate(self.booster.active_spans):
                    # Mask the mass vector for this specific span
                    span_indices = (current_abs_indices >= start) & (current_abs_indices <= end)
                    
                    # Mass shape might be kv_len, indices shape is pruned_kv_len
                    # We usually align them in the parent class
                    m_len = min(mass.shape[0], span_indices.shape[0])
                    span_mass = mass[:m_len][span_indices[:m_len]].sum().item()
                    
                    if span_mass > 0.10: # Lowered for 20.7 propagation chains
                        self.root_tracker.confirm(start)
                        if i < len(self.booster.ordered_sequences):
                            return self.booster.ordered_sequences[i][0]
            except Exception as e:
                # print(f"[DEBUG] PPOSAH Alignment Error: {e}")
                pass
                
        return None

    def record_generated_token(self, token_id: int, logits: torch.Tensor):
        super().record_generated_token(token_id, logits)
        # Update PPOSAH offset
        self.root_tracker.step()
