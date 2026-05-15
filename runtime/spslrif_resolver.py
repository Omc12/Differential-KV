
import torch
import time
import numpy as np
from typing import Optional, List, Dict, Any

from runtime.pposah_resolver import PPOSAHResolver

class SymbolicIdentityFlowTracker:
    """Tracks exact symbolic identity propagation decay."""
    def __init__(self):
        self.propagation_chain = []
        self.first_mutation_pos = -1
        
    def reset(self):
        self.propagation_chain = []
        self.first_mutation_pos = -1
        
    def record_step(self, generated_token: int, expected_token: int, pos: int):
        is_match = (generated_token == expected_token)
        self.propagation_chain.append(is_match)
        
        if not is_match and self.first_mutation_pos == -1:
            self.first_mutation_pos = pos
            
    def get_continuity_score(self, window: int = 16) -> float:
        if not self.propagation_chain: return 1.0
        recent = self.propagation_chain[-window:]
        return sum(recent) / len(recent)

class IdentityLineageRegistry:
    """Maintains symbolic ancestry and continuation roots."""
    def __init__(self):
        self.active_root = None
        self.ancestry_chain = []
        
    def reset(self):
        self.active_root = None
        self.ancestry_chain = []
        
    def register_root(self, root_start: int):
        self.active_root = root_start
        self.ancestry_chain.append(root_start)

class ContinuityMomentumField:
    """Biases logit distributions to maintain symbolic momentum."""
    def __init__(self, base_momentum_boost: float = 0.2, decay_rate: float = 0.95):
        self.momentum = 0.0
        self.base_momentum_boost = base_momentum_boost
        self.decay_rate = decay_rate
        
    def reset(self):
        self.momentum = 0.0
        
    def update(self, is_match: bool):
        if is_match:
            self.momentum = min(1.0, self.momentum + self.base_momentum_boost)
        else:
            self.momentum *= self.decay_rate
            
    def get_multiplier(self) -> float:
        return 1.0 + self.momentum

class PropagationEntropyBalancer:
    """Ensures symbolic continuity does not collapse entropy."""
    def __init__(self, min_entropy: float = 0.4):
        self.min_entropy = min_entropy
        
    def reset(self):
        pass
        
    def is_safe(self, current_entropy: float) -> bool:
        return current_entropy > self.min_entropy

class SPSLRIFResolver(PPOSAHResolver):
    """
    Phase 20.7: Symbolic Propagation Stability & Long-Range Identity Flow.
    Extends PPOSAH with identity flow tracking and continuity momentum.
    """
    def __init__(self, tokenizer, anchor_budget: int = 2048, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        # 20.7 Modules
        self.flow_tracker = SymbolicIdentityFlowTracker()
        self.lineage_registry = IdentityLineageRegistry()
        self.momentum_field = ContinuityMomentumField()
        self.entropy_balancer = PropagationEntropyBalancer()
        
        # Propagation specific trace
        self.propagation_trace = []

    def reset_generation_state(self):
        super().reset_generation_state()
        self.flow_tracker.reset()
        self.lineage_registry.reset()
        self.momentum_field.reset()
        self.entropy_balancer.reset()
        self.propagation_trace = []

    def guide_decoder(self, logits: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Phase 20.7: SPSLRIF GUIDANCE.
        Consolidates PPOSAH with continuity momentum.
        """
        # Step 1: Base Guidance (ALFSR + DTASCC)
        # We call the grandparent (SPSResolver's parent, which is ALFSR)
        # Actually, let's just use the PPOSAH base_guided_logits logic
        from runtime.sps_resolver import SPSResolver
        base_guided_logits = super(SPSResolver, self).guide_decoder(logits, attention_weights)
        
        # Step 2: Fused Analysis (PPOSAH module)
        self.logit_cache.update(base_guided_logits)
        
        # Step 3: Exact Alignment (PPOSAH module)
        expected_token_id = self._get_exact_symbolic_token(attention_weights)
        
        if expected_token_id is not None and expected_token_id != -1:
            # Register Lineage (20.7)
            if self.root_tracker.is_confirmed and self.lineage_registry.active_root is None:
                print(f"[DEBUG] 20.7 LOCKED: Root {self.root_tracker.root_start} at pos {self.root_tracker.current_offset}")
                self.lineage_registry.register_root(self.root_tracker.root_start)

            # Entropy Check (20.7)
            if not self.entropy_balancer.is_safe(self.logit_cache.entropy):
                return base_guided_logits

            # Step 4: Apply Momentum-Boosted Precision (20.7)
            coherence = self.coherence_tracker.get_coherence_score()
            # stab_factor from decay_balancer (PPOSAH logic)
            stab_factor = self.decay_balancer.step(0.0, coherence)
            
            # Boost stab_factor with momentum
            boosted_factor = stab_factor * self.momentum_field.get_multiplier()
            
            # Apply precision field (localized)
            precision_bias = self.precision_field.get_precision_logits(
                expected_token_id, base_guided_logits, boosted_factor
            )
            
            final_logits = base_guided_logits + precision_bias
            
            # Log telemetry (FUSED)
            metrics = self.fused_telemetry.measure(self.logit_cache, expected_token_id, boosted_factor)
            self.precision_trace.append(metrics)
            
            return final_logits
            
        return base_guided_logits

    def record_generated_token(self, token_id: int, logits: torch.Tensor):
        """Phase 20.7: Propagation Tracking & Momentum Update."""
        # Get current expectation before stepping
        expected_token = self._get_current_expected_anchor()
        
        if expected_token is not None:
            is_match = (token_id == expected_token)
            self.flow_tracker.record_step(token_id, expected_token, self.root_tracker.current_offset)
            self.momentum_field.update(is_match)
            
            # Capture propagation trace
            self.propagation_trace.append({
                "pos": self.root_tracker.current_offset,
                "is_match": is_match,
                "momentum": self.momentum_field.momentum,
                "entropy": self.logit_cache.entropy,
                "first_mutation": self.flow_tracker.first_mutation_pos
            })

        # Step standard trackers (PPOSAH steps root_tracker)
        super().record_generated_token(token_id, logits)

    def _get_current_expected_anchor(self) -> Optional[int]:
        """Peeks at the anchor index for the current position."""
        if self.root_tracker.is_confirmed:
            offset = self.root_tracker.current_offset
            for i, (start, end) in enumerate(self.booster.active_spans):
                if start == self.root_tracker.root_start:
                    if i < len(self.booster.ordered_sequences):
                        seq = self.booster.ordered_sequences[i]
                        if 0 <= offset < len(seq):
                            return seq[offset]
        return None
