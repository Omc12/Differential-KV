import torch
import time
from typing import Optional, List, Dict

from runtime.alfsr_resolver import ALFSRResolver
from memory.symbolic_precision_field import SymbolicPrecisionField
from analysis.symbolic_precision_analysis import LocalTokenCoherenceTracker, SymbolicDriftPredictor, PrecisionEntropyAuditor
from analysis.precision_decay_balancer import PrecisionDecayBalancer

class SPSResolver(ALFSRResolver):
    """
    SPS Phase 20.6: Symbolic Precision Stabilization Resolver.
    Extends ALFSR with probabilistic precision reinforcement.
    """
    def __init__(self, tokenizer, anchor_budget: int = 2048, fidelity_budget: int = 1024,
                 initial_strength: float = 1.0, max_strength: float = 8.0):
        super().__init__(tokenizer, anchor_budget, fidelity_budget, initial_strength, max_strength)
        self.tokenizer = tokenizer
        
        # SPS Modules
        self.precision_field = SymbolicPrecisionField(tokenizer)
        self.coherence_tracker = LocalTokenCoherenceTracker()
        self.drift_predictor = SymbolicDriftPredictor()
        self.entropy_auditor = PrecisionEntropyAuditor()
        self.decay_balancer = PrecisionDecayBalancer()
        self.precision_field.base_strength = 0.8 # Reduced for better entropy preservation
        
        # SPS State
        self.generated_token_ids = []
        self._confirmed_span_start = -1
        
        # Telemetry
        self.precision_trace = []
        
    def reset_generation_state(self):
        super().reset_generation_state()
        self.coherence_tracker = LocalTokenCoherenceTracker()
        self.entropy_auditor = PrecisionEntropyAuditor()
        self.decay_balancer.reset()
        self.generated_token_ids = []
        self._confirmed_span_start = -1
        self.precision_trace = []

    def guide_decoder(self, logits: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Phase 20.6: SPS-enhanced guidance.
        """
        if len(self.precision_trace) == 0:
            print(f"[DEBUG] SPSResolver.guide_decoder FIRST CALL. Booster Sequences: {len(self.booster.ordered_sequences)}")
        
        # Step 1: Base ALFSR Guidance
        # This handles the primary 'retrieval' steering
        was_confirmed_before = self._retrieval_confirmed
        base_guided_logits = super().guide_decoder(logits, attention_weights)
        
        # If retrieval was just confirmed, mark the offset
        if self._retrieval_confirmed and not was_confirmed_before:
             # We assume the span starts near the current generation step
             self._confirmed_span_start = len(self.generated_token_ids)
        
        # Step 2: SPS Precision Stabilization
        # Identify expected token from booster spans
        expected_token_id = self._get_expected_symbolic_token()
        
        if len(self.precision_trace) < 10:
             print(f"[DEBUG] SPS Step {len(self.precision_trace)} | Expected ID: {expected_token_id} | Confirmed: {self._retrieval_confirmed}")

        if expected_token_id is not None and expected_token_id != -1:
            # Phase 20.6: Probability Gate
            # Do NOT force if the model is extremely confident in something else
            # and the expected token is near-zero probability.
            probs = torch.softmax(base_guided_logits.float(), dim=-1)
            expected_prob = probs[0, expected_token_id].item()
            
            if expected_prob < 1e-4:
                if len(self.precision_trace) < 10:
                    print(f"[DEBUG] SPS Rejected: Prob {expected_prob:.6f} too low for ID {expected_token_id}")
                return base_guided_logits

            # Predict drift risk BEFORE applying stabilization
            risk = self.drift_predictor.predict_risk(base_guided_logits, expected_token_id)
            coherence = self.coherence_tracker.get_coherence_score()
            
            # Get stabilization factor from balancer
            stab_factor = self.decay_balancer.step(risk, coherence)
            
            # Apply precision field
            precision_bias = self.precision_field.get_precision_logits(
                expected_token_id, base_guided_logits, stab_factor
            )
            
            final_logits = base_guided_logits + precision_bias
            
            # Audit entropy of final logits
            audit_result = self.entropy_auditor.audit(final_logits)
            
            # Trace telemetry
            trace_entry = {
                "step": len(self.precision_trace),
                "expected_token_id": expected_token_id,
                "expected_token_str": self.tokenizer.decode([expected_token_id]),
                "drift_risk": risk,
                "coherence": coherence,
                "stab_factor": stab_factor,
                "entropy_nats": audit_result["entropy_nats"],
                "is_collapsed": audit_result["is_collapsed"]
            }
            self.precision_trace.append(trace_entry)
            
            # SPS Debug
            if len(self.precision_trace) % 5 == 0:
                print(f"[DEBUG] SPS: Step {trace_entry['step']} | Risk: {risk:.2f} | Stab: {stab_factor:.2f} | Entropy: {trace_entry['entropy_nats']:.3f} | Expected: '{trace_entry['expected_token_str']}'")
            
            return final_logits
            
        return base_guided_logits

    def _get_expected_symbolic_token(self) -> Optional[int]:
        """
        Infers the expected symbolic token from the currently active booster span.
        Pins alignment to the moment of retrieval confirmation.
        """
        # Phase 20.6: Use context-aware sequence alignment
        last_two = self.generated_token_ids[-2:] if len(self.generated_token_ids) >= 2 else None
        locked_ids = self.booster.get_ordered_symbolic_sequence(last_two)
        if not locked_ids:
            return None
            
        # Prioritize fixed offset if confirmed
        if self._retrieval_confirmed and self._confirmed_span_start != -1:
            offset = len(self.generated_token_ids) - self._confirmed_span_start
            if 0 <= offset < len(locked_ids):
                return locked_ids[offset]
        
        # Fallback for pre-confirmation or initial entry
        if not self.generated_token_ids:
            return locked_ids[0]

        # Fuzzy alignment if not yet pinned
        if len(self.generated_token_ids) >= 1:
            last_tokens = self.generated_token_ids[-2:] # will be 1 or 2 tokens
            for i in range(len(locked_ids) - len(last_tokens)):
                if locked_ids[i:i+len(last_tokens)] == last_tokens:
                    next_idx = i + len(last_tokens)
                    if next_idx < len(locked_ids):
                        return locked_ids[next_idx]
            
        return None

    def record_generated_token(self, token_id: int, logits: torch.Tensor):
        """
        Phase 20.6: Update coherence tracker.
        """
        # Track generated sequence
        self.generated_token_ids.append(token_id)
        
        # First, standard ALFSR tracking (updates _retrieval_confirmed)
        super().record_generated_token(token_id, logits)
        
        # Second, SPS coherence tracking
        expected_id = self._get_expected_symbolic_token()
        if expected_id is not None:
            self.coherence_tracker.record(token_id, expected_id)
