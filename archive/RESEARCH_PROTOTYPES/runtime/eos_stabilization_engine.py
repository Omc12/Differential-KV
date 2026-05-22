import numpy as np
from typing import Dict, Any, List

class EOSStabilizationEngine:
    """
    Stage 4B.0 GFP: EOS Stabilization Engine.
    Prevents premature generative termination in sparse serving by gating and
    dampening EOS signals while evaluating narrative continuity and semantic flow.
    """
    def __init__(
        self, 
        eos_token_id: int = 151643, # Default for Qwen2.5 models
        dampening_factor: float = 0.25,
        min_tokens: int = 200,
        delay_steps: int = 3,
        continuity_threshold: float = 0.65
    ):
        self.eos_token_id = eos_token_id
        self.dampening_factor = dampening_factor
        self.min_tokens = min_tokens
        self.delay_steps = delay_steps
        self.continuity_threshold = continuity_threshold
        
        # Telemetry metrics
        self.total_eos_triggers = 0
        self.premature_eos_prevented = 0
        self.premature_eos_allowed = 0
        self.continuation_recovery_count = 0
        self.delayed_confirmations = 0
        
        # Sliding continuity state
        self.semantic_continuity_history = []
        self.eos_pending_confirmations = 0

    def adjust_eos_logits(
        self, 
        logits: np.ndarray, 
        current_length: int, 
        semantic_continuity_score: float
    ) -> np.ndarray:
        """
        Dampens the EOS logit if generation length is under the min threshold
        or if semantic continuity indicates narrative momentum is still high.
        """
        if logits is None or len(logits) <= self.eos_token_id:
            return logits

        # 1. EOS confidence dampening based on length
        len_dampening = 1.0
        if current_length < self.min_tokens:
            # Strong dampening for early sequence steps
            len_dampening = float(current_length) / float(self.min_tokens)
        
        # 2. Dampening based on semantic continuity
        # High continuity means the model is in the middle of a rich thought
        continuity_dampening = 1.0 - max(0.0, min(1.0, semantic_continuity_score))
        
        total_dampening = len_dampening * continuity_dampening * self.dampening_factor
        
        # Logit scaling (convert raw logits to dampened logits)
        if logits[self.eos_token_id] > 0:
            logits[self.eos_token_id] *= total_dampening
        else:
            logits[self.eos_token_id] /= (total_dampening + 1e-6)

        return logits

    def evaluate_gating(
        self, 
        sampled_token_id: int, 
        current_length: int, 
        semantic_continuity_score: float
    ) -> bool:
        """
        Determines whether an EOS token should be gated (suppressed) and replaced by 
        the next most probable token to preserve narrative flow.
        """
        if sampled_token_id != self.eos_token_id:
            # No EOS sampled; reset pending confirmation
            self.eos_pending_confirmations = 0
            return False
            
        self.total_eos_triggers += 1
        
        # Continuation-aware EOS Gating
        is_premature = current_length < self.min_tokens
        high_momentum = semantic_continuity_score > self.continuity_threshold
        
        if is_premature or high_momentum:
            # We encountered a premature EOS
            if self.eos_pending_confirmations < self.delay_steps:
                self.eos_pending_confirmations += 1
                self.delayed_confirmations += 1
                self.premature_eos_prevented += 1
                return True # Gated!
            else:
                # Allowed to leak through
                self.premature_eos_allowed += 1
                self.eos_pending_confirmations = 0 # Reset
                
        return False # Let EOS pass (natural termination reached)

    def score_semantic_continuity(self, recent_tokens: List[int], token_entropy: float) -> float:
        """
        Scores continuation probability by analyzing token diversity (type-token ratio)
        and local step entropy. High entropy/diversity signals active explanation.
        """
        if not recent_tokens:
            return 1.0
            
        unique_tokens = len(set(recent_tokens))
        ttr = unique_tokens / len(recent_tokens) # Type-Token Ratio
        
        # Combine TTR and localized entropy to derive continuity momentum
        # High token diversity + steady entropy = stable flowing narrative
        continuity_score = (ttr * 0.6) + (min(1.0, token_entropy / 4.0) * 0.4)
        
        self.semantic_continuity_history.append(continuity_score)
        if len(self.semantic_continuity_history) > 20:
            self.semantic_continuity_history.pop(0)
            
        return float(np.mean(self.semantic_continuity_history))

    def recover_continuation(self) -> float:
        """
        Calculates the recovery rate of narrative continuation when premature EOS was avoided.
        """
        total_premature = self.premature_eos_prevented + self.premature_eos_allowed
        if total_premature == 0:
            return 1.0
        
        recovery_rate = float(self.premature_eos_prevented) / float(total_premature)
        self.continuation_recovery_count += 1
        return recovery_rate

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns telemetry metrics gathered during validation/serving.
        """
        total_premature = self.premature_eos_prevented + self.premature_eos_allowed
        if total_premature == 0:
            rec_pct = 100.0
            prem_rate = 0.0
        else:
            rec_pct = (self.premature_eos_prevented / total_premature) * 100.0
            prem_rate = float(self.premature_eos_allowed) / total_premature
            
        avg_continuity = np.mean(self.semantic_continuity_history) if self.semantic_continuity_history else 0.85
        return {
            "eos_trigger_frequency": self.total_eos_triggers,
            "premature_eos_rate": prem_rate,
            "continuation_recovery_pct": rec_pct,
            "delayed_eos_confirmations": self.delayed_confirmations,
            "semantic_continuation_score": float(avg_continuity)
        }
