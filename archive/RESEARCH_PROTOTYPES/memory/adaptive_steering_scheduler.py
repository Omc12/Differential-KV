import torch
import math

class AdaptiveSteeringScheduler:
    """
    PHASE 20.5: ALFSR - Adaptive Steering Scheduler.
    
    Dynamically scales steering strength based on:
    - Retrieval confidence (entropy of logits at steering-relevant tokens)
    - Symbolic continuity (how many span tokens were already predicted correctly)
    - Decoder entropy (health of the probability distribution)
    - Step count (decay over generation time)
    
    RULES:
    - high uncertainty  -> stronger guidance (but capped)
    - stabilized retrieval -> steering decays
    - strong continuity -> near-zero intervention
    FORBIDDEN: static global steering.
    """

    def __init__(
        self,
        base_strength: float = 1.0,
        max_strength: float = 8.0,
        min_strength: float = 0.0,
        decay_rate: float = 0.15,
        stabilization_threshold: float = 0.05,
    ):
        self.base_strength = base_strength
        self.max_strength = max_strength
        self.min_strength = min_strength
        self.decay_rate = decay_rate
        self.stabilization_threshold = stabilization_threshold

        # State
        self.step = 0
        self.retrieval_confirmed = False
        self.confirmation_step = -1
        self.current_strength = base_strength

    def step_update(
        self,
        logit_entropy: float,
        span_attention_mass: float,
        continuation_correct: bool,
    ) -> float:
        """
        Called once per decoding step. Returns the steering strength for this step.

        Args:
            logit_entropy:        Shannon entropy of the full logit distribution (nats).
            span_attention_mass:  Fraction of attention mass already on the symbolic span (0-1).
            continuation_correct: Whether the token generated last step was in the symbolic span.
        
        Returns:
            Steering strength to apply this step (>= 0).
        """
        self.step += 1

        # --- 1. Detect stabilization ---
        if not self.retrieval_confirmed and span_attention_mass >= self.stabilization_threshold:
            self.retrieval_confirmed = True
            self.confirmation_step = self.step

        # --- 2. Compute raw strength from uncertainty ---
        # Higher entropy => more uncertain => more guidance needed.
        # Logit entropy for a 7B vocab is typically 2-8 nats for normal generation.
        # We normalise into [0, 1] using a soft sigmoid centred at 5 nats.
        uncertainty_signal = 1.0 / (1.0 + math.exp(-(logit_entropy - 5.0)))

        raw_strength = self.base_strength + (self.max_strength - self.base_strength) * uncertainty_signal

        # --- 3. Apply decay after retrieval confirmation ---
        if self.retrieval_confirmed:
            steps_since_confirm = self.step - self.confirmation_step
            decay_factor = math.exp(-self.decay_rate * steps_since_confirm)
            raw_strength *= decay_factor

        # --- 4. Near-zero if strong continuity ---
        if continuation_correct and span_attention_mass > 0.5:
            raw_strength *= 0.1

        # --- 5. Clamp to [min, max] ---
        self.current_strength = max(self.min_strength, min(self.max_strength, raw_strength))
        return self.current_strength

    def reset(self):
        self.step = 0
        self.retrieval_confirmed = False
        self.confirmation_step = -1
        self.current_strength = self.base_strength
