import math
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DecaySnapshot:
    step: int
    raw_strength: float      # strength before decay
    decay_factor: float
    effective_strength: float
    trigger: str             # "init" | "natural" | "confirmed" | "forced_zero"

class SteeringDecayController:
    """
    PHASE 20.5: ALFSR - Steering Decay Controller.

    Reduces intervention strength over generation time:
    - Strong guidance only near retrieval initiation (first N steps)
    - Exponential decay after retrieval stabilisation
    - Restore natural decoder dynamics gradually

    GOAL: Avoid permanently constrained decoding.
    """

    def __init__(
        self,
        warmup_steps: int = 4,
        half_life_steps: float = 6.0,
        floor_strength: float = 0.0,
    ):
        """
        Args:
            warmup_steps:     Steps of full-strength steering at the start.
            half_life_steps:  Exponential decay half-life after warmup (in steps).
            floor_strength:   Minimum steering allowed (0 = full decay to nothing).
        """
        self.warmup_steps   = warmup_steps
        self.half_life      = half_life_steps
        self.floor_strength = floor_strength
        self.decay_lambda   = math.log(2) / max(half_life_steps, 1e-6)

        self.step           = 0
        self.confirm_step   = -1   # set when retrieval is confirmed
        self.log: List[DecaySnapshot] = []

    def compute(self, raw_strength: float, retrieval_confirmed: bool = False) -> float:
        """
        Compute effective steering strength for this decoding step.

        Args:
            raw_strength:         Strength computed by AdaptiveSteeringScheduler.
            retrieval_confirmed:  Whether span attention mass passed the threshold.

        Returns:
            Effective steering strength (>= floor_strength).
        """
        self.step += 1
        trigger = "natural"

        if retrieval_confirmed and self.confirm_step < 0:
            self.confirm_step = self.step
            trigger = "confirmed"

        # Warmup window: no decay
        if self.step <= self.warmup_steps:
            effective = raw_strength
            decay_factor = 1.0
            trigger = "init"
        elif self.confirm_step > 0:
            # Exponential decay from the confirmation point
            steps_elapsed = self.step - self.confirm_step
            decay_factor  = math.exp(-self.decay_lambda * steps_elapsed)
            effective     = max(self.floor_strength, raw_strength * decay_factor)
            trigger       = "confirmed"
        else:
            # No confirmation yet — mild linear ramp-down from warmup end
            steps_past_warmup = self.step - self.warmup_steps
            decay_factor = max(0.2, 1.0 - 0.05 * steps_past_warmup)
            effective = max(self.floor_strength, raw_strength * decay_factor)

        if effective <= self.floor_strength + 1e-6:
            trigger = "forced_zero"

        snap = DecaySnapshot(
            step=self.step,
            raw_strength=raw_strength,
            decay_factor=decay_factor,
            effective_strength=effective,
            trigger=trigger,
        )
        self.log.append(snap)
        return effective

    def export(self) -> List[dict]:
        return [
            {
                "step": s.step,
                "raw_strength": s.raw_strength,
                "decay_factor": s.decay_factor,
                "effective_strength": s.effective_strength,
                "trigger": s.trigger,
            }
            for s in self.log
        ]

    def reset(self):
        self.step         = 0
        self.confirm_step = -1
        self.log.clear()
