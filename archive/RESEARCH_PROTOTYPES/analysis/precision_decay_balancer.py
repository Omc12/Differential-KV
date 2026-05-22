import math

class PrecisionDecayBalancer:
    """
    SPS Phase 20.6: Precision Decay Balancer.
    Balances precision reinforcement vs decoder freedom.
    """
    def __init__(self, initial_factor: float = 1.0, decay_rate: float = 0.15):
        self.initial_factor = initial_factor
        self.decay_rate = decay_rate
        self.current_factor = initial_factor
        self.steps_since_reset = 0
        
    def step(self, drift_risk: float, coherence: float) -> float:
        """
        Calculates the stabilization factor for the next token.
        Increases reinforcement if risk is high or coherence is low.
        Decays reinforcement if sequence is stable.
        """
        # Base decay
        self.steps_since_reset += 1
        decay = math.exp(-self.decay_rate * self.steps_since_reset)
        
        # Reactive boost
        # If risk > 0.5, we re-spike stabilization
        boost = 0.0
        if drift_risk > 0.4:
            boost = drift_risk * 1.5
            self.steps_since_reset = max(0, self.steps_since_reset - 2) # Slow down decay
            
        # Coherence penalty
        if coherence < 0.8:
            boost += (1.0 - coherence)
            
        self.current_factor = min(2.0, (self.initial_factor * decay) + boost)
        return self.current_factor

    def reset(self):
        self.current_factor = self.initial_factor
        self.steps_since_reset = 0
