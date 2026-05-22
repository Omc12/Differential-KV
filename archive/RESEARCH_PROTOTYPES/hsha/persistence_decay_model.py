
import math
import time
from typing import Dict

class PersistenceDecayModel:
    """
    PHASE 21.4: LSCP - Persistence Decay Model.
    Calculates fading strength for symbolic objects over time.
    """
    def __init__(self, half_life_seconds: float = 3600):
        self.half_life = half_life_seconds
        self.initial_persistence = 1.0

    def calculate_persistence(self, last_access_time: float) -> float:
        """
        Calculates current persistence strength using exponential decay.
        """
        elapsed = time.time() - last_access_time
        # P(t) = P0 * e^(-lambda * t) where lambda = ln(2) / half_life
        decay_constant = math.log(2) / self.half_life
        persistence = self.initial_persistence * math.exp(-decay_constant * elapsed)
        return max(0.0, persistence)

    def should_forget(self, last_access_time: float, threshold: float = 0.01) -> bool:
        """Returns True if persistence has dropped below the visibility threshold."""
        return self.calculate_persistence(last_access_time) < threshold
