from typing import Dict, Any

class SemanticDriftReactiveFallback:
    """
    STAGE 2 - SRI: Semantic Drift Reactive Fallback
    Dynamically overrides governance suppression and forces fallback 
    when semantic drift pressure becomes too high.
    """
    def __init__(self, drift_threshold: float = 0.08, pressure_decay: float = 0.9):
        self.drift_threshold = drift_threshold
        self.pressure_decay = pressure_decay
        self.current_pressure = 0.0
        self.forced_fallbacks = 0
        
    def update_pressure(self, current_drift: float):
        """Updates the semantic pressure based on recent drift."""
        # Decay existing pressure
        self.current_pressure *= self.pressure_decay
        # Add new pressure if drift exceeds threshold
        if current_drift > self.drift_threshold:
            excess = current_drift - self.drift_threshold
            self.current_pressure += excess
            
    def requires_forced_fallback(self, pressure_limit: float = 0.05) -> bool:
        """
        Determines if the accumulated semantic pressure demands an immediate dense fallback,
        regardless of what the governance layer wants.
        """
        if self.current_pressure > pressure_limit:
            self.forced_fallbacks += 1
            # Reset pressure slightly after forcing fallback to avoid oscillation
            self.current_pressure *= 0.5 
            return True
        return False
        
    def get_status(self) -> Dict[str, float]:
        return {
            "semantic_pressure": self.current_pressure,
            "forced_fallbacks": self.forced_fallbacks
        }
