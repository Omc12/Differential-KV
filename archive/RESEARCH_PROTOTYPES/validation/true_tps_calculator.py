import time
import logging

class TrueTPSCalculator:
    """
    Calculates physically plausible TPS by enforcing strict timing boundaries.
    Prevents asynchronous timing leakage and microsecond artifacts.
    """
    def __init__(self, model_size_params=None):
        self.logger = logging.getLogger("TrueTPSCalculator")
        self.model_size_params = model_size_params

    def calculate_tps(self, token_count, start_time, end_time):
        duration = end_time - start_time
        
        # Enforce minimum physical duration to prevent division by near-zero
        # A single token generation on modern GPUs takes at least ~1-10ms
        min_duration = 0.001 * token_count 
        
        if duration < min_duration:
            self.logger.warning(f"Detected physically implausible duration: {duration:.6f}s for {token_count} tokens. Clamping to {min_duration:.6f}s")
            duration = min_duration
            
        tps = token_count / duration if duration > 0 else 0
        return tps

    def validate_plausibility(self, tps, hardware="A100", model_size="7B"):
        """
        Rejects TPS values that exceed theoretical hardware limits.
        """
        # Rough limits for single-node generation
        limits = {
            "7B": 200,
            "70B": 50,
            "0.5B": 1000
        }
        limit = limits.get(model_size, 500)
        
        if tps > limit:
            return False, f"TPS {tps:.2f} exceeds physical limit for {model_size} ({limit})"
        return True, "OK"
