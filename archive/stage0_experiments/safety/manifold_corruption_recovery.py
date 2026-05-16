import logging
import numpy as np

logger = logging.getLogger(__name__)

class ManifoldCorruptionRecovery:
    """
    Detects and rolls back corrupted latent states (e.g., due to NaN propagation
    or adversarial resonance collapse) before they infect the persistent snapshot.
    """
    def __init__(self, nan_tolerance: int = 0):
        self.nan_tolerance = nan_tolerance

    def scan_tensor(self, tensor: np.ndarray) -> bool:
        """Scans a latent tensor for corruption (NaNs or Infs). Returns True if corrupted."""
        has_nans = np.isnan(tensor).any()
        has_infs = np.isinf(tensor).any()
        
        if has_nans or has_infs:
            logger.error("CORRUPTION DETECTED: Latent tensor contains NaNs or Infs!")
            return True
        return False

    def rollback(self, current_state, last_good_state):
        """Reverts the active runtime state to the last known good configuration."""
        logger.info("Executing manifold rollback to last stable checkpoint.")
        return last_good_state
