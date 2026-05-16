import logging

logger = logging.getLogger(__name__)

class AdaptiveExecutionOffload:
    """
    Monitors runtime entropy. If an edge node struggles (entropy spikes),
    this system offloads the heaviest reasoning manifolds to the cloud.
    """
    def __init__(self, entropy_threshold: float = 0.8):
        self.entropy_threshold = entropy_threshold

    def evaluate_offload(self, current_entropy: float, active_manifolds: list) -> list:
        """
        Returns a list of manifolds that should be offloaded to cloud to stabilize the edge.
        """
        if current_entropy < self.entropy_threshold:
            return [] # No offload needed

        logger.warning(f"High entropy detected ({current_entropy}). Calculating offload targets...")
        
        # Simple heuristic: offload the first half of manifolds
        offload_count = max(1, len(active_manifolds) // 2)
        targets = active_manifolds[:offload_count]
        
        logger.info(f"Offloading manifolds: {targets}")
        return targets
