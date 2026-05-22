import time
import logging

logger = logging.getLogger(__name__)

class EntropyPressureMonitor:
    """
    Tracks the accumulation of cognitive entropy over time across sessions.
    Fires alerts if the rate of entropy increase exceeds sustainable bounds.
    """
    def __init__(self, warning_threshold: float = 0.7):
        self.warning_threshold = warning_threshold
        self.node_entropy_history = {}

    def log_entropy(self, node_id: str, entropy_val: float):
        if node_id not in self.node_entropy_history:
            self.node_entropy_history[node_id] = []
            
        self.node_entropy_history[node_id].append((time.time(), entropy_val))
        
        if entropy_val > self.warning_threshold:
            logger.warning(f"ENTROPY PRESSURE ALERT: Node {node_id} at {entropy_val:.2f}")

    def get_pressure_trend(self, node_id: str) -> float:
        """Calculates the derivative of entropy over the last 5 samples."""
        history = self.node_entropy_history.get(node_id, [])
        if len(history) < 2:
            return 0.0
        
        recent = history[-5:]
        dt = recent[-1][0] - recent[0][0]
        de = recent[-1][1] - recent[0][1]
        
        return de / dt if dt > 0 else 0.0
