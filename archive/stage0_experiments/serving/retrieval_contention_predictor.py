import numpy as np
from typing import List

class RetrievalContentionPredictor:
    """
    PHASE 7.5C: Retrieval Contention Predictor
    Analyzes multi-user request patterns to predict impending VRAM 
    bandwidth bottlenecks before they cause P95 latency spikes.
    """
    def __init__(self, history_len: int = 50):
        self.throughput_history = collections.deque(maxlen=history_len)
        self.latency_history = collections.deque(maxlen=history_len)

    def update_metrics(self, tps: float, latency_ms: float):
        """Updates the predictor with latest serving metrics."""
        self.throughput_history.append(tps)
        self.latency_history.append(latency_ms)

    def predict_contention(self) -> float:
        """
        Returns a contention probability (0.0 to 1.0).
        High probability suggests that the system is reaching saturation.
        """
        if len(self.latency_history) < 10:
            return 0.0
            
        # Check for rising latency trend while throughput stays flat (saturation)
        recent_latency = list(self.latency_history)[-10:]
        latency_trend = np.polyfit(range(len(recent_latency)), recent_latency, 1)[0]
        
        recent_tps = list(self.throughput_history)[-10:]
        tps_trend = np.polyfit(range(len(recent_tps)), recent_tps, 1)[0]
        
        # If latency is increasing fast but TPS is not, contention is likely
        if latency_trend > 5.0 and tps_trend <= 0.5:
            return min(1.0, latency_trend / 20.0)
            
        return 0.0

import collections
# Added collections import inside for robustness in this environment
