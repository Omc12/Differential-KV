import time
from typing import List, Dict, Any
import numpy as np

class SparseQoSStabilizer:
    """
    PSR System 2: Sparse QoS Stabilizer.
    Ensures latency stabilization, fairness, and prevents starvation under sparse load.
    """
    def __init__(self, target_p99_ms: float = 100.0):
        self.target_p99_ms = target_p99_ms
        self.latency_history: List[float] = []
        self.user_shares: Dict[str, int] = {}
        self.priority_map: Dict[str, float] = {}

    def update_latency_metric(self, latency_ms: float):
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > 1000:
            self.latency_history.pop(0)

    def calculate_fairness_score(self, user_id: str) -> float:
        """Calculates a fairness score based on current user share and latency."""
        total_shares = sum(self.user_shares.values()) or 1
        current_share = self.user_shares.get(user_id, 0)
        target_share = 1.0 / (len(self.user_shares) or 1)
        
        # Lower score means higher priority
        score = (current_share / total_shares) / target_share
        return max(0.01, score)

    def adjust_batch_window(self) -> float:
        """Dynamically adjusts the batch window to stabilize P99 latency."""
        if not self.latency_history:
            return 50.0  # Default 50ms window
        
        p99 = np.percentile(self.latency_history, 99)
        if p99 > self.target_p99_ms:
            return max(10.0, 50.0 * (self.target_p99_ms / p99))
        else:
            return min(200.0, 50.0 * (self.target_p99_ms / p99))

    def get_qos_control_signals(self, active_sessions: List[str]) -> Dict[str, Any]:
        """Generates QoS control signals for the scheduler."""
        signals = {}
        for session_id in active_sessions:
            signals[session_id] = {
                "priority": 1.0 / self.calculate_fairness_score(session_id),
                "is_starved": self.calculate_fairness_score(session_id) < 0.5
            }
        
        signals["global_batch_window"] = self.adjust_batch_window()
        return signals

    def suppress_tail_latency(self, request_id: str, current_latency: float):
        """Identifies and flags requests that are exceeding QoS targets."""
        if current_latency > self.target_p99_ms * 1.5:
            # Flag for aggressive sparse execution or priority boost
            return True
        return False
