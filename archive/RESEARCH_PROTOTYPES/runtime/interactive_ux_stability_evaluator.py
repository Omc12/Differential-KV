import time
import logging
from typing import Dict, Any, List

class InteractiveUXStabilityEvaluator:
    """
    RHU Phase 40.3: Interactive UX Stability Evaluator.
    Measures perceived interaction stability (smoothness, jitter, etc.).
    """
    def __init__(self):
        self.stability_metrics = {} # session_id -> metrics
        self.logger = logging.getLogger("UXEvaluator")

    def record_token_arrival(self, session_id: str, ts: float):
        """
        Tracks inter-token latency to measure smoothness and jitter.
        """
        if session_id not in self.stability_metrics:
            self.stability_metrics[session_id] = {
                "last_ts": ts,
                "intervals": [],
                "jitter": 0.0,
                "smoothness": 1.0
            }
        
        state = self.stability_metrics[session_id]
        interval = ts - state["last_ts"]
        if interval > 0:
            state["intervals"].append(interval)
            
            # Calculate jitter (variance in intervals)
            if len(state["intervals"]) > 5:
                avg = sum(state["intervals"][-10:]) / len(state["intervals"][-10:])
                variance = sum((x - avg)**2 for x in state["intervals"][-10:]) / 10
                state["jitter"] = variance ** 0.5
                
                # Smoothness is inverse of jitter relative to avg
                state["smoothness"] = max(0.0, 1.0 - (state["jitter"] / (avg + 1e-6)))

        state["last_ts"] = ts

    def get_ux_score(self, session_id: str) -> float:
        return self.stability_metrics.get(session_id, {}).get("smoothness", 1.0)

    def get_average_ux_stability(self) -> float:
        if not self.stability_metrics:
            return 1.0
        scores = [s["smoothness"] for s in self.stability_metrics.values()]
        return sum(scores) / len(scores)

    def detect_latency_spike(self, session_id: str, threshold: float = 1.0) -> bool:
        intervals = self.stability_metrics.get(session_id, {}).get("intervals", [])
        if intervals and intervals[-1] > threshold:
            self.logger.warning(f"UX Stability Alert: Latency spike ({intervals[-1]:.2f}s) in session {session_id}")
            return True
        return False
