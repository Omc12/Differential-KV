import time
import logging
from typing import Dict, Any, List, Optional

class LongSessionSemanticContinuityMonitor:
    """
    ORX Phase 40.2: Long-Session Semantic Continuity Monitor.
    Tracks continuity and degradation over extended interactions.
    """
    def __init__(self):
        self.session_continuity = {} # session_id -> score
        self.logger = logging.getLogger("ContinuityMonitor")

    def track_continuity(self, session_id: str, drift: float, tokens: int):
        """
        Updates the continuity score for a session based on drift and length.
        """
        if session_id not in self.session_continuity:
            self.session_continuity[session_id] = {
                "score": 1.0,
                "history": [],
                "total_tokens": 0
            }
        
        state = self.session_continuity[session_id]
        state["total_tokens"] += tokens
        
        # Continuity degrades with drift
        # Score = 1.0 / (1.0 + total_drift)
        current_score = state["score"]
        new_score = current_score * (1.0 - (drift * 0.1)) # Simple decay model
        state["score"] = max(0.0, new_score)
        state["history"].append({"ts": time.time(), "score": state["score"], "drift": drift})
        
        if state["score"] < 0.5:
            self.logger.warning(f"Semantic continuity low for session {session_id}: {state['score']:.2f}")

    def get_continuity_score(self, session_id: str) -> float:
        return self.session_continuity.get(session_id, {}).get("score", 1.0)

    def get_average_continuity(self) -> float:
        if not self.session_continuity:
            return 1.0
        scores = [s["score"] for s in self.session_continuity.values()]
        return sum(scores) / len(scores)

    def get_monitor_report(self) -> Dict[str, Any]:
        return {
            "avg_continuity": self.get_average_continuity(),
            "active_monitored_sessions": len(self.session_continuity)
        }
