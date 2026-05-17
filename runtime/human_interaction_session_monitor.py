import time
import logging
from typing import Dict, Any, List, Optional

class HumanInteractionSessionMonitor:
    """
    RHU Phase 40.3: Human Interaction Session Monitor.
    Tracks real human usage patterns and conversational continuity.
    """
    def __init__(self):
        self.usage_patterns = {} # session_id -> data
        self.logger = logging.getLogger("HumanMonitor")

    def log_interaction(self, session_id: str, event_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Logs human-driven events like message cadence, edits, and retries.
        """
        if session_id not in self.usage_patterns:
            self.usage_patterns[session_id] = {
                "events": [],
                "message_count": 0,
                "edit_count": 0,
                "retry_count": 0,
                "last_event_ts": time.time()
            }
        
        state = self.usage_patterns[session_id]
        ts = time.time()
        cadence = ts - state["last_event_ts"]
        
        event = {
            "ts": ts,
            "type": event_type,
            "cadence": cadence,
            "metadata": metadata or {}
        }
        
        state["events"].append(event)
        state["last_event_ts"] = ts
        
        if event_type == "message": state["message_count"] += 1
        elif event_type == "edit": state["edit_count"] += 1
        elif event_type == "retry": state["retry_count"] += 1

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        return self.usage_patterns.get(session_id, {})

    def get_global_cadence(self) -> float:
        cadences = []
        for s in self.usage_patterns.values():
            cadences.extend([e["cadence"] for e in s["events"] if e["cadence"] > 0])
        return sum(cadences) / len(cadences) if cadences else 0.0
