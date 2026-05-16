import time
import numpy as np
import logging
from typing import Dict, List, Any

class RealStreamingStabilityEngine:
    """
    Tracks token flush cadence, inter-token jitter, and streaming smoothness.
    Ensures real-time interaction quality.
    """
    def __init__(self):
        self.logger = logging.getLogger("StreamingStabilityEngine")
        self.session_tracks = {} # session_id -> list of timestamps

    def record_token_flush(self, session_id: str):
        """
        Records the timestamp of a token being flushed to the user.
        """
        if session_id not in self.session_tracks:
            self.session_tracks[session_id] = []
        self.session_tracks[session_id].append(time.time())

    def get_stability_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Calculates jitter and burstiness for a specific session.
        """
        timestamps = self.session_tracks.get(session_id, [])
        if len(timestamps) < 2:
            return {"itl_avg_ms": 0, "itl_jitter_ms": 0, "burstiness": 0}
            
        intervals = np.diff(timestamps) * 1000 # to ms
        avg_itl = np.mean(intervals)
        jitter = np.std(intervals)
        
        # Burstiness: coefficient of variation or similar
        burstiness = jitter / avg_itl if avg_itl > 0 else 0
        
        return {
            "itl_avg_ms": float(avg_itl),
            "itl_jitter_ms": float(jitter),
            "burstiness": float(burstiness),
            "token_count": len(timestamps)
        }

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Aggregates metrics across all active sessions.
        """
        all_itls = []
        all_jitters = []
        
        for sid in self.session_tracks:
            m = self.get_stability_metrics(sid)
            if m["token_count"] >= 2:
                all_itls.append(m["itl_avg_ms"])
                all_jitters.append(m["itl_jitter_ms"])
                
        if not all_itls:
            return {"avg_itl_ms": 0, "avg_jitter_ms": 0}
            
        return {
            "avg_itl_ms": float(np.mean(all_itls)),
            "avg_jitter_ms": float(np.mean(all_jitters)),
            "p95_itl_ms": float(np.percentile(all_itls, 95)) if all_itls else 0
        }

    def clear(self):
        self.session_tracks = {}
