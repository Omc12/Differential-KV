import time
import json
from typing import Dict, List, Any, Optional

class SustainedDecodeContinuityMonitor:
    """
    STAGE 2 DQO: Sustained Decode Continuity Monitor.
    Tracks decode gaps, idle intervals, and overlap persistence using real timestamps.
    """
    def __init__(self, trace_path: str = "traces/stage2/phase_38_8_dqo/live_decode_windows.jsonl"):
        self.trace_path = trace_path
        self.last_decode_finish: Optional[float] = None
        self.active_window_start: Optional[float] = None
        
        self.decode_gaps: List[float] = []
        self.active_durations: List[float] = []
        self.overlap_counts: List[int] = []
        
        # Continuity Metrics
        self.total_monitored_time = 0.0
        self.total_decode_time = 0.0
        self.monitoring_start_ts = time.time()
        
    def record_step_start(self, active_session_count: int):
        now = time.time()
        
        if self.last_decode_finish is not None:
            gap = now - self.last_decode_finish
            if gap > 0:
                self.decode_gaps.append(gap)
        
        if self.active_window_start is None:
            self.active_window_start = now
            
        self.overlap_counts.append(active_session_count)

    def record_step_finish(self):
        now = time.time()
        if self.active_window_start is not None:
            duration = now - self.active_window_start
            self.active_durations.append(duration)
            self.total_decode_time += duration
            
            # Log to trace
            self._log_window(self.active_window_start, now, duration)
            
        self.last_decode_finish = now
        self.active_window_start = None

    def _log_window(self, start: float, end: float, duration: float):
        entry = {
            "timestamp": start,
            "end_timestamp": end,
            "duration_ms": duration * 1000,
            "gap_since_last_ms": (self.decode_gaps[-1] * 1000) if self.decode_gaps else 0
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_continuity_score(self) -> float:
        now = time.time()
        total_elapsed = now - self.monitoring_start_ts
        if total_elapsed <= 0:
            return 1.0
        return self.total_decode_time / total_elapsed

    def get_metrics(self) -> Dict[str, Any]:
        avg_gap = sum(self.decode_gaps) / len(self.decode_gaps) if self.decode_gaps else 0
        avg_overlap = sum(self.overlap_counts) / len(self.overlap_counts) if self.overlap_counts else 0
        
        return {
            "continuity_score": self.get_continuity_score(),
            "avg_decode_gap_ms": avg_gap * 1000,
            "avg_overlap": avg_overlap,
            "sustained_windows": len(self.active_durations)
        }
