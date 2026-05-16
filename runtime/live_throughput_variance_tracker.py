import time
import json
import numpy as np
from typing import Dict, List, Any

class LiveThroughputVarianceTracker:
    """
    STAGE 2 DQO: Live Throughput Variance Tracker.
    Tracks tokens/sec variance, jitter, and stream smoothness.
    """
    def __init__(self, trace_path: str = "traces/stage2/phase_38_8_dqo/live_throughput_trace.jsonl"):
        self.trace_path = trace_path
        self.session_token_timestamps: Dict[str, List[float]] = {}
        self.global_token_counts = 0
        self.start_ts = time.time()
        
        # Windows for variance calculation
        self.throughput_history: List[float] = []
        self.last_window_ts = time.time()
        self.tokens_in_window = 0
        self.window_size_sec = 1.0

    def record_tokens(self, session_id: str, count: int):
        now = time.time()
        if session_id not in self.session_token_timestamps:
            self.session_token_timestamps[session_id] = []
        
        for _ in range(count):
            self.session_token_timestamps[session_id].append(now)
        
        self.tokens_in_window += count
        self.global_token_counts += count
        
        # Update windowed throughput
        if now - self.last_window_ts >= self.window_size_sec:
            tps = self.tokens_in_window / (now - self.last_window_ts)
            self.throughput_history.append(tps)
            self._log_throughput(now, tps)
            
            self.tokens_in_window = 0
            self.last_window_ts = now

    def _log_throughput(self, ts: float, tps: float):
        entry = {
            "timestamp": ts,
            "tokens_per_sec": tps,
            "active_sessions": len(self.session_token_timestamps)
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_variance_metrics(self) -> Dict[str, Any]:
        if len(self.throughput_history) < 2:
            return {"tps_variance": 0.0, "tps_jitter": 0.0, "current_tps": 0.0}
        
        tps_array = np.array(self.throughput_history[-10:]) # Last 10 windows
        variance = np.var(tps_array)
        jitter = np.mean(np.abs(np.diff(tps_array)))
        
        return {
            "tps_variance": float(variance),
            "tps_jitter": float(jitter),
            "current_tps": float(tps_array[-1])
        }

    def get_fairness_score(self) -> float:
        """Jain's Fairness Index for per-session throughput."""
        if not self.session_token_timestamps:
            return 1.0
            
        now = time.time()
        rates = []
        for sid, ts_list in self.session_token_timestamps.items():
            if not ts_list:
                continue
            duration = now - ts_list[0]
            if duration > 0:
                rates.append(len(ts_list) / duration)
        
        if not rates:
            return 1.0
            
        sum_rates = sum(rates)
        sum_sq_rates = sum(r*r for r in rates)
        
        return (sum_rates**2) / (len(rates) * sum_sq_rates) if sum_sq_rates > 0 else 1.0
