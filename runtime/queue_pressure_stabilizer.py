import time
import json
from typing import Dict, List, Any

class QueuePressureStabilizer:
    """
    STAGE 2 DQO: Queue Pressure Stabilizer.
    Reduces oscillation between idle and burst congestion.
    """
    def __init__(self, 
                 trace_path: str = "traces/stage2/phase_38_8_dqo/live_queue_pressure.jsonl",
                 damping_factor: float = 0.5):
        self.trace_path = trace_path
        self.damping_factor = damping_factor
        self.smoothed_pressure = 0.0
        self.last_update_ts = time.time()
        
    def update_pressure(self, current_queue_depth: int) -> float:
        """
        Apply Exponential Moving Average (EMA) to smooth out pressure signals.
        """
        now = time.time()
        delta = now - self.last_update_ts
        self.last_update_ts = now
        
        # Adjust alpha based on time delta if necessary, but keep it simple for now
        alpha = self.damping_factor
        self.smoothed_pressure = (alpha * current_queue_depth) + ((1 - alpha) * self.smoothed_pressure)
        
        self._log_pressure(current_queue_depth, self.smoothed_pressure)
        return self.smoothed_pressure

    def _log_pressure(self, raw: int, smoothed: float):
        entry = {
            "timestamp": time.time(),
            "raw_queue_depth": raw,
            "smoothed_pressure": smoothed
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def should_admit_request(self, current_queue_depth: int, max_queue_capacity: int) -> bool:
        """
        Anti-starvation and burst control logic.
        """
        if current_queue_depth >= max_queue_capacity:
            return False
            
        # If pressure is rising too fast, we might want to slow down admission
        # but for now, we prioritize throughput if capacity exists.
        return True
