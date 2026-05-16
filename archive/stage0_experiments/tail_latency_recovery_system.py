import time
import logging
from typing import Dict, List, Any

class TailLatencyRecoverySystem:
    """
    Implements p95/p99 stabilization, starvation prevention, 
    and scheduler fairness recovery.
    """
    def __init__(self, p99_threshold_ms: float = 1000.0):
        self.logger = logging.getLogger("TailLatencyRecoverySystem")
        self.p99_threshold_ms = p99_threshold_ms
        self.latency_history = []
        self.starvation_counters = {} # session_id -> queue_wait_start

    def monitor_latency(self, latency_ms: float):
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > 1000:
            self.latency_history.pop(0)

    def check_starvation(self, session_id: str, arrival_time: float) -> bool:
        """
        Returns True if a session has been waiting for too long.
        """
        wait_time = (time.time() - arrival_time) * 1000
        if wait_time > self.p99_threshold_ms:
            self.logger.warning(f"Session {session_id} is STARVING (wait: {wait_time:.2f}ms). Escalating priority.")
            return True
        return False

    def get_recovery_action(self, current_p95: float) -> str:
        """
        Determines if we need to take action to stabilize tail latency.
        """
        if current_p95 > self.p99_threshold_ms * 0.8:
            return "REDUCE_BATCH_SIZE"
        elif current_p95 > self.p99_threshold_ms:
            return "FLUSH_QUEUE"
        return "NONE"

    def get_tail_metrics(self) -> Dict[str, Any]:
        import numpy as np
        if not self.latency_history:
            return {"p50": 0, "p95": 0, "p99": 0}
        
        arr = np.array(self.latency_history)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99))
        }
