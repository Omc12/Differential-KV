"""
hardware_materialization/runtime_degradation_monitor.py

Monitors Differential KV runtime performance to detect degradation over time.
"""

import logging
import time
from collections import deque
from typing import Dict, List, Any

logger = logging.getLogger("DegradationMonitor")

class RuntimeDegradationMonitor:
    """
    Tracks TPS, occupancy, and latency to detect progressive performance collapse.
    """
    def __init__(self, window_size: int = 100):
        self.tps_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.start_tps = None

    def record_step(self, tokens: int, latency_ms: float):
        """Records metrics for a single serving step."""
        tps = tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0
        self.tps_history.append(tps)
        self.latency_history.append(latency_ms)
        
        if self.start_tps is None and len(self.tps_history) > 10:
            self.start_tps = sum(self.tps_history) / len(self.tps_history)

    def get_degradation_index(self) -> float:
        """
        Calculates degradation: (Initial TPS - Current TPS) / Initial TPS.
        0.0 = no degradation, 1.0 = total collapse.
        """
        if self.start_tps is None or len(self.tps_history) < 10:
            return 0.0
            
        current_tps = sum(list(self.tps_history)[-10:]) / 10.0
        degradation = (self.start_tps - current_tps) / self.start_tps
        return max(0.0, degradation)

    def check_instability(self) -> bool:
        """Detects high variance in latency (jitter)."""
        if len(self.latency_history) < 10:
            return False
        
        lats = list(self.latency_history)
        avg = sum(lats) / len(lats)
        variance = sum((x - avg)**2 for x in lats) / len(lats)
        
        # If std_dev > 50% of avg, consider it unstable
        if (variance**0.5) > (avg * 0.5):
            logger.warning(f"High runtime jitter detected: avg={avg:.2f}ms, std={variance**0.5:.2f}ms")
            return True
        return False
