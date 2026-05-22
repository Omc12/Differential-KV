import time
import logging
from typing import Dict, Any, List

class OperationalHealthMonitor:
    """
    Tracks serving uptime, crash frequency, and operational stability.
    Maintains production-grade operational visibility.
    """
    def __init__(self):
        self.logger = logging.getLogger("OperationalHealthMonitor")
        self.start_time = time.time()
        self.crash_count = 0
        self.last_recovery_time = None
        self.stability_window = [] # List of (timestamp, is_healthy)

    def record_heartbeat(self, is_healthy: bool = True):
        self.stability_window.append((time.time(), is_healthy))
        if len(self.stability_window) > 1000:
            self.stability_window.pop(0)

    def record_crash(self):
        self.crash_count += 1
        self.last_recovery_time = time.time()
        self.logger.error(f"Operational Health: CRASH detected. Total crashes: {self.crash_count}")

    def get_health_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        
        # Calculate uptime % from stability window
        if not self.stability_window:
            uptime_pct = 100.0
        else:
            healthy_samples = sum(1 for _, h in self.stability_window if h)
            uptime_pct = (healthy_samples / len(self.stability_window)) * 100
            
        return {
            "uptime_seconds": float(uptime),
            "uptime_pct": float(uptime_pct),
            "crash_count": int(self.crash_count),
            "last_recovery_timestamp": self.last_recovery_time,
            "is_stable": bool(uptime_pct > 99.0 and self.crash_count < 5)
        }

    def get_operational_stability_index(self) -> float:
        metrics = self.get_health_metrics()
        # Simple index: uptime_pct - (crashes * 5)
        idx = metrics["uptime_pct"] - (metrics["crash_count"] * 5)
        return float(max(0.0, idx))
