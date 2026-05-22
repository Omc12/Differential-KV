from typing import Dict, Any, List
import time

class OrchestrationEfficiencyTracker:
    """
    Tracks serving efficiency and hardware utilization.
    Provides metrics for orchestration scaling and runtime performance.
    """
    def __init__(self):
        self.metrics_log = []

    def record_event(self, event_type: str, duration: float, hardware_stats: Dict[str, float]):
        """Records an orchestration event with timing and hardware context."""
        self.metrics_log.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "duration": duration,
            "hardware": hardware_stats
        })

    def get_efficiency_report(self) -> Dict[str, Any]:
        """Generates a report on orchestration efficiency."""
        if not self.metrics_log:
            return {"status": "NO_DATA"}

        avg_duration = sum(e["duration"] for e in self.metrics_log) / len(self.metrics_log)
        
        return {
            "total_events": len(self.metrics_log),
            "avg_latency": avg_duration,
            "utilization": self._compute_utilization()
        }

    def _compute_utilization(self) -> Dict[str, float]:
        # Aggregate CPU/Mem from logs
        if not self.metrics_log: return {}
        cpu = sum(e["hardware"].get("cpu_percent", 0) for e in self.metrics_log) / len(self.metrics_log)
        mem = sum(e["hardware"].get("mem_percent", 0) for e in self.metrics_log) / len(self.metrics_log)
        return {"avg_cpu": cpu, "avg_mem": mem}
