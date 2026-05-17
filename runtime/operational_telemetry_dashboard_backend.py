import time
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

class OperationalTelemetryDashboardBackend:
    """
    OIS Phase 40.1: Operational Telemetry Dashboard Backend.
    Provides real-time telemetry feeds for monitoring.
    """
    def __init__(self, trace_path: Optional[Path] = None):
        self.trace_path = trace_path
        self.metrics = {
            "active_sessions": 0,
            "batch_size": 0,
            "queue_depth": 0,
            "tokens_per_sec": 0.0,
            "semantic_recovery_activity": 0,
            "sparse_ratio": 0.0,
            "gpu_utilization": 0.0,
            "vram_usage_gb": 0.0,
            "recovery_frequency": 0.0
        }
        self.logger = logging.getLogger("TelemetryDashboard")

    def update_metrics(self, new_metrics: Dict[str, Any]):
        self.metrics.update(new_metrics)
        if self.trace_path:
            self._persist_metrics()

    def _persist_metrics(self):
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                rec = {
                    "timestamp": time.time(),
                    **self.metrics
                }
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to persist telemetry: {e}")

    def get_live_feed(self) -> Dict[str, Any]:
        """Returns the current state of all metrics."""
        return self.metrics

    def format_live_output(self) -> str:
        """Formats metrics for live CLI printing."""
        m = self.metrics
        return (f"[LIVE] Sessions: {m['active_sessions']} | Batch: {m['batch_size']} | "
                f"TPS: {m['tokens_per_sec']:.1f} | Sparse: {m['sparse_ratio']:.1%}")

    def log_recovery_event(self):
        self.metrics["semantic_recovery_activity"] += 1
        self.metrics["recovery_frequency"] = self.metrics["semantic_recovery_activity"] / (time.time() % 3600 + 1)
