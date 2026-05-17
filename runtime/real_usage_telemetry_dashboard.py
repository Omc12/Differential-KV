import time
import json
import logging
from typing import Dict, Any, Optional

class RealUsageTelemetryDashboard:
    """
    RHU Phase 40.3: Real Usage Telemetry Dashboard.
    Provides live visibility into real human usage and UX stability.
    """
    def __init__(self, trace_path: Optional[str] = None):
        self.trace_path = trace_path
        self.logger = logging.getLogger("RealUsageDashboard")
        self.current_metrics = {}

    def update_metrics(self, metrics: Dict[str, Any]):
        self.current_metrics = {
            "timestamp": time.time(),
            **metrics
        }
        if self.trace_path:
            self._persist_metrics()

    def _persist_metrics(self):
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.current_metrics) + "\n")
        except Exception as e:
            self.logger.error(f"Dashboard persistence failed: {e}")

    def format_live_line(self) -> str:
        m = self.current_metrics
        return (f"[RHU-LIVE] Browsers: {m.get('active_browsers', 0)} | "
                f"UX-Stab: {m.get('ux_stability', 0.0):.2f} | "
                f"WS-Health: {m.get('websocket_health', 'OK')} | "
                f"Smoothness: {m.get('stream_smoothness', 0.0):.2f} | "
                f"Cont: {m.get('continuity_score', 0.0):.2f}")

    def get_summary(self) -> str:
        return json.dumps(self.current_metrics, indent=2)
