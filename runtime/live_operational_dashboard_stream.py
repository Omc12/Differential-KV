import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional

class LiveOperationalDashboardStream:
    """
    ORX Phase 40.2: Live Operational Dashboard Stream.
    Provides real-time visibility into combined operational tracks.
    """
    def __init__(self, trace_path: Optional[str] = None):
        self.trace_path = trace_path
        self.logger = logging.getLogger("DashboardStream")
        self.current_state = {}

    def update_state(self, metrics: Dict[str, Any]):
        """Updates the dashboard state and logs it."""
        self.current_state = {
            "timestamp": time.time(),
            **metrics
        }
        if self.trace_path:
            self._persist_state()

    def _persist_state(self):
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.current_state) + "\n")
        except Exception as e:
            self.logger.error(f"Dashboard persistence failed: {e}")

    def format_live_line(self) -> str:
        """Formats the current state for live CLI output."""
        s = self.current_state
        return (f"[ORX-LIVE] Sess: {s.get('active_sessions', 0)} | "
                f"Cont: {s.get('continuity_score', 0.0):.2f} | "
                f"Q-Turb: {s.get('queue_turbulence', 0)} | "
                f"S-Overlap: {s.get('stream_overlap', 0)} | "
                f"Coherence: {s.get('coherence_score', 0.0):.2f}")

    async def run_stream_loop(self, interval: float = 1.0):
        """Continuously prints the live dashboard."""
        while True:
            if self.current_state:
                print(self.format_live_line())
            await asyncio.sleep(interval)
