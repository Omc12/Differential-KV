import time
from typing import Dict
from .adaptive_anchor_modes import AnchorSpacingMode

class AnchorSpacingProfiler:
    """
    Profiles the performance impact of different anchor spacing modes.
    Feeds back into the budgeter and controller.
    """
    def __init__(self):
        self.mode_latencies: Dict[AnchorSpacingMode, list] = {m: [] for m in AnchorSpacingMode}

    def record_latency(self, mode: AnchorSpacingMode, latency_ms: float):
        self.mode_latencies[mode].append(latency_ms)
        if len(self.mode_latencies[mode]) > 100:
            self.mode_latencies[mode].pop(0)

    def get_mode_overhead(self, mode: AnchorSpacingMode) -> float:
        """Returns average latency for a mode in ms."""
        lats = self.mode_latencies.get(mode, [])
        if not lats:
            return 0.0
        return sum(lats) / len(lats)

    def get_efficiency_report(self) -> dict:
        return {mode.name: self.get_mode_overhead(mode) for mode in AnchorSpacingMode}
