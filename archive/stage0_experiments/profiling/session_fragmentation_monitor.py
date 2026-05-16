import os
from typing import Dict, Any

class SessionFragmentationMonitor:
    """
    Tracks memory holes and allocation efficiency during long sessions.
    """
    def __init__(self):
        self.fragmentation_history = []

    def record_fragmentation(self, allocated: int, total_reserved: int):
        if total_reserved == 0:
            frag = 0.0
        else:
            frag = 1.0 - (allocated / total_reserved)
        self.fragmentation_history.append(frag)
        return frag

    def get_summary(self) -> Dict[str, Any]:
        if not self.fragmentation_history:
            return {}
        return {
            "initial_frag": self.fragmentation_history[0],
            "current_frag": self.fragmentation_history[-1],
            "growth": self.fragmentation_history[-1] - self.fragmentation_history[0]
        }
