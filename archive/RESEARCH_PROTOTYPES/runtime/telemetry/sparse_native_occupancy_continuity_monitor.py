"""
Sparse-Native Occupancy Continuity Monitor

Measures launch persistence and occupancy collapse windows.
"""

class SparseNativeOccupancyContinuityMonitor:
    def __init__(self):
        self.continuity_score = 100.0
        self.idle_gaps = 0
        self.collapse_windows = 0
        
    def log_launch(self, persistent=True, idle_gap_ms=0.0):
        if not persistent:
            self.continuity_score -= 0.1
            self.collapse_windows += 1
        if idle_gap_ms > 0.5:
            self.idle_gaps += 1
            
    def get_report(self):
        return {
            "occupancy_continuity_score": self.continuity_score,
            "kernel_idle_gaps": self.idle_gaps,
            "occupancy_collapse_windows": self.collapse_windows
        }
