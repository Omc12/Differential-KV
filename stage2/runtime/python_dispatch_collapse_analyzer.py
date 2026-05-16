"""
Python Dispatch Collapse Analyzer

Measures Python scheduling overhead, sync boundaries, and fragmentation.
"""
import time

class PythonDispatchCollapseAnalyzer:
    def __init__(self):
        self.overhead_ms = 0.0
        self.sync_boundaries = 0
        self.fragmentation_events = 0
        self.start_time = None
        
    def start_dispatch(self):
        self.start_time = time.perf_counter()
        
    def end_dispatch(self, sync_occurred=False, fragmented=False):
        if self.start_time:
            self.overhead_ms += (time.perf_counter() - self.start_time) * 1000.0
        if sync_occurred:
            self.sync_boundaries += 1
        if fragmented:
            self.fragmentation_events += 1
            
    def get_report(self):
        return {
            "python_overhead_ms": self.overhead_ms,
            "sync_boundaries": self.sync_boundaries,
            "fragmentation_events": self.fragmentation_events
        }
