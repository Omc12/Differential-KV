"""
validation/trace_to_metric_mapper.py

Maps hardware trace durations to high-level performance metrics.
Provides the bridge between raw kernel events and 'Serving TPS'.
"""

from typing import List, Dict, Any
import logging

class TraceToMetricMapper:
    """
    Calculator for trace-derived performance metrics.
    """
    def __init__(self):
        self.logger = logging.getLogger("TraceToMetricMapper")

    def map_trace_to_tps(self, events: List[Dict[str, Any]], total_tokens: int) -> float:
        """
        Calculates TPS purely from kernel trace durations.
        """
        kernel_events = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
        if not kernel_events: return 0.0
        
        # total_duration = sum of non-overlapping kernel active time
        # For simplicity, we'll use wall time of the kernel stream
        start = min(e['ts'] for e in kernel_events)
        end = max(e['ts'] + e['dur'] for e in kernel_events)
        duration_sec = (end - start) / 1000000.0
        
        if duration_sec <= 0: return 0.0
        return total_tokens / duration_sec

    def map_trace_to_occupancy(self, events: List[Dict[str, Any]]) -> float:
        """
        Calculates occupancy: active_kernel_time / total_wall_time.
        """
        # Simplistic version
        return 0.85 # Placeholder
