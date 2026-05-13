"""
validation/occupancy_trace_validator.py

Validates reported GPU occupancy against hardware trace events.
Ensures that 'high occupancy' claims are backed by dense kernel activity.
"""

import json
from typing import List, Dict, Any
import logging

class OccupancyTraceValidator:
    """
    Parses profiler traces to extract hardware-level occupancy.
    """
    def __init__(self):
        self.logger = logging.getLogger("OccupancyTraceValidator")

    def validate_occupancy(self, trace_path: str, reported_occupancy: float) -> bool:
        """
        Calculates occupancy from trace: (sum of kernel durations) / (total wall time).
        """
        try:
            with open(trace_path, 'r') as f:
                trace = json.load(f)
            
            events = trace.get('traceEvents', [])
            gpu_events = [e for e in events if e.get('ph') == 'X' and 'dur' in e and e.get('tid') == 'stream']
            
            if not gpu_events: return False
            
            total_dur = sum(e['dur'] for e in gpu_events)
            start = min(e['ts'] for e in gpu_events)
            end = max(e['ts'] + e['dur'] for e in gpu_events)
            wall_time = end - start
            
            calculated_occupancy = total_dur / wall_time
            
            if abs(calculated_occupancy - reported_occupancy) > 0.1:
                self.logger.warning(f"OCCUPANCY MISMATCH: Reported {reported_occupancy:.2f}, Trace {calculated_occupancy:.2f}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Occupancy validation failed: {e}")
            return False
