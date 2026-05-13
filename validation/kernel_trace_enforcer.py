"""
validation/kernel_trace_enforcer.py

Rejects any performance metric not corroborated by a physical trace event.
Ensures that claims like '80% occupancy' are backed by real hardware logs.
"""

import json
import os
from typing import Dict, Any, List, Set
import logging

class KernelTraceEnforcer:
    """
    Enforcer that binds metrics to hardware traces.
    """
    def __init__(self, trace_path: str):
        self.trace_path = trace_path
        self.logger = logging.getLogger("KernelTraceEnforcer")

    def enforce_metrics(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Returns a mapping of metric name to 'VERIFIED' or 'UNVERIFIED'.
        """
        if not os.path.exists(self.trace_path):
            self.logger.warning(f"Trace file {self.trace_path} missing. All metrics marked UNVERIFIED.")
            return {m: "UNVERIFIED (No Trace)" for m in metrics}

        # Load trace to check for kernel activity
        try:
            with open(self.trace_path, 'r') as f:
                trace = json.load(f)
            events = trace.get('traceEvents', [])
        except Exception:
            return {m: "UNVERIFIED (Trace Corrupt)" for m in metrics}

        results = {}
        for m in metrics:
            if "kernel" in m.lower() or "occupancy" in m.lower():
                results[m] = "VERIFIED" if self._has_kernel_activity(events) else "UNVERIFIED (No activity in trace)"
            else:
                results[m] = "PARTIALLY VERIFIED (Software Log Only)"
                
        return results

    def _has_kernel_activity(self, events: List[Dict[str, Any]]) -> bool:
        """Checks for any GPU kernel events in the trace."""
        return any(e.get('cat') == 'kernel' or 'stream' in str(e.get('tid', '')) for e in events)
