"""
hardware_materialization/profiler_trace_collector.py

Collects detailed runtime traces and aggregates CUDA timing data for Differential KV.
"""

import torch
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("TraceCollector")

class ProfilerTraceCollector:
    """
    Integrates torch profiler and CUDA events to capture execution timelines.
    """
    def __init__(self, export_path: str = "profiler_traces/"):
        self.export_path = export_path
        self.traces: List[Dict[str, Any]] = []
        self.enabled = torch.cuda.is_available()
        self.prof: Optional[torch.profiler.profile] = None

    def start_collection(self):
        """Starts torch profiler collection."""
        if not self.enabled:
            return
            
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.export_path)
        )
        self.prof.start()
        logger.info("Profiler trace collection started.")

    def stop_collection(self):
        """Stops torch profiler and exports data."""
        if self.prof:
            self.prof.stop()
            logger.info(f"Profiler trace collection stopped. Data exported to {self.export_path}")
            self.prof = None

    def record_event_timing(self, name: str, duration_ms: float):
        """Records a manually measured CUDA event duration."""
        self.traces.append({
            "event": name,
            "duration_ms": duration_ms,
            "timestamp": time.time()
        })

    def get_summary(self) -> Dict[str, float]:
        """Returns aggregated timing statistics."""
        if not self.traces:
            return {}
            
        summary = {}
        for t in self.traces:
            name = t["event"]
            dur = t["duration_ms"]
            if name not in summary:
                summary[name] = []
            summary[name].append(dur)
            
        return {name: sum(durs)/len(durs) for name, durs in summary.items()}
