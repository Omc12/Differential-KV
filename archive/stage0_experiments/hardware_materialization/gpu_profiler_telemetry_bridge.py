"""
hardware_materialization/gpu_profiler_telemetry_bridge.py

Collects real GPU execution telemetry using CUDA events and memory APIs.
"""

import torch
import time
from typing import Dict, Any, Optional

class GPUProfilerTelemetryBridge:
    """
    Bridges real GPU hardware metrics into the Differential KV telemetry system.
    """
    def __init__(self):
        self.events: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.enabled = torch.cuda.is_available()

    def start_timer(self, name: str):
        """Starts a CUDA event timer."""
        if not self.enabled:
            return
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.events[name] = (start, end)

    def stop_timer(self, name: str) -> float:
        """Stops a CUDA event timer and returns duration in ms."""
        if not self.enabled or name not in self.events:
            return 0.0
        
        # We cannot synchronize during CUDA graph capture
        if torch.cuda.is_current_stream_capturing():
            return 0.0

        start, end = self.events[name]
        end.record()
        torch.cuda.synchronize()
        duration = start.elapsed_time(end)
        return duration

    def get_vram_stats(self) -> Dict[str, float]:
        """Collects current VRAM allocation and peak usage."""
        if not self.enabled:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
            
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024)
        }

    def reset_peak_stats(self):
        """Resets peak VRAM tracking."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()

    def capture_telemetry(self) -> Dict[str, Any]:
        """Captures a snapshot of current GPU state."""
        vram = self.get_vram_stats()
        return {
            "timestamp": time.time(),
            "gpu_name": torch.cuda.get_device_name(0) if self.enabled else "None",
            **vram
        }
