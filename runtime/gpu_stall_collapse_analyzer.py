import os
import time
import json
from pathlib import Path

class GpuStallCollapseAnalyzer:
    """
    CGO Phase 42.0 — GPU Stall Collapse Analyzer.
    Traces precise GPU idle intervals, synchronization gaps, CPU orchestration delays,
    memcpy wait times, and scheduler starvations.
    Persists only raw timing values.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "traces/stage3c/phase_42_0_cgo"
        self.trace_path = self.trace_dir / "gpu_stall_trace.jsonl"
        self.last_activity_time = time.perf_counter()

    def record_stall(self, stall_type: str, duration_ms: float, metadata: dict = None):
        """
        Records a physical stall window with duration and stall type.
        """
        os.makedirs(self.trace_dir, exist_ok=True)
        
        record_data = {
            "timestamp": time.time(),
            "stall_type": stall_type,      # sync_bubble, memcpy_wait, cpu_orchestration_delay, scheduler_starvation
            "duration_ms": duration_ms,
            "recorded_perf_ts": time.perf_counter()
        }
        
        if metadata:
            record_data.update(metadata)
            
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_data) + "\n")

    def mark_activity(self):
        """Calculates exact CPU-side orchestration gaps since last GPU activity."""
        curr = time.perf_counter()
        gap_ms = (curr - self.last_activity_time) * 1000.0
        self.last_activity_time = curr
        
        # Log CPU orchestration delays if they exceed 1ms
        if gap_ms > 1.0:
            self.record_stall("cpu_orchestration_delay", gap_ms)
