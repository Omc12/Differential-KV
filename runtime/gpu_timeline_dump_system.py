import os
import time
import json
from pathlib import Path

class GpuTimelineDumpSystem:
    """
    RHD Phase 41.4.6 — GPU Timeline Dump System.
    Exports raw GPU execution timelines (CUDA stream timelines, kernel windows,
    synchronization gaps, compute bursts, idle gaps, memory transfers).
    No interpretations or success narratives. Persists raw timelines only.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "traces/stage3b/phase_41_4_6_rhd"
        self.trace_path = self.trace_dir / "raw_gpu_timeline_trace.jsonl"
        
        self.last_event_time = time.perf_counter()

    def record_event(
        self, 
        phase: str, 
        event_type: str, 
        duration_ms: float = 0.0, 
        stream_id: int = 0, 
        metadata: dict = None
    ):
        """
        Records a raw timeline event with duration, type, stream, and timestamp.
        """
        os.makedirs(self.trace_dir, exist_ok=True)
        
        current_perf_time = time.perf_counter()
        idle_gap_ms = (current_perf_time - self.last_event_time) * 1000.0
        self.last_event_time = current_perf_time + (duration_ms / 1000.0)

        record_data = {
            "timestamp": time.time(),
            "perf_timestamp": current_perf_time,
            "phase": phase,          # e.g., "compute_burst", "sync_gap", "memcpy", "idle_gap"
            "event_type": event_type,  # e.g., "kernel_window", "cuda_sync", "host_to_device"
            "duration_ms": duration_ms,
            "idle_gap_since_last_event_ms": idle_gap_ms,
            "cuda_stream_id": stream_id
        }
        
        if metadata:
            record_data.update(metadata)
            
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_data) + "\n")
