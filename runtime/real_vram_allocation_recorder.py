import os
import time
import json
import torch
from pathlib import Path

class RealVramAllocationRecorder:
    """
    RHD Phase 41.4.6 — Real VRAM Allocation Recorder.
    Records physical VRAM usage directly from PyTorch CUDA APIs.
    Tracks memory allocated, memory reserved, peak allocations, and allocation deltas.
    Persists only raw evidence to JSONL.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "traces/stage3b/phase_41_4_6_rhd"
        self.trace_path = self.trace_dir / "raw_vram_trace.jsonl"
        self.last_allocated = 0

    def record(self, step: int, seq_len: int, extra_metadata: dict = None):
        if not torch.cuda.is_available():
            return
            
        os.makedirs(self.trace_dir, exist_ok=True)
        
        # Collect physical VRAM measurements
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        
        delta = allocated - self.last_allocated
        self.last_allocated = allocated
        
        # Calculate standard allocator fragmentation
        fragmentation = max(0, reserved - allocated)
        
        record_data = {
            "timestamp": time.time(),
            "step": step,
            "seq_len": seq_len,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "max_allocated_bytes": max_allocated,
            "max_reserved_bytes": max_reserved,
            "allocation_delta_bytes": delta,
            "allocator_fragmentation_bytes": fragmentation,
        }
        
        if extra_metadata:
            record_data.update(extra_metadata)
            
        # Write to JSONL
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_data) + "\n")
            
        # Optional: reset peak memory stats to track peak between steps
        # torch.cuda.reset_peak_memory_stats()
