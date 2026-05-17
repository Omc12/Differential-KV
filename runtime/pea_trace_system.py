import os
import json
from typing import Dict, Any

class PeaTraceSystem:
    """
    STAGE 4A.3 — PEA: PEA Trace System.
    Synchronously persists raw, hardware-grounded serving events to disk 
    across 10 distinct trace logs.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.file_names = {
            "tensor_residency": "tensor_residency_trace.jsonl",
            "allocator_fragmentation": "allocator_fragmentation_trace.jsonl",
            "allocation_reuse": "allocation_reuse_trace.jsonl",
            "replay_memory": "replay_memory_trace.jsonl",
            "pointer_stability": "pointer_stability_trace.jsonl",
            "stream_affinity": "stream_affinity_trace.jsonl",
            "warm_start": "warm_start_trace.jsonl",
            "allocation_pressure": "allocation_pressure_trace.jsonl",
            "allocator_tail": "allocator_tail_trace.jsonl",
            "replay_invalidation_memory": "replay_invalidation_memory_trace.jsonl"
        }
        
        self.files = {}
        for key, fname in self.file_names.items():
            path = os.path.join(output_dir, fname)
            self.files[key] = open(path, "w", encoding="utf-8")
            
    def log_trace(self, trace_key: str, data: Dict[str, Any]):
        """Synchronously appends a trace record, guaranteeing clean file buffer flushing."""
        if trace_key in self.files:
            record = {"timestamp_epoch": os.times()[4]}
            record.update(data)
            self.files[trace_key].write(json.dumps(record) + "\n")
            self.files[trace_key].flush()
            
    def close(self):
        """Cleanly releases all active log writers."""
        for f in self.files.values():
            try:
                f.close()
            except:
                pass
        self.files.clear()
