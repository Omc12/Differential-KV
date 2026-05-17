import os
import time
import json
import torch
from pathlib import Path
from typing import Dict, List, Any

class RawCudaEventTraceLayer:
    """
    RHD Phase 41.4.6 — Raw CUDA Event Trace Layer.
    Records actual CUDA event timings using torch.cuda.Event.
    Tracks kernel durations, synchronizations, memcpy activity, wait gaps,
    and sparse-vs-dense timings. Exports raw data only.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "telemetry/stage3b/phase_41_4_6_rhd"
        self.trace_path = self.trace_dir / "raw_cuda_event_trace.json"
        
        self.records: List[Dict[str, Any]] = []
        self.active_events: Dict[str, tuple] = {}

    def record_start(self, event_name: str, event_type: str = "kernel"):
        """Starts timing a GPU operation using raw CUDA events."""
        if not torch.cuda.is_available():
            return
            
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        self.active_events[event_name] = (start_evt, end_evt, event_type, time.time())

    def record_end(self, event_name: str):
        """Ends timing a GPU operation using raw CUDA events."""
        if not torch.cuda.is_available() or event_name not in self.active_events:
            return
            
        start_evt, end_evt, event_type, start_wall_time = self.active_events[event_name]
        end_evt.record()
        
        # Save a reference to compute later upon synchronization
        self.records.append({
            "event_name": event_name,
            "event_type": event_type,
            "start_time": start_wall_time,
            "end_time": time.time(),
            "start_evt": start_evt,
            "end_evt": end_evt,
            "completed": False
        })
        
        del self.active_events[event_name]

    def synchronize_and_flush(self):
        """Synchronizes the CUDA context and flushes event timings to raw JSON."""
        if not torch.cuda.is_available():
            return
            
        torch.cuda.synchronize()
        
        raw_events_data = []
        for r in self.records:
            if not r["completed"]:
                try:
                    duration_ms = r["start_evt"].elapsed_time(r["end_evt"])
                except Exception as e:
                    duration_ms = -1.0
                
                raw_events_data.append({
                    "event_name": r["event_name"],
                    "event_type": r["event_type"],
                    "start_timestamp": r["start_time"],
                    "end_timestamp": r["end_time"],
                    "duration_ms": duration_ms,
                    "cuda_stream": torch.cuda.current_stream().cuda_stream
                })
                r["completed"] = True
            
        # Append or rewrite to the JSON file
        os.makedirs(self.trace_dir, exist_ok=True)
        
        # Load existing data if file exists
        existing_data = []
        if self.trace_path.exists():
            try:
                with open(self.trace_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = []
                
        existing_data.extend(raw_events_data)
        
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)
            
        # Clear the in-memory records
        self.records = []
