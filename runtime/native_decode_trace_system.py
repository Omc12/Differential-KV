import os
import sys
import time
import json
from pathlib import Path

class NativeDecodeTraceSystem:
    """
    NDX Phase 42.1.5 — Native Decode Trace System.
    Manages and persists raw physically-derived JSONL traces.
    NO synthesized conclusions or summaries allowed.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "traces/stage3c/phase_42_1_5_ndx"
        os.makedirs(self.trace_dir, exist_ok=True)
        
        self.exec_trace_path = self.trace_dir / "native_decode_execution_trace.jsonl"
        self.residency_trace_path = self.trace_dir / "native_batch_residency_trace.jsonl"
        self.stream_trace_path = self.trace_dir / "native_stream_trace.jsonl"
        self.graph_trace_path = self.trace_dir / "cuda_graph_replay_trace.jsonl"
        self.queue_trace_path = self.trace_dir / "native_queue_trace.jsonl"
        self.violation_trace_path = self.trace_dir / "fallback_violation_trace.jsonl"
        self.lineage_trace_path = self.trace_dir / "execution_lineage_trace.jsonl"
        
        # Touch all trace files to guarantee their physical existence even if empty
        for path in [
            self.exec_trace_path, self.residency_trace_path, self.stream_trace_path,
            self.graph_trace_path, self.queue_trace_path, self.violation_trace_path,
            self.lineage_trace_path
        ]:
            if not path.exists():
                with open(path, "w", encoding="utf-8") as f:
                    pass

    def _write_trace(self, file_path: Path, record: dict):
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def record_execution(self, step: int, latency_ms: float, launches: int, active_slots: int):
        self._write_trace(self.exec_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "latency_ms": latency_ms,
            "launches": launches,
            "active_slots": active_slots
        })

    def record_residency(self, step: int, occupancy_rate: float, slots_occupied: int):
        self._write_trace(self.residency_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "occupancy_rate": occupancy_rate,
            "slots_occupied": slots_occupied
        })

    def record_stream(self, step: int, overlap_ms: float, is_active: bool):
        self._write_trace(self.stream_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "overlap_ms": overlap_ms,
            "is_active": is_active
        })

    def record_graph_replay(self, step: int, replay_count: int, active: bool):
        self._write_trace(self.graph_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "replay_count": replay_count,
            "active": active
        })

    def record_queue(self, step: int, depth: int, active_head_id: int):
        self._write_trace(self.queue_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "queue_depth": depth,
            "active_head_id": active_head_id
        })

    def record_violation(self, message: str):
        self._write_trace(self.violation_trace_path, {
            "timestamp": time.time(),
            "message": message
        })

    def record_lineage(self, step: int, target: str, lineage: str):
        self._write_trace(self.lineage_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "target": target,
            "lineage": lineage
        })
