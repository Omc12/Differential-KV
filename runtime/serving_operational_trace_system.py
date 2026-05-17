import json
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class ServingOperationalTraceSystem:
    """
    SGC Stage 3C.4: Serving Operational Trace System.
    Saves raw, physically-derived operational traces continuously to high-frequency JSONL files.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.trace_dir = self.workspace_root / "traces" / "stage3c" / "phase_42_4_sop"
        os.makedirs(self.trace_dir, exist_ok=True)

    def _write_jsonl(self, filename: str, record: Dict[str, Any]):
        """
        Appends a record to the specified JSONL trace file.
        """
        filepath = self.trace_dir / filename
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def record_continuous_batch(self, step: int, active_sessions: int, continuity: float):
        self._write_jsonl("continuous_batch_trace.jsonl", {
            "step": step,
            "active_sessions": active_sessions,
            "continuity": continuity
        })

    def record_decode_stream(self, step: int, stream_continuity: float, idle_gap_ms: float):
        self._write_jsonl("decode_stream_trace.jsonl", {
            "step": step,
            "stream_continuity": stream_continuity,
            "idle_gap_ms": idle_gap_ms
        })

    def record_kv_residency(self, step: int, reuse_ratio: float, migration_cost_ms: float):
        self._write_jsonl("kv_residency_trace.jsonl", {
            "step": step,
            "reuse_ratio": reuse_ratio,
            "migration_cost_ms": migration_cost_ms
        })

    def record_async_overlap(self, step: int, overlap_efficiency: float, saved_latency_ms: float):
        self._write_jsonl("async_overlap_trace.jsonl", {
            "step": step,
            "overlap_efficiency": overlap_efficiency,
            "saved_latency_ms": saved_latency_ms
        })

    def record_decode_fusion(self, step: int, launches_per_token: float, amortization: float):
        self._write_jsonl("decode_fusion_trace.jsonl", {
            "step": step,
            "launches_per_token": launches_per_token,
            "amortization": amortization
        })

    def record_gpu_starvation(self, step: int, starvation_pct: float, starvation_events: int):
        self._write_jsonl("gpu_starvation_trace.jsonl", {
            "step": step,
            "starvation_pct": starvation_pct,
            "starvation_events": starvation_events
        })

    def record_rolling_occupancy(self, step: int, occupancy: float):
        self._write_jsonl("rolling_occupancy_trace.jsonl", {
            "step": step,
            "occupancy": occupancy
        })

    def record_launch_amortization(self, step: int, launches_saved: int, amortization_score: float):
        self._write_jsonl("launch_amortization_trace.jsonl", {
            "step": step,
            "launches_saved": launches_saved,
            "amortization_score": amortization_score
        })

    def record_tail_latency(self, step: int, latency_ms: float, tail_latency_ms: float):
        self._write_jsonl("tail_latency_trace.jsonl", {
            "step": step,
            "latency_ms": latency_ms,
            "tail_latency_ms": tail_latency_ms
        })
