import os
import time
import json
from pathlib import Path
from runtime.persistent_token_decode_runtime import PersistentTokenDecodeRuntime
from runtime.decode_launch_collapse_engine import DecodeLaunchCollapseEngine
from runtime.continuous_decode_residency_scheduler import ContinuousDecodeResidencyScheduler
from runtime.synchronization_bubble_eliminator import SynchronizationBubbleEliminator
from runtime.async_decode_overlap_runtime import AsyncDecodeOverlapRuntime
from runtime.native_decode_loop_executor import NativeDecodeLoopExecutor

class DecodePipelineTraceSystem:
    """
    DPC Phase 42.1 — Decode Pipeline Trace System.
    Coordinates all pipeline collapse, residency, launch, synchronization, and native loop 
    telemetry trace events. Saves raw physical logs only.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "traces/stage3c/phase_42_1_dpc"
        
        # Sub-systems
        self.decode_runtime = PersistentTokenDecodeRuntime(workspace_root)
        self.launch_engine = DecodeLaunchCollapseEngine(workspace_root)
        self.scheduler = ContinuousDecodeResidencyScheduler(workspace_root)
        self.bubble_eliminator = SynchronizationBubbleEliminator(workspace_root)
        self.async_overlap = AsyncDecodeOverlapRuntime(workspace_root)
        self.native_executor = NativeDecodeLoopExecutor(workspace_root)
        
        # Traces
        self.launch_trace_path = self.trace_dir / "decode_launch_trace.jsonl"
        self.residency_trace_path = self.trace_dir / "decode_residency_trace.jsonl"
        self.bubble_trace_path = self.trace_dir / "synchronization_bubble_trace.jsonl"
        self.async_trace_path = self.trace_dir / "async_decode_trace.jsonl"
        self.native_trace_path = self.trace_dir / "native_decode_loop_trace.jsonl"
        self.continuity_trace_path = self.trace_dir / "pipeline_continuity_trace.jsonl"
        self.idle_gap_trace_path = self.trace_dir / "gpu_idle_gap_trace.jsonl"

    def record_launch(self, step: int, raw: int, collapsed: int, reduction_pct: float):
        os.makedirs(self.trace_dir, exist_ok=True)
        self._write_trace(self.launch_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "raw_launches": raw,
            "collapsed_launches": collapsed,
            "reduction_pct": reduction_pct
        })

    def record_residency(self, step: int, active_slots: int, residency_ratio: float):
        os.makedirs(self.trace_dir, exist_ok=True)
        self._write_trace(self.residency_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "active_slots": active_slots,
            "residency_ratio": residency_ratio
        })

    def record_bubble(self, step: int, duration_ms: float, average_bubble_ms: float):
        os.makedirs(self.trace_dir, exist_ok=True)
        self._write_trace(self.bubble_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "sync_bubble_ms": duration_ms,
            "average_sync_bubble_ms": average_bubble_ms
        })

    def record_async(self, step: int, overlap_active: bool, overlap_pct: float):
        os.makedirs(self.trace_dir, exist_ok=True)
        self._write_trace(self.async_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "overlap_active": overlap_active,
            "overlap_pct": overlap_pct
        })

    def record_native(self, step: int, latency_ms: float, is_compiled: bool):
        os.makedirs(self.trace_dir, exist_ok=True)
        self._write_trace(self.native_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "native_latency_ms": latency_ms,
            "is_compiled": is_compiled
        })

    def record_continuity(self, step: int, continuity_pct: float):
        os.makedirs(self.trace_dir, exist_ok=True)
        self._write_trace(self.continuity_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "continuity_pct": continuity_pct
        })

    def record_idle_gap(self, step: int, idle_gap_pct: float):
        os.makedirs(self.trace_dir, exist_ok=True)
        self._write_trace(self.idle_gap_trace_path, {
            "timestamp": time.time(),
            "step": step,
            "idle_gap_pct": idle_gap_pct
        })

    def _write_trace(self, path: Path, data: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
