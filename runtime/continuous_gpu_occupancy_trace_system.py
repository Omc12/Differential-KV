import os
import time
import json
from pathlib import Path
from runtime.persistent_decode_graph_executor import PersistentDecodeGraphExecutor
from runtime.continuous_batch_residency_engine import ContinuousBatchResidencyEngine
from runtime.gpu_stall_collapse_analyzer import GpuStallCollapseAnalyzer
from runtime.tensor_core_saturation_optimizer import TensorCoreSaturationOptimizer
from runtime.sparse_decode_fusion_compiler import SparseDecodeFusionCompiler
from runtime.async_kv_streaming_runtime import AsyncKvStreamingRuntime
from runtime.real_throughput_comparator import RealThroughputComparator

class ContinuousGpuOccupancyTraceSystem:
    """
    CGO Phase 42.0 — Continuous GPU Occupancy Trace System.
    Coordinates all occupancy, stall, tensor core, batch residency, sparse fusion,
    throughput comparison, and async KV traces. Saves raw physical logs only.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "traces/stage3c/phase_42_0_cgo"
        
        # Sub-systems
        self.graph_executor = PersistentDecodeGraphExecutor(workspace_root)
        self.residency_engine = ContinuousBatchResidencyEngine(workspace_root)
        self.stall_analyzer = GpuStallCollapseAnalyzer(workspace_root)
        self.tensor_optimizer = TensorCoreSaturationOptimizer(workspace_root)
        self.fusion_compiler = SparseDecodeFusionCompiler(workspace_root)
        self.async_kv = AsyncKvStreamingRuntime(workspace_root)
        self.comparator = RealThroughputComparator(workspace_root)
        
        self.decode_trace_path = self.trace_dir / "decode_continuity_trace.jsonl"
        self.batch_trace_path = self.trace_dir / "batch_residency_trace.jsonl"

    def record_decode_continuity(self, step: int, continuity_pct: float, idle_gap_pct: float, kernel_launches_per_sec: float):
        """Records decode phase continuity and launch frequencies."""
        os.makedirs(self.trace_dir, exist_ok=True)
        
        record_data = {
            "timestamp": time.time(),
            "step": step,
            "continuity_pct": continuity_pct,
            "idle_gap_pct": idle_gap_pct,
            "kernel_launches_per_sec": kernel_launches_per_sec
        }
        
        with open(self.decode_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_data) + "\n")

    def record_batch_residency(self, step: int, occupancy_pct: float, slots_active: int):
        """Records rolling slot occupancy metrics."""
        os.makedirs(self.trace_dir, exist_ok=True)
        
        record_data = {
            "timestamp": time.time(),
            "step": step,
            "occupancy_pct": occupancy_pct,
            "slots_active": slots_active
        }
        
        with open(self.batch_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_data) + "\n")
