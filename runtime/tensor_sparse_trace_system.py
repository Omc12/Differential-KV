import os
import json
import time
from pathlib import Path

class TensorSparseTraceSystem:
    """
    SGC Stage 3C.3: Tensor Sparse Trace System.
    Manages and persists raw, physically-derived execution records for TSO components.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.trace_dir = self.workspace_root / "traces" / "stage3c" / "phase_42_3_tso"
        os.makedirs(self.trace_dir, exist_ok=True)

        # Trace paths
        self.paths = {
            "triton_kernel": self.trace_dir / "triton_kernel_trace.jsonl",
            "flash_sparse": self.trace_dir / "flash_sparse_trace.jsonl",
            "tensor_core": self.trace_dir / "tensor_core_execution_trace.jsonl",
            "shared_memory": self.trace_dir / "shared_memory_trace.jsonl",
            "register_pressure": self.trace_dir / "register_pressure_trace.jsonl",
            "persistent_attention": self.trace_dir / "persistent_attention_trace.jsonl",
            "launch_fragmentation": self.trace_dir / "launch_fragmentation_trace.jsonl",
            "bandwidth": self.trace_dir / "bandwidth_trace.jsonl"
        }
        self._initialize_traces()

    def _initialize_traces(self):
        # Truncate and prep files
        for name, p in self.paths.items():
            if p.exists():
                try:
                    os.remove(p)
                except Exception:
                    pass
            # Open and touch file
            with open(p, "w", encoding="utf-8") as f:
                pass

    def record_triton(self, step: int, latency_ms: float, launches: int):
        rec = {"timestamp": time.time(), "step": step, "latency_ms": latency_ms, "launches": launches}
        self._write_line("triton_kernel", rec)

    def record_flash(self, step: int, sram_read_bytes: int, active_tiles: int):
        rec = {"timestamp": time.time(), "step": step, "sram_read_bytes": sram_read_bytes, "active_tiles": active_tiles}
        self._write_line("flash_sparse", rec)

    def record_tensor_core(self, step: int, hmma_active_cycles: int, util_pct: float):
        rec = {"timestamp": time.time(), "step": step, "hmma_active_cycles": hmma_active_cycles, "util_pct": util_pct}
        self._write_line("tensor_core", rec)

    def record_shared_memory(self, step: int, cache_hits: int, efficiency_pct: float):
        rec = {"timestamp": time.time(), "step": step, "cache_hits": cache_hits, "efficiency_pct": efficiency_pct}
        self._write_line("shared_memory", rec)

    def record_register_pressure(self, step: int, regs_per_thread: int, pressure_score: float):
        rec = {"timestamp": time.time(), "step": step, "registers_per_thread": regs_per_thread, "pressure_score": pressure_score}
        self._write_line("register_pressure", rec)

    def record_persistent_attention(self, step: int, residency_pct: float, hits: int):
        rec = {"timestamp": time.time(), "step": step, "residency_pct": residency_pct, "hits": hits}
        self._write_line("persistent_attention", rec)

    def record_launch_fragmentation(self, step: int, driver_launches: int, ratio: float):
        rec = {"timestamp": time.time(), "step": step, "driver_launches": driver_launches, "ratio": ratio}
        self._write_line("launch_fragmentation", rec)

    def record_bandwidth(self, step: int, read_gb_s: float, write_gb_s: float, stall_pct: float):
        rec = {"timestamp": time.time(), "step": step, "read_gb_s": read_gb_s, "write_gb_s": write_gb_s, "stall_pct": stall_pct}
        self._write_line("bandwidth", rec)

    def _write_line(self, key: str, record: dict):
        p = self.paths.get(key)
        if p:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
