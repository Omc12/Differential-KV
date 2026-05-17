import os
import time
import json
from pathlib import Path
import torch
from typing import Dict, Any

class RealThroughputComparator:
    """
    CGO Phase 42.0 — Real Throughput Comparator.
    Compares the wall-clock speed, latency, VRAM footprint, and occupancy of
    Differential KV versus standard dense execution baselines (e.g. HuggingFace).
    Saves raw benchmark evidence only.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "traces/stage3c/phase_42_0_cgo"
        self.trace_path = self.trace_dir / "throughput_comparison_trace.jsonl"

    def record_comparison(
        self,
        model_id: str,
        context_len: int,
        diffkv_tokens_per_sec: float,
        diffkv_latency_ms: float,
        diffkv_vram_bytes: int,
        dense_tokens_per_sec: float,
        dense_latency_ms: float,
        dense_vram_bytes: int
    ):
        """
        Records the raw comparison metrics of DiffKV vs Dense execution.
        """
        os.makedirs(self.trace_dir, exist_ok=True)
        
        record_data = {
            "timestamp": time.time(),
            "model_id": model_id,
            "context_length": context_len,
            "diffkv_tokens_per_sec": diffkv_tokens_per_sec,
            "diffkv_latency_ms": diffkv_latency_ms,
            "diffkv_vram_bytes": diffkv_vram_bytes,
            "dense_tokens_per_sec": dense_tokens_per_sec,
            "dense_latency_ms": dense_latency_ms,
            "dense_vram_bytes": dense_vram_bytes,
            "throughput_improvement_pct": ((diffkv_tokens_per_sec - dense_tokens_per_sec) / max(0.001, dense_tokens_per_sec)) * 100.0,
            "vram_saving_pct": ((dense_vram_bytes - diffkv_vram_bytes) / max(1, dense_vram_bytes)) * 100.0
        }
        
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_data) + "\n")
