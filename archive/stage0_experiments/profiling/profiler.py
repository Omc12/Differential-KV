"""
profiling/profiler.py

Two profiling primitives:

1. ReconstructionProfiler
   Records per-call reconstruction statistics across multiple calls,
   then aggregates them into a summary. Used by benchmark runners.

2. MemoryBandwidthEstimator
   Estimates effective memory bandwidth for:
     - FP16 baseline (read full KV)
     - Differential KV (read anchor + delta)
     - FP8 simulated baseline
     - INT8 naive baseline

All measurements are model-based (not hardware-measured) in Phase 1.
Hardware perf counters can be added later via torch.cuda.Event.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


# ---------------------------------------------------------------------------
# ReconstructionProfiler
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    token_start: int
    token_end: int
    elapsed_ms: float
    anchor_loads: int
    delta_loads: int
    bytes_read: int
    reconstruction_error: float = 0.0   # mean relative L2


@dataclass
class ProfilerSummary:
    num_calls: int
    total_tokens_reconstructed: int
    total_elapsed_ms: float
    total_bytes_read: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_error: float
    max_error: float
    mean_anchor_loads: float
    mean_delta_loads: float
    tokens_per_second: float

    def __str__(self) -> str:
        lines = [
            f"  Calls              : {self.num_calls}",
            f"  Total tokens       : {self.total_tokens_reconstructed}",
            f"  Total elapsed      : {self.total_elapsed_ms:.2f} ms",
            f"  Mean latency       : {self.mean_latency_ms:.3f} ms/call",
            f"  P95 latency        : {self.p95_latency_ms:.3f} ms/call",
            f"  Tokens/sec         : {self.tokens_per_second:,.0f}",
            f"  Total bytes read   : {self.total_bytes_read / 1024:.1f} KB",
            f"  Mean recon error   : {self.mean_error:.6f}",
            f"  Max recon error    : {self.max_error:.6f}",
            f"  Mean anchor loads  : {self.mean_anchor_loads:.2f}",
            f"  Mean delta loads   : {self.mean_delta_loads:.2f}",
        ]
        return "\n".join(lines)


class ReconstructionProfiler:
    """
    Records reconstruction call statistics and produces aggregate summaries.

    Usage
    -----
    profiler = ReconstructionProfiler()
    profiler.record(result, error)   # after each reconstruct_range() call
    summary = profiler.summarize()
    """

    def __init__(self):
        self._records: List[CallRecord] = []

    def reset(self):
        self._records.clear()

    def record(self, token_start: int, token_end: int, elapsed_ms: float,
               anchor_loads: int, delta_loads: int, bytes_read: int,
               reconstruction_error: float = 0.0):
        self._records.append(CallRecord(
            token_start=token_start,
            token_end=token_end,
            elapsed_ms=elapsed_ms,
            anchor_loads=anchor_loads,
            delta_loads=delta_loads,
            bytes_read=bytes_read,
            reconstruction_error=reconstruction_error,
        ))

    def record_result(self, result, error: float = 0.0):
        """Convenience wrapper for ReconstructionResult objects."""
        self.record(
            token_start=result.token_start,
            token_end=result.token_end,
            elapsed_ms=result.elapsed_ms,
            anchor_loads=result.anchor_loads,
            delta_loads=result.delta_loads,
            bytes_read=result.bytes_read,
            reconstruction_error=error,
        )

    def summarize(self) -> ProfilerSummary:
        if not self._records:
            raise RuntimeError("No records to summarize.")

        import statistics
        latencies = [r.elapsed_ms for r in self._records]
        errors = [r.reconstruction_error for r in self._records]
        total_tokens = sum(r.token_end - r.token_start + 1 for r in self._records)
        total_ms = sum(latencies)
        total_bytes = sum(r.bytes_read for r in self._records)

        sorted_lat = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_lat))
        p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

        tokens_per_sec = (total_tokens / total_ms * 1000.0) if total_ms > 0 else 0.0

        return ProfilerSummary(
            num_calls=len(self._records),
            total_tokens_reconstructed=total_tokens,
            total_elapsed_ms=total_ms,
            total_bytes_read=total_bytes,
            mean_latency_ms=statistics.mean(latencies),
            p95_latency_ms=p95,
            mean_error=statistics.mean(errors) if errors else 0.0,
            max_error=max(errors) if errors else 0.0,
            mean_anchor_loads=statistics.mean(r.anchor_loads for r in self._records),
            mean_delta_loads=statistics.mean(r.delta_loads for r in self._records),
            tokens_per_second=tokens_per_sec,
        )


# ---------------------------------------------------------------------------
# MemoryBandwidthEstimator
# ---------------------------------------------------------------------------

@dataclass
class BandwidthEstimate:
    """Estimated memory movement for one configuration."""
    label: str
    seq_len: int
    num_layers: int
    num_heads: int
    head_dim: int
    bytes_per_element: float         # e.g. 2.0 for FP16, 1.0 for INT8
    total_bytes: int
    compression_ratio: float = 1.0

    @property
    def gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    def __str__(self) -> str:
        return (
            f"  [{self.label}] seq={self.seq_len} | "
            f"{self.total_bytes / 1024:.0f} KB | "
            f"ratio={self.compression_ratio:.2f}x"
        )


class MemoryBandwidthEstimator:
    """
    Model-based memory bandwidth estimator for KV-cache configurations.

    Parameters
    ----------
    num_layers : int
    num_heads  : int
    head_dim   : int
    """

    def __init__(self, num_layers: int = 32, num_heads: int = 32, head_dim: int = 128):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _kv_elements(self, seq_len: int) -> int:
        # 2 for K and V
        return seq_len * 2 * self.num_heads * self.head_dim * self.num_layers

    def estimate_fp16(self, seq_len: int) -> BandwidthEstimate:
        total = self._kv_elements(seq_len) * 2  # 2 bytes per FP16
        return BandwidthEstimate("FP16", seq_len, self.num_layers,
                                 self.num_heads, self.head_dim, 2.0, total, 1.0)

    def estimate_fp8(self, seq_len: int) -> BandwidthEstimate:
        total = self._kv_elements(seq_len) * 1  # 1 byte per FP8
        fp16 = self._kv_elements(seq_len) * 2
        return BandwidthEstimate("FP8", seq_len, self.num_layers,
                                 self.num_heads, self.head_dim, 1.0, total,
                                 fp16 / total)

    def estimate_int8_naive(self, seq_len: int) -> BandwidthEstimate:
        """INT8 without delta — just every token quantized to INT8."""
        total = self._kv_elements(seq_len) * 1
        fp16 = self._kv_elements(seq_len) * 2
        return BandwidthEstimate("INT8-naive", seq_len, self.num_layers,
                                 self.num_heads, self.head_dim, 1.0, total,
                                 fp16 / total)

    def estimate_differential_kv(
        self,
        seq_len: int,
        anchor_density: float,         # fraction of tokens that are anchors
        delta_bytes_per_token: int,    # avg bytes per delta token (INT8 + scale ≈ head_dim*2*heads*1 + 4)
    ) -> BandwidthEstimate:
        """
        Differential KV estimate:
          anchors: stored at FP16
          deltas: stored as INT8 + scale
        """
        num_anchor_tokens = int(seq_len * anchor_density)
        num_delta_tokens = seq_len - num_anchor_tokens

        elems_per_token = 2 * self.num_heads * self.head_dim  # K+V
        anchor_bytes = num_anchor_tokens * elems_per_token * 2 * self.num_layers  # FP16
        delta_bytes = num_delta_tokens * delta_bytes_per_token * self.num_layers

        total = anchor_bytes + delta_bytes
        fp16 = self._kv_elements(seq_len) * 2
        ratio = fp16 / total if total > 0 else 1.0

        return BandwidthEstimate("DiffKV", seq_len, self.num_layers,
                                 self.num_heads, self.head_dim,
                                 bytes_per_element=1.0 + anchor_density,
                                 total_bytes=total,
                                 compression_ratio=ratio)

    def compare_all(
        self, seq_len: int, anchor_density: float = 0.02
    ) -> Dict[str, BandwidthEstimate]:
        """Return all estimates for a given sequence length."""
        elems_per_token = 2 * self.num_heads * self.head_dim
        # INT8 delta + 4-byte scale per token
        delta_bytes_per_token = elems_per_token * 1 + 4

        return {
            "fp16": self.estimate_fp16(seq_len),
            "fp8": self.estimate_fp8(seq_len),
            "int8_naive": self.estimate_int8_naive(seq_len),
            "diff_kv": self.estimate_differential_kv(
                seq_len, anchor_density, delta_bytes_per_token
            ),
        }
