"""
benchmarks/baselines.py

Measures memory and timing for competing KV-cache strategies:

  FP16  — store every token's KV in full FP16 (baseline)
  FP8   — simulated FP8 (quantize to 8-bit float via scale trick)
  INT8  — naive per-tensor INT8 of full KV (no delta)

For each baseline we record:
  - total bytes stored
  - simulated read latency (bytes / assumed BW)
  - reconstruction error vs original FP16

These are the numbers Differential KV must beat (or at least match).
"""

import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from benchmarks.kv_generator import KVGenerator


@dataclass
class BaselineResult:
    label: str
    seq_len: int
    num_heads: int
    head_dim: int
    total_bytes: int
    compression_ratio: float          # vs FP16
    mean_relative_error: float        # vs original FP16
    max_relative_error: float
    encode_ms: float                  # time to quantize
    decode_ms: float                  # time to dequantize back to FP16

    def __str__(self) -> str:
        return (
            f"[{self.label:<12}] seq={self.seq_len:>6} | "
            f"{self.total_bytes/1024:>8.1f} KB | "
            f"ratio={self.compression_ratio:.2f}x | "
            f"err={self.mean_relative_error:.5f} | "
            f"enc={self.encode_ms:.2f}ms dec={self.decode_ms:.2f}ms"
        )


def _simulate_fp8(kv: torch.Tensor) -> torch.Tensor:
    """
    Simulate FP8 by quantizing to a scaled INT8 and back.
    FP8 (E4M3) has ~3 bits of mantissa; we approximate with
    aggressive INT8 quantization (scale=max/127).
    Not a true FP8 — just a bandwidth simulation.
    """
    x = kv.float()
    scale = x.abs().max() / 127.0
    if scale < 1e-9:
        return kv.clone()
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return (q.float() * scale).to(kv.dtype)


def _simulate_int8(kv: torch.Tensor):
    """Naive INT8 quantization: quantize and return (quantized_int8, scale)."""
    x = kv.float()
    scale = x.abs().max() / 127.0
    if scale < 1e-9:
        return torch.zeros_like(kv, dtype=torch.int8), 1.0
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def _relative_error(original: torch.Tensor, reconstructed: torch.Tensor):
    diff = (original.float() - reconstructed.float())
    l2_per_tok = torch.linalg.vector_norm(diff, dim=(-1, -2, -3))
    orig_norm = torch.linalg.vector_norm(original.float(), dim=(-1, -2, -3))
    rel = l2_per_tok / (orig_norm + 1e-9)
    return rel.mean().item(), rel.max().item()


class BaselineRunner:
    """
    Runs all baseline KV-cache strategies on a given KV sequence.

    Parameters
    ----------
    num_heads : int
    head_dim  : int
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128):
        self.num_heads = num_heads
        self.head_dim = head_dim

    def run_fp16(self, kv: torch.Tensor) -> BaselineResult:
        """FP16 baseline — no compression, just measure bytes."""
        seq_len = kv.shape[0]
        total_bytes = kv.numel() * 2

        t0 = time.perf_counter()
        stored = kv.clone()  # simulate store
        enc_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        read = stored.clone()  # simulate read
        dec_ms = (time.perf_counter() - t0) * 1000

        return BaselineResult(
            label="FP16",
            seq_len=seq_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            total_bytes=total_bytes,
            compression_ratio=1.0,
            mean_relative_error=0.0,
            max_relative_error=0.0,
            encode_ms=enc_ms,
            decode_ms=dec_ms,
        )

    def run_fp8_simulated(self, kv: torch.Tensor) -> BaselineResult:
        """Simulated FP8 baseline."""
        seq_len = kv.shape[0]
        # FP8 = 1 byte per element
        total_bytes = kv.numel() * 1

        t0 = time.perf_counter()
        quantized = _simulate_fp8(kv)
        enc_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reconstructed = quantized.to(torch.float16)
        dec_ms = (time.perf_counter() - t0) * 1000

        fp16_bytes = kv.numel() * 2
        mean_err, max_err = _relative_error(kv, reconstructed)

        return BaselineResult(
            label="FP8-sim",
            seq_len=seq_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            total_bytes=total_bytes,
            compression_ratio=fp16_bytes / total_bytes,
            mean_relative_error=mean_err,
            max_relative_error=max_err,
            encode_ms=enc_ms,
            decode_ms=dec_ms,
        )

    def run_int8_naive(self, kv: torch.Tensor) -> BaselineResult:
        """Naive INT8: quantize entire KV tensor, no delta structure."""
        seq_len = kv.shape[0]
        # INT8 data + 1 float32 scale for the whole tensor
        total_bytes = kv.numel() * 1 + 4

        t0 = time.perf_counter()
        q, scale = _simulate_int8(kv)
        enc_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reconstructed = (q.float() * scale).to(torch.float16)
        dec_ms = (time.perf_counter() - t0) * 1000

        fp16_bytes = kv.numel() * 2
        mean_err, max_err = _relative_error(kv, reconstructed)

        return BaselineResult(
            label="INT8-naive",
            seq_len=seq_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            total_bytes=total_bytes,
            compression_ratio=fp16_bytes / total_bytes,
            mean_relative_error=mean_err,
            max_relative_error=max_err,
            encode_ms=enc_ms,
            decode_ms=dec_ms,
        )

    def run_all(self, kv: torch.Tensor) -> Dict[str, BaselineResult]:
        return {
            "fp16": self.run_fp16(kv),
            "fp8": self.run_fp8_simulated(kv),
            "int8": self.run_int8_naive(kv),
        }
