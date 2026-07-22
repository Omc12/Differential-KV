"""
native_core/mac_utils.py

Apple Silicon (MPS / Metal) compatibility layer for Differential KV.

Provides unified device detection and helpers so every other module can call
`get_best_device()` instead of hard-coding "cuda".

Priority order:
  1. CUDA  — if a CUDA GPU is present (cloud / external GPU)
  2. MPS   — if running on Apple Silicon (M-series, torch >= 2.1)
  3. CPU   — universal fallback

MLX note:
  Apple MLX (https://ml-explore.github.io/mlx) is a separate array library.
  DKV keeps its core in PyTorch; MLX is bridged here for SVD acceleration
  when running on Apple Silicon without CUDA.  The bridge is optional — if
  `mlx` is not installed, everything falls back to PyTorch on MPS.
"""

import sys
import torch
from typing import Optional

# ── CUDA / MPS detection ──────────────────────────────────────────────────────

def has_cuda() -> bool:
    return torch.cuda.is_available()

def has_mps() -> bool:
    """True on Apple Silicon with PyTorch >= 2.1 MPS backend."""
    return (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

def get_best_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on what's available."""
    if has_cuda():
        return "cuda"
    if has_mps():
        return "mps"
    return "cpu"

def is_apple_silicon() -> bool:
    """True when running on macOS (regardless of MPS availability)."""
    return sys.platform == "darwin"


# ── Accelerator-aware empty_cache ─────────────────────────────────────────────

def empty_cache(device: Optional[str] = None) -> None:
    """
    Release unused memory back to the OS.
    Works on CUDA, MPS, and CPU (no-op on CPU).
    """
    dev = device or get_best_device()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        # MPS does not expose an explicit empty_cache() in all PyTorch builds;
        # call it when available, silently skip otherwise.
        fn = getattr(torch.mps, "empty_cache", None)
        if fn is not None:
            fn()


def synchronize(device: Optional[str] = None) -> None:
    """Block until all pending GPU/MPS ops are done."""
    dev = device or get_best_device()
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        fn = getattr(torch.mps, "synchronize", None)
        if fn is not None:
            fn()


# ── Stream / Event shims ──────────────────────────────────────────────────────

class _NullEvent:
    """CPU / MPS no-op replacement for torch.cuda.Event."""
    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


def new_event(device: Optional[str] = None):
    """
    Returns a torch.cuda.Event on CUDA, or a NullEvent shim on MPS/CPU.
    Usage pattern stays the same across both backends.
    """
    dev = device or get_best_device()
    if dev == "cuda":
        return torch.cuda.Event()
    return _NullEvent()


# ── NVTX range shims ─────────────────────────────────────────────────────────

def nvtx_push(label: str, device: Optional[str] = None) -> None:
    dev = device or get_best_device()
    if dev == "cuda":
        torch.cuda.nvtx.range_push(label)


def nvtx_pop(device: Optional[str] = None) -> None:
    dev = device or get_best_device()
    if dev == "cuda":
        torch.cuda.nvtx.range_pop()


# ── MLX optional bridge ───────────────────────────────────────────────────────

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


def mlx_available() -> bool:
    return _HAS_MLX


def mlx_svd_lowrank(
    delta_cpu: "torch.Tensor",
    rank: int,
    n_oversamples: int = 5,
    n_iter: int = 2,
):
    """
    Randomized SVD via MLX on Apple Silicon.

    Args:
        delta_cpu : Float32 CPU tensor [n, d]
        rank      : Target rank (r)
        n_oversamples, n_iter: rSVD tuning knobs

    Returns:
        (U_cpu, S_cpu, Vh_cpu) as float32 CPU tensors.
        Returns None if MLX is unavailable or if an error occurs.
    """
    if not _HAS_MLX:
        return None

    try:
        import mlx.core as mx
        import numpy as np

        n, d = delta_cpu.shape
        r_proj = min(rank + n_oversamples, n, d)
        if r_proj < 1:
            return None

        # numpy → mlx (zero-copy on unified-memory M-series)
        x_np = delta_cpu.numpy().astype(np.float32)
        x = mx.array(x_np)

        # Randomized projection
        Omega = mx.random.normal(shape=(d, r_proj))
        Y = x @ Omega
        for _ in range(n_iter):
            Y = x @ (x.T @ Y)

        # QR for stable orthonormal basis
        Q, _ = mx.linalg.qr(Y, stream=mx.cpu)

        # Project and SVD on small matrix
        B = Q.T @ x
        U_b, S, Vh = mx.linalg.svd(B, stream=mx.cpu)
        U = Q @ U_b

        mx.eval(U, S, Vh)

        # Back to numpy → torch (slice Vh to r_proj rows since MLX svd always computes full V)
        U_cpu  = torch.from_numpy(np.array(U, copy=False))
        S_cpu  = torch.from_numpy(np.array(S, copy=False))
        Vh_cpu = torch.from_numpy(np.array(Vh[:r_proj, :], copy=False))
        return U_cpu, S_cpu, Vh_cpu
    except Exception as e:
        # Any MLX failure → caller falls back to PyTorch CPU SVD
        return None


# ── torch.compile backend selection ──────────────────────────────────────────

def get_compile_backend() -> Optional[str]:
    """
    Return the best torch.compile backend for the current platform.

    - CUDA: 'inductor'  (TorchInductor, CUDA-native kernels)
    - MPS : 'inductor'
    - CPU : 'aot_eager'
    """
    if has_cuda() or has_mps():
        return "inductor"
    return "aot_eager"


def get_compile_mode() -> str:
    """
    Return the best torch.compile mode for the current platform.
    'reduce-overhead' requires CUDAGraph support → not available on MPS.
    """
    if has_cuda():
        return "reduce-overhead"
    return "default"


# ── dtype helpers ─────────────────────────────────────────────────────────────

def get_default_dtype(device: Optional[str] = None) -> torch.dtype:
    """
    Best half-precision dtype for the platform.
    - CUDA / MPS: float16  (both support it since PyTorch 2.x)
    - CPU       : bfloat16 (no rounding-overflow for large activations)
    """
    dev = device or get_best_device()
    if dev in ("cuda", "mps"):
        return torch.float16
    return torch.bfloat16


# Monkeypatch torch.mps.capture_to_graph to prevent graph compilation memory leaks on dynamic shapes
if hasattr(torch, "mps"):
    import os
    if os.environ.get("DKV_MPS_CAPTURE_GRAPH", "0") != "1" or not hasattr(torch.mps, "capture_to_graph"):
        class _CaptureToGraphContext:
            def __init__(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def _capture_to_graph():
            return _CaptureToGraphContext()

        torch.mps.capture_to_graph = _capture_to_graph


def get_true_dkv_memory_mb() -> dict:
    """
    Get true allocated/reserved memory in MB.
    Returns:
        dict with keys: 'allocated_mb', 'reserved_mb', 'rss_mb'
    """
    res = {'allocated_mb': 0.0, 'reserved_mb': 0.0, 'rss_mb': 0.0}
    
    # 1. RSS Memory
    try:
        import psutil
        res['rss_mb'] = psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
        
    # 2. Accelerator Memory
    dev = get_best_device()
    if dev == "cuda":
        res['allocated_mb'] = torch.cuda.memory_allocated() / 1e6
        res['reserved_mb'] = torch.cuda.memory_reserved() / 1e6
    elif dev == "mps":
        try:
            res['allocated_mb'] = torch.mps.current_allocated_memory() / 1e6
            res['reserved_mb'] = torch.mps.driver_allocated_memory() / 1e6
        except Exception:
            pass
            
    return res

