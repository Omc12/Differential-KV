"""
dkv.backend
Hardware auto-detection and execution backend selector for DKV.
Supports Apple Silicon (MLX + Metal) and Linux (CUDA + PyTorch).
"""

import sys

def is_mlx_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False

def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_backend() -> str:
    """
    Returns the active execution backend name: 'mlx' or 'cuda' or 'cpu'.
    """
    if sys.platform == "darwin":
        if is_mlx_available():
            return "mlx"
        return "mac_cpu"
    elif is_cuda_available():
        return "cuda"
    else:
        return "cpu"

def is_dkv_core_available() -> bool:
    """
    Checks if the C++/CUDA/Metal native extension dkv_core is loaded.
    """
    try:
        import dkv_core
        return True
    except ImportError:
        return False

def info():
    """
    Prints system hardware and DKV active backend telemetry.
    """
    backend = get_backend()
    has_native = is_dkv_core_available()
    print("=" * 60)
    print("  Differential-KV (dkv) System Telemetry")
    print("=" * 60)
    print(f"  Platform         : {sys.platform}")
    print(f"  Active Backend   : {backend.upper()}")
    print(f"  MLX Available    : {is_mlx_available()}")
    print(f"  CUDA Available   : {is_cuda_available()}")
    print(f"  Native dkv_core  : {'Loaded (C++/Metal/CUDA)' if has_native else 'Not loaded (Python fallback)'}")
    print("=" * 60)
