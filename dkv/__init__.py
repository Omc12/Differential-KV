"""
dkv: Differential KV-Cache Management & High-Efficiency Attention Engine
Private Release Package for Apple Silicon (MLX/Metal) & CUDA (Linux/PyTorch)
"""

from dkv.version import __version__
from dkv.backend import (
    get_backend,
    is_mlx_available,
    is_cuda_available,
    is_dkv_core_available,
    info,
)

__all__ = [
    "__version__",
    "get_backend",
    "is_mlx_available",
    "is_cuda_available",
    "is_dkv_core_available",
    "info",
]
