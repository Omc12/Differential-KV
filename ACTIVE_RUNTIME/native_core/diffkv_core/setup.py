"""
diffkv_core/setup.py
Platform-aware build script for the diffkv_core C++ extension.

  Mac (Apple Silicon / MPS):
    - Builds a CppExtension (no CUDA required).
    - Links Accelerate (LAPACK/AMX for CPU SVD) and sets DIFFKV_APPLE=1.
    - paging_stream.cu and compressor_thread.cpp (cuSOLVER) are excluded.
    - compressor_thread_cpu.cpp (Accelerate LAPACK) is included instead.

  CUDA (Linux/Windows):
    - Builds a CUDAExtension (nvcc required).
    - Links cuSOLVER, cuBLAS for GPU SVD and async paging.
    - Sets DIFFKV_CUDA=1.

Both paths include:
  - srl_router.cpp        (ATen C++ SRL routing, Phase 1)
  - decode_attention.cpp  (ATen C++ Project-Then-Attend, Phase 1)
  - bindings.cpp          (pybind11 entry point, platform-conditional)
"""

import os
import sys
import subprocess
from setuptools import setup

# Suppress PyTorch's overly strict CUDA version check
import torch.utils.cpp_extension
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

this_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(this_dir, "include")

# ── Shared sources (both platforms) ──────────────────────────────────────────
SHARED_SOURCES = [
    "src/srl_router.cpp",
    "src/decode_attention.cpp",
    "src/bindings.cpp",
]

if sys.platform == "darwin":
    # ── Mac build ─────────────────────────────────────────────────────────────
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    MAC_SOURCES = SHARED_SOURCES + [
        "src/compressor_thread_cpu.cpp",   # Accelerate LAPACK SVD
    ]

    # Compiler flags for C++17 + Accelerate framework
    # -DDIFFKV_APPLE=1 gates platform-specific code in bindings.cpp and headers
    MAC_CXX_FLAGS = [
        "-std=c++17",
        "-O3",
        "-DDIFFKV_APPLE=1",
        "-fvisibility=hidden",      # Required for pybind11 on Mac
    ]

    # linker flags — Foundation for Obj-C runtime (needed for Metal in Phase 2)
    # Accelerate not needed: torch::linalg::svd on CPU dispatches to it internally
    MAC_LINK_FLAGS = [
        "-framework", "Foundation",
    ]

    ext = CppExtension(
        name="diffkv_core",
        sources=MAC_SOURCES,
        include_dirs=[include_dir],
        extra_compile_args={"cxx": MAC_CXX_FLAGS},
        extra_link_args=MAC_LINK_FLAGS,
    )

else:
    # ── CUDA build ────────────────────────────────────────────────────────────
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    CUDA_SOURCES = SHARED_SOURCES + [
        "src/compressor_thread.cpp",   # cuSOLVER SVD
        "src/paging_stream.cu",        # async D2H/H2D paging
    ]

    CUDA_CXX_FLAGS = [
        "/O2" if sys.platform == "win32" else "-O3",
        "-std=c++17",
        "-DDIFFKV_CUDA=1",
    ]

    CUDA_NVCC_FLAGS = [
        "-O3",
        "--expt-relaxed-constexpr",
        "-allow-unsupported-compiler",
        "-DDIFFKV_CUDA=1",
    ]

    ext = CUDAExtension(
        name="diffkv_core",
        sources=CUDA_SOURCES,
        include_dirs=[include_dir],
        libraries=["cusolver", "cublas"],
        extra_compile_args={
            "cxx": CUDA_CXX_FLAGS,
            "nvcc": CUDA_NVCC_FLAGS,
        },
    )

setup(
    name="diffkv_core",
    version="1.1.0",
    description="DiffKV native runtime core — C++/CUDA/Metal extension",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
