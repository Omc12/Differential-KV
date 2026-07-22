"""
dkv_core/setup.py
Platform-aware build script for the dkv_core C++ extension.

  Mac (Apple Silicon / MPS):
    - Builds a CppExtension (no CUDA required).
    - Links Accelerate (LAPACK/AMX for CPU SVD) and sets DKV_APPLE=1.
    - paging_stream.cu and compressor_thread.cpp (cuSOLVER) are excluded.
    - compressor_thread_cpu.cpp (Accelerate LAPACK) is included instead.

  CUDA (Linux/Windows):
    - Builds a CUDAExtension (nvcc required).
    - Links cuSOLVER, cuBLAS for GPU SVD and async paging.
    - Sets DKV_CUDA=1.

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

# ── Generate embedded Metal library byte array (Mac-only) ──────────────────────
def generate_embedded_metallib():
    if sys.platform != "darwin":
        return
    
    metal_src = os.path.join(this_dir, "metal", "dkv_decode.metal")
    air_file = os.path.join(this_dir, "dkv_decode.air")
    lib_file = os.path.join(this_dir, "dkv.metallib")
    header_file = os.path.join(this_dir, "src", "dkv_metallib.hpp")
    
    # Check if Metal source exists
    if not os.path.exists(metal_src):
        # Write dummy header if source is missing to avoid compiler errors
        if not os.path.exists(header_file):
            with open(header_file, "w") as f:
                f.write("#pragma once\nunsigned int dkv_metallib_len = 0;\nunsigned char dkv_metallib[] = { 0 };\n")
        return
        
    try:
        print("[DKV Build] Compiling Metal shader to AIR...")
        subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", metal_src, "-o", air_file], check=True)
        
        print("[DKV Build] Compiling AIR to metallib...")
        subprocess.run(["xcrun", "-sdk", "macosx", "metallib", air_file, "-o", lib_file], check=True)
        
        # Read the binary metallib
        with open(lib_file, "rb") as f:
            data = f.read()
            
        # Write C++ header file
        print(f"[DKV Build] Embedding {len(data)} bytes of metallib into C++ header...")
        with open(header_file, "w") as f:
            f.write("#pragma once\n")
            f.write(f"unsigned int dkv_metallib_len = {len(data)};\n")
            f.write("unsigned char dkv_metallib[] = {\n")
            # Write as hex chunks (16 bytes per line)
            hex_bytes = [f"0x{b:02x}" for b in data]
            for i in range(0, len(hex_bytes), 16):
                f.write("    " + ", ".join(hex_bytes[i:i+16]) + ",\n")
            f.write("};\n")
            
        # Clean up temporary files
        if os.path.exists(air_file):
            os.remove(air_file)
        if os.path.exists(lib_file):
            os.remove(lib_file)
        print("[DKV Build] Embedded metallib successfully.")
        
    except Exception as e:
        print(f"[DKV Build] WARNING: Failed to compile Metal library ({e}).")
        # Write fallback dummy header so C++ compilation does not fail
        if not os.path.exists(header_file):
            with open(header_file, "w") as f:
                f.write("#pragma once\nunsigned int dkv_metallib_len = 0;\nunsigned char dkv_metallib[] = { 0 };\n")

# Run generator before setuptools setup()
generate_embedded_metallib()

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
        "src/metal_runtime.mm",            # Custom Metal runtime
    ]

    # Compiler flags for C++20 + Accelerate framework
    # -DDKV_APPLE=1 gates platform-specific code in bindings.cpp and headers
    MAC_CXX_FLAGS = [
        "-std=c++20",
        "-O3",
        "-DDKV_APPLE=1",
        "-fvisibility=hidden",      # Required for pybind11 on Mac
    ]

    # linker flags — Foundation for Obj-C runtime (needed for Metal in Phase 2)
    # Accelerate not needed: torch::linalg::svd on CPU dispatches to it internally
    import torch as _torch
    _torch_lib_dir = os.path.join(os.path.dirname(_torch.__file__), "lib")
    MAC_LINK_FLAGS = [
        "-framework", "Foundation",
        "-framework", "Metal",
        f"-Wl,-rpath,{_torch_lib_dir}",
    ]

    ext = CppExtension(
        name="dkv_core",
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
        "-DDKV_CUDA=1",
    ]

    CUDA_NVCC_FLAGS = [
        "-O3",
        "--expt-relaxed-constexpr",
        "-allow-unsupported-compiler",
        "-DDKV_CUDA=1",
    ]

    ext = CUDAExtension(
        name="dkv_core",
        sources=CUDA_SOURCES,
        include_dirs=[include_dir],
        libraries=["cusolver", "cublas"],
        extra_compile_args={
            "cxx": CUDA_CXX_FLAGS,
            "nvcc": CUDA_NVCC_FLAGS,
        },
    )

setup(
    name="dkv_core",
    version="1.1.0",
    description="DKV native runtime core — C++/CUDA/Metal extension",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
