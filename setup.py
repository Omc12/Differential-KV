"""
Root setup.py for Differential-KV (dkv)
Platform-aware installer for Apple Silicon (MLX/Metal) and Linux (CUDA/PyTorch).
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages

# Suppress PyTorch's overly strict CUDA version check if torch is imported
try:
    import torch.utils.cpp_extension
    torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None
except ImportError:
    pass

this_dir = os.path.dirname(os.path.abspath(__file__))
native_dir = os.path.join(this_dir, "ACTIVE_RUNTIME", "native_core", "dkv_core")
include_dir = os.path.join(native_dir, "include")

def generate_embedded_metallib():
    if sys.platform != "darwin":
        return
    
    metal_src = os.path.join(native_dir, "metal", "dkv_decode.metal")
    air_file = os.path.join(native_dir, "dkv_decode.air")
    lib_file = os.path.join(native_dir, "dkv.metallib")
    header_file = os.path.join(native_dir, "src", "dkv_metallib.hpp")
    
    if not os.path.exists(metal_src):
        if not os.path.exists(header_file):
            with open(header_file, "w") as f:
                f.write("#pragma once\nunsigned int dkv_metallib_len = 0;\nunsigned char dkv_metallib[] = { 0 };\n")
        return
        
    try:
        print("[DKV Build] Compiling Metal shader to AIR...")
        subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", metal_src, "-o", air_file], check=True)
        
        print("[DKV Build] Compiling AIR to metallib...")
        subprocess.run(["xcrun", "-sdk", "macosx", "metallib", air_file, "-o", lib_file], check=True)
        
        with open(lib_file, "rb") as f:
            data = f.read()
            
        print(f"[DKV Build] Embedding {len(data)} bytes of metallib into C++ header...")
        with open(header_file, "w") as f:
            f.write("#pragma once\n")
            f.write(f"unsigned int dkv_metallib_len = {len(data)};\n")
            f.write("unsigned char dkv_metallib[] = {\n")
            hex_bytes = [f"0x{b:02x}" for b in data]
            for i in range(0, len(hex_bytes), 16):
                f.write("    " + ", ".join(hex_bytes[i:i+16]) + ",\n")
            f.write("};\n")
            
        if os.path.exists(air_file):
            os.remove(air_file)
        if os.path.exists(lib_file):
            os.remove(lib_file)
        print("[DKV Build] Embedded metallib successfully.")
        
    except Exception as e:
        print(f"[DKV Build] WARNING: Failed to compile Metal library ({e}). Using fallback dummy header.")
        if not os.path.exists(header_file):
            with open(header_file, "w") as f:
                f.write("#pragma once\nunsigned int dkv_metallib_len = 0;\nunsigned char dkv_metallib[] = { 0 };\n")

# Run Metal generator on macOS
generate_embedded_metallib()

# Relative paths for extension compilation
def rel(path):
    return os.path.relpath(os.path.join(native_dir, path), this_dir)

SHARED_SOURCES = [
    rel("src/srl_router.cpp"),
    rel("src/decode_attention.cpp"),
    rel("src/bindings.cpp"),
]

ext_modules = []
cmdclass = {}

try:
    if sys.platform == "darwin":
        from torch.utils.cpp_extension import BuildExtension, CppExtension

        MAC_SOURCES = SHARED_SOURCES + [
            rel("src/compressor_thread_cpu.cpp"),
            rel("src/metal_runtime.mm"),
        ]

        MAC_CXX_FLAGS = [
            "-std=c++20",
            "-O3",
            "-DDKV_APPLE=1",
            "-fvisibility=hidden",
        ]

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
            include_dirs=[native_dir, os.path.join(native_dir, "include")],
            extra_compile_args={"cxx": MAC_CXX_FLAGS},
            extra_link_args=MAC_LINK_FLAGS,
        )
        ext_modules = [ext]
        cmdclass = {"build_ext": BuildExtension}

    else:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        CUDA_SOURCES = SHARED_SOURCES + [
            rel("src/compressor_thread.cpp"),
            rel("src/paging_stream.cu"),
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
            include_dirs=[native_dir, os.path.join(native_dir, "include")],
            libraries=["cusolver", "cublas"],
            extra_compile_args={
                "cxx": CUDA_CXX_FLAGS,
                "nvcc": CUDA_NVCC_FLAGS,
            },
        )
        ext_modules = [ext]
        cmdclass = {"build_ext": BuildExtension}

except Exception as e:
    print(f"[DKV Build] WARNING: Could not set up C++/CUDA extension ({e}). Installing Python-only package.")

setup(
    name="dkv",
    version="1.2.0",
    description="Differential KV-Cache Management & High-Efficiency Attention Engine",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
