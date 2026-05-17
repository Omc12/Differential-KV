import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path
import torch

_LOADED_LIB = None

class FusedSparseAttentionKernel:
    """
    SGC Stage 3C.2: Fused Sparse Attention Kernel Executor.
    Links the PyTorch tensor interface directly with the raw CUDA kernel.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        global _LOADED_LIB
        if _LOADED_LIB is None:
            self.dll_path = self.workspace_root / "runtime" / f"fused_sparse_attention_kernel_{os.getpid()}_{int(time.time())}.dll"
            self._compile_and_load()
            _LOADED_LIB = self.lib
        else:
            self.lib = _LOADED_LIB

    def _compile_and_load(self):
        cu_file = self.workspace_root / "runtime" / "fused_sparse_attention_kernel.cu"
        
        # Paths for Windows Visual Studio 2022 MSVC
        msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.40.33807\\bin\\Hostx64\\x64"
        if not os.path.exists(msvc_bin):
            msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64"
            
        print(f"[SKF Compile] Compiling Fused Sparse Attention CUDA kernel: {cu_file.name}...")
        
        cmd = [
            "nvcc", "-shared", "-O3",
            "-allow-unsupported-compiler",
            "--compiler-options", "/D_USRDLL /D_WINDLL",
            "-ccbin", msvc_bin,
            str(cu_file),
            "-o", str(self.dll_path)
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            print(f"[SKF Compile] Fused Attention DLL compiled successfully: {self.dll_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[SKF Compile Error] Compilation failed:\n{e.stderr}\n{e.stdout}", file=sys.stderr)
            raise RuntimeError(f"Failed to compile CUDA kernel: {cu_file.name}")

        self.lib = ctypes.CDLL(str(self.dll_path))
        self.lib.launch_fused_sparse_attention.argtypes = [
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # sparse_indices
            ctypes.c_void_p,  # O
            ctypes.c_int,     # B
            ctypes.c_int,     # H
            ctypes.c_int,     # S_q
            ctypes.c_int,     # S_k
            ctypes.c_int,     # D
            ctypes.c_int,     # num_sparse_blocks
            ctypes.c_int,     # block_size
            ctypes.c_float    # scale
        ]
        self.lib.launch_fused_sparse_attention.restype = None

    def execute(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sparse_indices: torch.Tensor, block_size: int, scale: float) -> torch.Tensor:
        """
        Executes fused sparse attention.
        """
        if not Q.is_cuda or not K.is_cuda or not V.is_cuda or not sparse_indices.is_cuda:
            raise ValueError("All input tensors must be reside on CUDA device!")

        B, H, S_q, D = Q.shape
        _, _, S_k, _ = K.shape
        num_sparse_blocks = sparse_indices.shape[-1]

        # Allocate output tensor contiguously in float32
        O_f32 = torch.zeros((B, H, S_q, D), dtype=torch.float32, device=Q.device)

        self.lib.launch_fused_sparse_attention(
            ctypes.c_void_p(Q.float().contiguous().data_ptr()),
            ctypes.c_void_p(K.float().contiguous().data_ptr()),
            ctypes.c_void_p(V.float().contiguous().data_ptr()),
            ctypes.c_void_p(sparse_indices.int().contiguous().data_ptr()),
            ctypes.c_void_p(O_f32.data_ptr()),
            B, H, S_q, S_k, D, num_sparse_blocks, block_size, scale
        )
        return O_f32.to(Q.dtype)

    def __del__(self):
        pass
