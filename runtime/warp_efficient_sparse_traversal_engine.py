import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path
import torch

_LOADED_LIB = None

class WarpEfficientSparseTraversalEngine:
    """
    SGC Stage 3C.2: Warp-Efficient Sparse Traversal Engine.
    Eliminates warp divergence by aligning and sorting block access indices.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        global _LOADED_LIB
        if _LOADED_LIB is None:
            self.dll_path = self.workspace_root / "runtime" / f"warp_efficient_sparse_traversal_engine_{os.getpid()}_{int(time.time())}.dll"
            self._compile_and_load()
            _LOADED_LIB = self.lib
        else:
            self.lib = _LOADED_LIB

    def _compile_and_load(self):
        cu_file = self.workspace_root / "runtime" / "warp_efficient_sparse_traversal_engine.cu"
        
        msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.40.33807\\bin\\Hostx64\\x64"
        if not os.path.exists(msvc_bin):
            msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64"
            
        print(f"[SKF Compile] Compiling Warp-Efficient Traversal CUDA kernel: {cu_file.name}...")
        
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
            print(f"[SKF Compile] Warp Traversal DLL compiled successfully: {self.dll_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[SKF Compile Error] Compilation failed:\n{e.stderr}\n{e.stdout}", file=sys.stderr)
            raise RuntimeError(f"Failed to compile CUDA kernel: {cu_file.name}")

        self.lib = ctypes.CDLL(str(self.dll_path))
        self.lib.launch_warp_efficient_traversal.argtypes = [
            ctypes.c_void_p,  # input_indices
            ctypes.c_void_p,  # aligned_indices
            ctypes.c_int,     # B
            ctypes.c_int,     # S_q
            ctypes.c_int      # num_sparse_blocks
        ]
        self.lib.launch_warp_efficient_traversal.restype = None

    def align_indices(self, sparse_indices: torch.Tensor) -> torch.Tensor:
        """
        Takes sparse indices [B, S_q, N] and aligns them to minimize warp divergence.
        """
        if not sparse_indices.is_cuda:
            raise ValueError("Sparse indices must be reside on CUDA device!")

        B, S_q, num_sparse_blocks = sparse_indices.shape
        aligned_indices = torch.zeros_like(sparse_indices)

        self.lib.launch_warp_efficient_traversal(
            ctypes.c_void_p(sparse_indices.data_ptr()),
            ctypes.c_void_p(aligned_indices.data_ptr()),
            B, S_q, num_sparse_blocks
        )
        return aligned_indices

    def __del__(self):
        pass
