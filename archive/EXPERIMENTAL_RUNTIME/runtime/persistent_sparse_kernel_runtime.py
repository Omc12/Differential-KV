import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path
import torch

_LOADED_LIB = None

class PersistentSparseKernelRuntime:
    """
    SGC Stage 3C.2: Persistent Sparse Kernel Runtime.
    Caches attention states inside persistent GPU buffers to bypass host-side kernel launch churn.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        global _LOADED_LIB
        if _LOADED_LIB is None:
            self.dll_path = self.workspace_root / "runtime" / f"persistent_sparse_kernel_runtime_{os.getpid()}_{int(time.time())}.dll"
            self._compile_and_load()
            _LOADED_LIB = self.lib
        else:
            self.lib = _LOADED_LIB
        self.persistent_buffers = {}  # session_id -> persistent GPU tensor

    def _compile_and_load(self):
        cu_file = self.workspace_root / "runtime" / "persistent_sparse_kernel_runtime.cu"
        
        msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.40.33807\\bin\\Hostx64\\x64"
        if not os.path.exists(msvc_bin):
            msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64"
            
        print(f"[SKF Compile] Compiling Persistent Sparse Kernel CUDA runtime: {cu_file.name}...")
        
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
            print(f"[SKF Compile] Persistent Kernel DLL compiled successfully: {self.dll_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[SKF Compile Error] Compilation failed:\n{e.stderr}\n{e.stdout}", file=sys.stderr)
            raise RuntimeError(f"Failed to compile CUDA kernel: {cu_file.name}")

        self.lib = ctypes.CDLL(str(self.dll_path))
        self.lib.launch_persistent_sparse.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # persistent_buffer
            ctypes.c_int,     # B
            ctypes.c_int      # Size
        ]
        self.lib.launch_persistent_sparse.restype = None

    def execute_persistent_accumulation(self, session_id: str, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Maintains a persistent buffer on the GPU for the session, accumulating input.
        """
        if not input_tensor.is_cuda:
            raise ValueError("Input tensor must reside on CUDA device!")

        B = input_tensor.shape[0]
        Size = input_tensor.numel() // B

        # Allocate persistent buffer if it doesn't exist
        if session_id not in self.persistent_buffers:
            self.persistent_buffers[session_id] = torch.zeros(input_tensor.shape, dtype=torch.float32, device=input_tensor.device)

        p_buffer = self.persistent_buffers[session_id]

        self.lib.launch_persistent_sparse(
            ctypes.c_void_p(input_tensor.float().contiguous().data_ptr()),
            ctypes.c_void_p(p_buffer.data_ptr()),
            B, Size
        )
        return p_buffer.to(input_tensor.dtype)

    def clear_session(self, session_id: str):
        if session_id in self.persistent_buffers:
            del self.persistent_buffers[session_id]

    def __del__(self):
        pass
