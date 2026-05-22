import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path
import torch

_LOADED_LIB = None

class PersistentSparseAttentionRuntime:
    """
    SGC Stage 3C.3: Persistent Sparse Attention Runtime.
    Avoids repeated driver-level setup and launch latency by launching 
    persistent GPU blocks that process consecutive tokens via active step signals.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        global _LOADED_LIB
        if _LOADED_LIB is None:
            self.dll_path = self.workspace_root / "runtime" / f"persistent_sparse_attention_runtime_{os.getpid()}_{int(time.time())}.dll"
            self._compile_and_load()
            _LOADED_LIB = self.lib
        else:
            self.lib = _LOADED_LIB
            
        self.is_active = False

    def _compile_and_load(self):
        cu_file = self.workspace_root / "runtime" / "persistent_sparse_attention_runtime.cu"
        
        msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.40.33807\\bin\\Hostx64\\x64"
        if not os.path.exists(msvc_bin):
            msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64"
            
        print(f"[TSO Compile] Compiling Persistent Sparse Attention CUDA kernel: {cu_file.name}...")
        
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
            print(f"[TSO Compile] Persistent Sparse Attention DLL compiled successfully: {self.dll_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[TSO Compile Error] Compilation failed:\n{e.stderr}\n{e.stdout}", file=sys.stderr)
            raise RuntimeError(f"Failed to compile CUDA kernel: {cu_file.name}")

        self.lib = ctypes.CDLL(str(self.dll_path))
        self.lib.launch_persistent_attention.argtypes = [
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
            ctypes.c_float,   # scale
            ctypes.c_int      # current_step
        ]
        self.lib.launch_persistent_attention.restype = None

    def start_session(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sparse_indices: torch.Tensor, O: torch.Tensor, block_size: int, scale: float):
        """
        Registers active session buffers.
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.sparse_indices = sparse_indices
        self.O = O
        self.block_size = block_size
        self.scale = scale
        self.is_active = True

    def trigger_step(self, step_idx: int):
        """
        Triggers execution on the active cached buffer.
        """
        if self.is_active:
            B, H, S_q, D = self.Q.shape
            _, _, S_k, _ = self.K.shape
            num_sparse_blocks = self.sparse_indices.shape[-1]

            # Enforce float32/int32 contiguous buffers for zero-copy high-performance DLL safety
            q_ptr = self.Q.float().contiguous().data_ptr() if self.Q.dtype != torch.float32 else self.Q.data_ptr()
            k_ptr = self.K.float().contiguous().data_ptr() if self.K.dtype != torch.float32 else self.K.data_ptr()
            v_ptr = self.V.float().contiguous().data_ptr() if self.V.dtype != torch.float32 else self.V.data_ptr()
            idx_ptr = self.sparse_indices.int().contiguous().data_ptr() if self.sparse_indices.dtype != torch.int32 else self.sparse_indices.data_ptr()
            
            if self.O.dtype != torch.float32:
                # Type safe staging to prevent GPU buffer overflow memory corruption
                O_f32 = torch.zeros_like(self.O, dtype=torch.float32)
                self.lib.launch_persistent_attention(
                    ctypes.c_void_p(q_ptr),
                    ctypes.c_void_p(k_ptr),
                    ctypes.c_void_p(v_ptr),
                    ctypes.c_void_p(idx_ptr),
                    ctypes.c_void_p(O_f32.data_ptr()),
                    B, H, S_q, S_k, D, num_sparse_blocks, self.block_size, self.scale,
                    step_idx
                )
                self.O.copy_(O_f32)
            else:
                self.lib.launch_persistent_attention(
                    ctypes.c_void_p(q_ptr),
                    ctypes.c_void_p(k_ptr),
                    ctypes.c_void_p(v_ptr),
                    ctypes.c_void_p(idx_ptr),
                    ctypes.c_void_p(self.O.data_ptr()),
                    B, H, S_q, S_k, D, num_sparse_blocks, self.block_size, self.scale,
                    step_idx
                )
            torch.cuda.synchronize()

    def terminate_session(self):
        """
        Cleans up the active session.
        """
        self.is_active = False

    def __del__(self):
        self.terminate_session()
