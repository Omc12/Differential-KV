import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path
import torch

_LOADED_LIB = None

class FusedMetadataExecutionLayer:
    """
    SGC Stage 3C.2: Fused Metadata Execution Layer.
    Natively selects and populates sparse block indices completely inside the GPU.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        global _LOADED_LIB
        if _LOADED_LIB is None:
            self.dll_path = self.workspace_root / "runtime" / f"fused_metadata_execution_layer_{os.getpid()}_{int(time.time())}.dll"
            self._compile_and_load()
            _LOADED_LIB = self.lib
        else:
            self.lib = _LOADED_LIB

    def _compile_and_load(self):
        cu_file = self.workspace_root / "runtime" / "fused_metadata_execution_layer.cu"
        
        msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.40.33807\\bin\\Hostx64\\x64"
        if not os.path.exists(msvc_bin):
            msvc_bin = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64"
            
        print(f"[SKF Compile] Compiling Fused Metadata Execution CUDA kernel: {cu_file.name}...")
        
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
            print(f"[SKF Compile] Fused Metadata DLL compiled successfully: {self.dll_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[SKF Compile Error] Compilation failed:\n{e.stderr}\n{e.stdout}", file=sys.stderr)
            raise RuntimeError(f"Failed to compile CUDA kernel: {cu_file.name}")

        self.lib = ctypes.CDLL(str(self.dll_path))
        self.lib.launch_fused_metadata.argtypes = [
            ctypes.c_void_p,  # attention_scores
            ctypes.c_void_p,  # sparse_indices
            ctypes.c_int,     # B
            ctypes.c_int,     # H
            ctypes.c_int,     # S_q
            ctypes.c_int,     # S_k
            ctypes.c_int,     # num_sparse_blocks
            ctypes.c_int,     # block_size
            ctypes.c_float    # confidence_threshold
        ]
        self.lib.launch_fused_metadata.restype = None

    def execute_metadata_routing(
        self, attention_scores: torch.Tensor, seq_len_q: int, num_sparse_blocks: int, block_size: int, threshold: float
    ) -> torch.Tensor:
        """
        Natively computes and schedules sparse indexes entirely on the GPU.
        """
        if not attention_scores.is_cuda:
            raise ValueError("Attention scores must reside on CUDA device!")

        B, H, S_k = attention_scores.shape
        sparse_indices = torch.zeros((B, seq_len_q, num_sparse_blocks), dtype=torch.int32, device=attention_scores.device)

        self.lib.launch_fused_metadata(
            ctypes.c_void_p(attention_scores.float().contiguous().data_ptr()),
            ctypes.c_void_p(sparse_indices.data_ptr()),
            B, H, seq_len_q, S_k, num_sparse_blocks, block_size, threshold
        )
        return sparse_indices

    def __del__(self):
        pass
