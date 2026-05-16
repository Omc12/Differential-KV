import torch
import time
from typing import Dict, Any

class TritonTokenCollapseKernel:
    """
    Simulated Triton gather/scatter kernels for token collapse.
    Ensures that token-level sparsity is hardware-visible.
    """
    def __init__(self):
        self.launch_count = 0
        self.total_runtime = 0.0

    def gather_active_tokens(self, x: torch.Tensor, indices: torch.Tensor):
        t0 = time.perf_counter()
        # Simulated gather kernel
        res = x[:, indices, :]
        self.launch_count += 1
        self.total_runtime += (time.perf_counter() - t0)
        return res

    def scatter_active_tokens(self, x: torch.Tensor, indices: torch.Tensor, target_len: int):
        t0 = time.perf_counter()
        # Simulated scatter kernel
        bsz, _, d = x.shape
        out = torch.zeros((bsz, target_len, d), device=x.device, dtype=x.dtype)
        out[:, indices, :] = x
        self.launch_count += 1
        self.total_runtime += (time.perf_counter() - t0)
        return out

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "triton_atc_launch_count": self.launch_count,
            "triton_atc_runtime_ms": self.total_runtime * 1000
        }

# Global singleton
atc_kernel = TritonTokenCollapseKernel()
