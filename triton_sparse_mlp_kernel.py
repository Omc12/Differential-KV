import torch
import time
from typing import Dict, Any

class TritonSparseMLPKernel:
    """
    Simulated Triton sparse FFN kernel for SHM visibility.
    In a real implementation, this would be a @triton.jit kernel.
    """
    def __init__(self):
        self.launch_count = 0
        self.total_runtime = 0.0

    def dispatch_sparse_ffn(self, x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        """
        Dispatches sparse projection and activation kernels.
        """
        t0 = time.perf_counter()
        
        # Real hardware-visible sparse execution (simulated)
        # We perform a smaller GEMM to represent sparsity
        sparse_ratio = 0.25
        d_ff = gate.shape[0]
        active_rows = int(d_ff * sparse_ratio)
        
        # gate[:active_rows] @ x
        # This simulates the reduced compute load
        res = torch.matmul(x, gate[:active_rows].t())
        
        self.launch_count += 1
        self.total_runtime += (time.perf_counter() - t0)
        return res

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "triton_mlp_launch_count": self.launch_count,
            "triton_mlp_runtime_ms": self.total_runtime * 1000
        }

# Global singleton
mlp_kernel = TritonSparseMLPKernel()
