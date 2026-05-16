"""
hardware_materialization/cuda_extension_sparse_ops.py

Provides CUDA-backed sparse primitives for gather/scatter and fused operations.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger("CUDASparseOps")

class CUDASparseOps:
    """
    Materializes hot sparse primitives into CUDA-backed operations.
    Focuses on gather, scatter, and fused sparse updates.
    """
    
    @staticmethod
    def gather_kv(k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather sparse KV pairs from dense cache using CUDA-optimized indexing.
        """
        if not k.is_cuda:
            # Fallback for CPU if needed, though HKM focuses on GPU
            logger.debug("KV gather falling back to CPU implementation.")
            
        # [bsz, n_heads, seq_len, head_dim] -> [bsz, n_heads, top_k, head_dim]
        # Using advanced indexing which is highly optimized in PyTorch CUDA backend
        bsz, n_heads, _, d = k.shape
        batch_idx = torch.arange(bsz, device=k.device).view(bsz, 1, 1)
        head_idx = torch.arange(n_heads, device=k.device).view(1, n_heads, 1)
        
        # This triggers ATen::index which is a compiled CUDA kernel
        k_sparse = k[batch_idx, head_idx, indices, :]
        v_sparse = v[batch_idx, head_idx, indices, :]
        
        return k_sparse, v_sparse

    @staticmethod
    def scatter_update(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor):
        """
        Perform sparse update (scatter-add) into a target tensor.
        """
        # Ensure indices are long for index_add
        if indices.dtype != torch.long:
            indices = indices.long()
            
        # Fused atomic add on CUDA
        target.view(-1).index_add_(0, indices.view(-1), values.view(-1).to(target.dtype))
        return target

    @staticmethod
    def fused_sparse_recon(u, v, anchor, indices, values, scale=1.0):
        """
        Fused low-rank reconstruction + sparse delta.
        """
        # 1. Low-rank: u @ v.T + anchor
        # Use torch.addmm for fused GEMM + bias if possible, or simple matmul
        out = torch.addmm(anchor, u, v, alpha=scale, beta=1.0)
        
        # 2. Sparse update
        if indices is not None and indices.numel() > 0:
            CUDASparseOps.scatter_update(out, indices, values)
            
        return out

    @staticmethod
    def check_capability():
        """Returns True if CUDA operations are available."""
        return torch.cuda.is_available()
