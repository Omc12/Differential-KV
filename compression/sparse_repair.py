"""
compression/sparse_repair.py — Phase 4

Implements Low-Rank + Sparse Repair for KV Deltas.
ΔKV ≈ U @ V.T + S (Sparse)
"""

import torch
from dataclasses import dataclass
from typing import Tuple, Optional
from .lowrank import LowRankDelta, compress_lowrank

@dataclass
class LowRankSparseDelta:
    low_rank: LowRankDelta
    sparse_indices: torch.Tensor  # [N_sparse]
    sparse_values: torch.Tensor   # [N_sparse]
    sparse_shape: tuple           # (n_deltas, feat_dim)
    
    def nbytes(self) -> int:
        # low_rank + indices (int32) + values (float16)
        return (self.low_rank.nbytes() + 
                self.sparse_indices.numel() * 4 + 
                self.sparse_values.numel() * 2)

def compress_lowrank_sparse(
    deltas: torch.Tensor, 
    rank: int, 
    sparse_ratio: float = 0.01
) -> LowRankSparseDelta:
    """
    Compress deltas using Low-Rank + Sparse Repair.
    sparse_ratio: fraction of total elements to keep as sparse outliers.
    """
    n, d = deltas.shape
    # 1. Get low-rank approximation
    lr = compress_lowrank(deltas, rank)
    
    # 2. Compute residual
    recon_lr = (lr.U.float().to(deltas.device) @ lr.V.to(deltas.device) * lr.scale).to(deltas.dtype)
    residual = deltas - recon_lr
    
    # 3. Select top-k outliers from residual
    k = int(n * d * sparse_ratio)
    if k > 0:
        flat_res = residual.abs().view(-1)
        values, indices = torch.topk(flat_res, k)
        # Store signed values
        sparse_values = residual.view(-1)[indices].to(torch.float16)
        sparse_indices = indices.to(torch.int32)
    else:
        sparse_values = torch.zeros(0, dtype=torch.float16)
        sparse_indices = torch.zeros(0, dtype=torch.int32)
        
    return LowRankSparseDelta(
        low_rank=lr,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        sparse_shape=(n, d)
    )

def decompress_lowrank_sparse(
    lrs: LowRankSparseDelta,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Reconstruct with sparse repair."""
    # 1. Base low-rank reconstruction
    recon = (lrs.low_rank.U.float() @ lrs.low_rank.V * lrs.low_rank.scale).to(dtype)
    
    # 2. Add sparse repair
    if lrs.sparse_indices.numel() > 0:
        flat_recon = recon.view(-1)
        flat_recon.scatter_add_(0, lrs.sparse_indices.long(), lrs.sparse_values.to(dtype))
        recon = flat_recon.view(lrs.sparse_shape)
        
    return recon
