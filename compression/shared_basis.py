"""
compression/shared_basis.py

Phase 6: Global Basis Differential KV
Implements sharing of the V matrix (basis) across blocks, layers, or heads.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union
from .lowrank import LowRankDelta, compress_lowrank

@dataclass
class SharedBasis:
    V: torch.Tensor          # [rank, feat_dim]
    basis_id: str            # Unique identifier (e.g. "layer_0", "global")
    rank: int
    feat_dim: int
    
    def nbytes(self) -> int:
        return self.V.numel() * 4 # FP32 basis

@dataclass
class SharedBasisDelta:
    U: torch.Tensor          # [n, rank] float16
    basis_id: str
    scale: float
    shape: tuple             # (n, feat_dim)
    sparse_indices: Optional[torch.Tensor] = None # [k] int32
    sparse_values: Optional[torch.Tensor] = None  # [k] float16

    def nbytes(self) -> int:
        nb = self.U.numel() * 2 # FP16
        if self.sparse_indices is not None:
            nb += self.sparse_indices.numel() * 4 # int32
            nb += self.sparse_values.numel() * 2  # float16
        return nb

class SharedBasisManager:
    """
    Manages the lifecycle of shared bases.
    """
    def __init__(self):
        self.bases: Dict[str, SharedBasis] = {}

    def create_basis(self, deltas: torch.Tensor, rank: int, basis_id: str) -> SharedBasis:
        """Extracts basis V from a collection of deltas using SVD."""
        # deltas: [N_total, feat_dim]
        lr = compress_lowrank(deltas, rank)
        basis = SharedBasis(
            V=lr.V,
            basis_id=basis_id,
            rank=rank,
            feat_dim=deltas.shape[1]
        )
        self.bases[basis_id] = basis
        return basis

    def get_basis(self, basis_id: str) -> SharedBasis:
        return self.bases[basis_id]

    def compress_block(
        self, 
        deltas: torch.Tensor, 
        basis_id: str, 
        sparse_ratio: float = 0.0,
        scale: float = 1.0
    ) -> SharedBasisDelta:
        """Projects deltas onto a shared basis."""
        basis = self.get_basis(basis_id)
        V = basis.V.to(deltas.device)
        
        # Project: U = Deltas @ V.T
        # deltas: [n, d], V: [r, d] -> U: [n, r]
        U = (deltas / scale) @ V.t()
        U_fp16 = U.to(torch.float16)
        
        s_idx, s_val = None, None
        if sparse_ratio > 0:
            recon = (U_fp16.float() @ V) * scale
            residual = deltas - recon
            
            n, d = deltas.shape
            k = int(n * d * sparse_ratio)
            if k > 0:
                flat_res = residual.abs().view(-1)
                values, indices = torch.topk(flat_res, k)
                s_val = residual.view(-1)[indices].to(torch.float16)
                s_idx = indices.to(torch.int32)
        
        return SharedBasisDelta(
            U=U_fp16,
            basis_id=basis_id,
            scale=scale,
            sparse_indices=s_idx,
            sparse_values=s_val,
            shape=deltas.shape
        )

    def decompress_block(self, sbd: SharedBasisDelta) -> torch.Tensor:
        """Reconstruct deltas from coefficients and shared basis."""
        basis = self.get_basis(sbd.basis_id)
        V = basis.V.to(sbd.U.device)
        
        recon = (sbd.U.float() @ V) * sbd.scale
        
        if sbd.sparse_indices is not None and sbd.sparse_indices.numel() > 0:
            flat_recon = recon.view(-1)
            flat_recon.scatter_add_(0, sbd.sparse_indices.long(), sbd.sparse_values.float())
            recon = flat_recon.view(sbd.shape)
            
        return recon.to(torch.float16)
