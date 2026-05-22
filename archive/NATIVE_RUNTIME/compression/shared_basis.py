"""
compression/shared_basis.py

P1 Salvage Integration — Differential KV Active Runtime

Cross-block Shared Basis Compression.
Originally from RESEARCH_PROTOTYPES/compression/shared_basis.py.
Ported into ACTIVE_RUNTIME/compression/ so it is importable on the hot path.

Concept:
  Instead of computing a fresh V (right singular vectors) per compressed block,
  SharedBasisManager extracts a SINGLE basis V from a collection of delta blocks
  for a given layer/session. All blocks then store only their U (coefficients).

  This reduces VRAM for long sessions from:
    O(N_blocks * rank * feat_dim)     [per-block V]
  to:
    O(rank * feat_dim)               [one V per layer]
    + O(N_blocks * seq_len * rank)   [U matrices]

  For 64 blocks at rank=16 and feat_dim=1024:
    Per-block V cost: 64 * 16 * 1024 * 2 bytes = 2MB
    Shared V cost:         16 * 1024 * 2 bytes = 32KB

Usage in KVRuntimeManager (future integration):
    mgr = SharedBasisManager()
    basis = mgr.create_basis(stacked_deltas, rank=16, basis_id="layer_0_session_A")
    compressed = mgr.compress_block(delta_block, basis_id="layer_0_session_A")
    delta_reconstructed = mgr.decompress_block(compressed)
"""

import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from compression.lowrank import compress_lowrank


@dataclass
class SharedBasis:
    V: torch.Tensor          # [rank, feat_dim]  (the shared right singular vectors)
    basis_id: str
    rank: int
    feat_dim: int

    def nbytes(self) -> int:
        return self.V.numel() * 4  # FP32 basis

    def to(self, device):
        self.V = self.V.to(device)
        return self


@dataclass
class SharedBasisDelta:
    U: torch.Tensor          # [n, rank] float16   (per-block coefficients)
    basis_id: str
    scale: float
    shape: tuple             # (n, feat_dim)
    rank: int
    # Optional sparse residual repair
    sparse_indices: Optional[torch.Tensor] = None  # [k] int32
    sparse_values:  Optional[torch.Tensor] = None  # [k] float16

    def nbytes(self) -> int:
        nb = self.U.numel() * 2  # FP16
        if self.sparse_indices is not None:
            nb += self.sparse_indices.numel() * 4  # int32
            nb += self.sparse_values.numel() * 2   # float16
        return nb


class SharedBasisManager:
    """
    Lifecycle manager for shared bases across blocks.

    One basis per (layer, session) pair is the typical usage.
    The basis is created from the first N blocks of a session
    and reused for subsequent blocks.
    """

    def __init__(self):
        self.bases: Dict[str, SharedBasis] = {}

    def create_basis(self, deltas: torch.Tensor, rank: int,
                     basis_id: str) -> SharedBasis:
        """
        Extracts a shared basis V from a collection of delta vectors.

        Parameters
        ----------
        deltas    : [N, feat_dim] float32 — stacked delta blocks
        rank      : target rank
        basis_id  : unique key (e.g. "layer_0_sess_A")

        Returns
        -------
        SharedBasis with V = [rank, feat_dim]
        """
        lr = compress_lowrank(deltas, rank)
        basis = SharedBasis(
            V=lr.V,
            basis_id=basis_id,
            rank=rank,
            feat_dim=deltas.shape[1],
        )
        self.bases[basis_id] = basis
        return basis

    def has_basis(self, basis_id: str) -> bool:
        return basis_id in self.bases

    def get_basis(self, basis_id: str) -> SharedBasis:
        return self.bases[basis_id]

    def compress_block(
        self,
        deltas: torch.Tensor,           # [n, feat_dim] float32
        basis_id: str,
        rank: Optional[int] = None,
        sparse_ratio: float = 0.0,      # fraction of elements to repair via sparse residual
    ) -> SharedBasisDelta:
        """
        Project a delta block onto the shared basis.

        Parameters
        ----------
        deltas      : [n, feat_dim] float32
        basis_id    : must match a previously created basis
        rank        : optional sub-rank (uses full basis rank if None)
        sparse_ratio: if > 0, compute sparse residual repair

        Returns
        -------
        SharedBasisDelta — stores U only; V is shared in the manager
        """
        basis = self.get_basis(basis_id)
        r = min(rank if rank is not None else basis.rank, basis.rank)

        V = basis.V[:r, :].to(deltas.device).float()

        # Scale to prevent FP16 overflow in U
        scale = deltas.abs().max().item() + 1e-6
        U = ((deltas / scale) @ V.t())            # [n, r]
        U_fp16 = U.to(torch.float16)

        s_idx, s_val = None, None
        if sparse_ratio > 0:
            recon = (U_fp16.float() @ V) * scale
            residual = deltas - recon
            n, d = deltas.shape
            k = max(1, int(n * d * sparse_ratio))
            k = min(k, residual.numel())
            flat_res = residual.abs().view(-1)
            _, top_indices = torch.topk(flat_res, k)
            s_val = residual.view(-1)[top_indices].to(torch.float16)
            s_idx = top_indices.to(torch.int32)

        return SharedBasisDelta(
            U=U_fp16,
            basis_id=basis_id,
            scale=scale,
            rank=r,
            shape=deltas.shape,
            sparse_indices=s_idx,
            sparse_values=s_val,
        )

    def decompress_block(self, sbd: SharedBasisDelta) -> torch.Tensor:
        """
        Reconstruct delta from U coefficients + shared V.

        Returns
        -------
        [n, feat_dim] float16
        """
        basis = self.get_basis(sbd.basis_id)
        V = basis.V[:sbd.rank, :].to(sbd.U.device).float()
        recon = (sbd.U.float() @ V) * sbd.scale

        if sbd.sparse_indices is not None and sbd.sparse_indices.numel() > 0:
            flat_recon = recon.view(-1)
            flat_recon.scatter_add_(
                0,
                sbd.sparse_indices.long(),
                sbd.sparse_values.float()
            )
            recon = flat_recon.view(sbd.shape)

        return recon.to(torch.float16)

    def clear_session(self, prefix: str):
        """Remove all bases whose basis_id starts with `prefix`."""
        to_del = [k for k in self.bases if k.startswith(prefix)]
        for k in to_del:
            del self.bases[k]

    def memory_bytes(self) -> int:
        """Total bytes consumed by all cached shared bases."""
        return sum(b.nbytes() for b in self.bases.values())
