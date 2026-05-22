"""
compression/lowrank.py — Phase 3 Stage A

Low-rank delta representation for KV cache compression.
ΔKV ≈ U @ V.T  where U=[n_deltas, rank], V=[rank, feat_dim]
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass
class LowRankDelta:
    U: torch.Tensor       # [n_deltas, rank] float16
    V: torch.Tensor       # [rank, feat_dim] float16
    shape: tuple
    rank: int
    scale: float
    energy_retained: float = 0.0
    cosine_sim: float = 1.0
    norm_drift: float = 0.0

    def nbytes(self) -> int:
        return self.U.numel() * 2 + self.V.numel() * 2

    def nbytes_vs_fp16(self) -> int:
        return self.U.shape[0] * self.V.shape[1] * 2

    def nbytes_vs_int8(self) -> int:
        return self.U.shape[0] * self.V.shape[1] + 4

    def ratio_vs_fp16(self) -> float:
        return self.nbytes_vs_fp16() / (self.nbytes() + 1e-9)

    def ratio_vs_int8(self) -> float:
        return self.nbytes_vs_int8() / (self.nbytes() + 1e-9)

    def estimate_compute_ops(self) -> int:
        """Estimate FLOPs for reconstruction: U @ V.T"""
        n, r = self.U.shape
        _, d = self.V.shape
        return n * r * d * 2

    def estimate_bandwidth_bytes(self) -> int:
        return self.nbytes()


def compress_lowrank(deltas: torch.Tensor, rank: int) -> LowRankDelta:
    """Compress [n, feat_dim] float32 delta matrix to rank-r approximation."""
    assert deltas.dim() == 2
    n, d = deltas.shape
    rank = min(rank, n, d)
    
    device = deltas.device

    if deltas.numel() == 0:
        return LowRankDelta(
            U=torch.zeros(n, rank, dtype=torch.float16, device=device),
            V=torch.zeros(rank, d, dtype=torch.float16, device=device),
            shape=(n, d), rank=rank, scale=1.0, energy_retained=0.0
        )

    # Perform all operations on CPU to guarantee zero GPU-CPU telemetry synchronizations (.item())
    deltas_cpu = deltas.cpu() if device.type != "cpu" else deltas
    
    scale = deltas_cpu.abs().max().item()
    if scale < 1e-9:
        return LowRankDelta(
            U=torch.zeros(n, rank, dtype=torch.float16, device=device),
            V=torch.zeros(rank, d, dtype=torch.float16, device=device),
            shape=(n, d), rank=rank, scale=1.0, energy_retained=0.0
        )

    x = deltas_cpu / scale
    
    # Sanitize inputs to prevent NaNs/Infs from causing SVD failures
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    except Exception:
        return LowRankDelta(
            U=torch.zeros(n, rank, dtype=torch.float16, device=device),
            V=torch.zeros(rank, d, dtype=torch.float16, device=device),
            shape=(n, d), rank=rank, scale=scale, energy_retained=0.0
        )

    U_r = (U[:, :rank] * S[:rank].unsqueeze(0)).to(torch.float16)
    V_r = Vh[:rank, :].to(torch.float16)
    total = (S**2).sum().item()
    retained = (S[:rank]**2).sum().item() / (total + 1e-12)
    
    # Calculate reconstruction metrics on CPU
    recon = (U_r.float() @ V_r.float()) * scale
    
    orig_norm = deltas_cpu.norm().item()
    recon_norm = recon.norm().item()
    norm_drift = abs(orig_norm - recon_norm) / (orig_norm + 1e-12)
    
    # Flatten for cosine similarity
    orig_flat = deltas_cpu.reshape(-1)
    recon_flat = recon.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)).item()

    # Move U_r and V_r to the original device
    U_r_out = U_r.to(device)
    V_r_out = V_r.to(device)

    return LowRankDelta(U=U_r_out, V=V_r_out, shape=(n, d),
                        rank=rank, scale=scale, energy_retained=float(retained),
                        cosine_sim=cos_sim, norm_drift=norm_drift)


def decompress_lowrank(lr: LowRankDelta,
                       dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Reconstruct [n_deltas, feat_dim] from LowRankDelta."""
    return (lr.U.float() @ lr.V * lr.scale).to(dtype)


def compress_kv_sequence_lowrank(
    kv: torch.Tensor,          # [seq_len, 2, heads, dim]
    anchor_positions: List[int],
    rank: int,
) -> Tuple[dict, dict]:
    """
    Compress all delta blocks in a KV sequence.

    Returns
    -------
    blocks      : dict[anchor_idx -> (LowRankDelta, [token_indices])]
    kv_anchors  : dict[anchor_idx -> Tensor[2, heads, dim]]
    """
    seq_len, _, heads, dim = kv.shape
    feat_dim    = 2 * heads * dim
    anchor_set  = set(anchor_positions)
    sorted_anc  = sorted(anchor_positions)
    blocks      = {}
    kv_anchors  = {ai: kv[ai].clone() for ai in anchor_set if ai < seq_len}

    for i, anchor_idx in enumerate(sorted_anc):
        next_anchor = sorted_anc[i + 1] if i + 1 < len(sorted_anc) else seq_len
        anchor_kv   = kv[anchor_idx].float()
        rows, toks  = [], []

        for t in range(anchor_idx + 1, next_anchor):
            if t in anchor_set:
                break
            rows.append((kv[t].float() - anchor_kv).reshape(-1))
            toks.append(t)

        if rows:
            lr = compress_lowrank(torch.stack(rows), rank)
            blocks[anchor_idx] = (lr, toks)
        else:
            blocks[anchor_idx] = None

    return blocks, kv_anchors


def decompress_kv_sequence_lowrank(
    blocks: dict,
    kv_anchors: dict,
    kv_shape: tuple,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct full KV sequence from low-rank blocks."""
    seq_len, _, heads, dim = kv_shape
    out = torch.zeros(seq_len, 2, heads, dim, dtype=dtype)

    for ai, kv_a in kv_anchors.items():
        if ai < seq_len:
            out[ai] = kv_a.to(dtype)

    for anchor_idx, block_data in blocks.items():
        if block_data is None:
            continue
        lr, toks = block_data
        anchor_kv   = kv_anchors[anchor_idx].float()
        recon_matrix = decompress_lowrank(lr, dtype=torch.float32)  # [n, feat_dim]
        for local_i, tok in enumerate(toks):
            if tok < seq_len:
                delta = recon_matrix[local_i].reshape(2, heads, dim)
                out[tok] = (anchor_kv + delta).to(dtype)

    return out


def estimate_memory(seq_len: int, heads: int, dim: int,
                    rank: int, interval: int = 64) -> dict:
    """Compare memory for FP16 / INT8-DiffKV / LowRank-DiffKV."""
    feat   = 2 * heads * dim
    n_anc  = max(1, seq_len // interval)
    n_del  = seq_len - n_anc

    fp16   = seq_len * feat * 2
    int8   = n_anc * feat * 2 + n_del * feat + n_anc * 4
    lr     = n_anc * feat * 2 + n_del * rank * 2 + n_anc * rank * feat * 4

    return {
        "fp16_bytes":    fp16,
        "int8_bytes":    int8,
        "lowrank_bytes": lr,
        "ratio_fp16":    round(fp16 / (lr + 1e-9), 3),
        "ratio_int8":    round(int8 / (lr + 1e-9), 3),
        "rank": rank, "seq_len": seq_len,
        "recon_flops_per_token": rank * feat * 2,
        "recon_bandwidth_per_token": rank * 2 + (rank * feat * 4) / (seq_len/interval)
    }
