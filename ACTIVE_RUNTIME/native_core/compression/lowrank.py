"""
compression/lowrank.py — Phase 3 Stage A

Low-rank delta representation for KV cache compression.
ΔKV ≈ U @ V.T  where U=[n_deltas, rank], V=[rank, feat_dim]

Mac/MPS: when MLX is installed, uses mlx_svd_lowrank() for rSVD on the
Apple Neural Engine / GPU via unified memory — significantly faster than
PyTorch CPU for large blocks.  Falls back to PyTorch rSVD if MLX fails.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

try:
    from native_core.mac_utils import mlx_svd_lowrank as _mlx_svd, mlx_available as _mlx_available, has_cuda as _has_cuda
except ImportError:
    def _mlx_svd(*a, **kw): return None
    def _mlx_available(): return False
    def _has_cuda(): return torch.cuda.is_available()


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
    dynamic_rank: int = -1

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
    """
    Compress [n, feat_dim] float32 delta matrix to rank-r approximation.
    Uses Phase 36 Randomized SVD (rSVD) and Energy-Preserving Dynamic Rank.
    """
    assert deltas.dim() == 2
    n, d = deltas.shape
    rank = min(rank, n, d)
    
    device = deltas.device

    if deltas.numel() == 0:
        return LowRankDelta(
            U=torch.zeros(n, rank, dtype=torch.float16, device=device),
            V=torch.zeros(rank, d, dtype=torch.float16, device=device),
            shape=(n, d), rank=rank, scale=1.0, energy_retained=0.0, dynamic_rank=rank
        )

    # Perform all operations on CPU to guarantee zero GPU-CPU telemetry synchronizations (.item())
    deltas_cpu = deltas.cpu() if device.type != "cpu" else deltas
    
    scale = deltas_cpu.abs().max().item()
    if scale < 1e-9:
        return LowRankDelta(
            U=torch.zeros(n, rank, dtype=torch.float16, device=device),
            V=torch.zeros(rank, d, dtype=torch.float16, device=device),
            shape=(n, d), rank=rank, scale=1.0, energy_retained=0.0, dynamic_rank=rank
        )

    x = deltas_cpu / scale
    
    # Sanitize inputs to prevent NaNs/Infs from causing SVD failures
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Phase 36 Randomized SVD (rSVD) with Graceful Fallback ---
    # On Apple Silicon without CUDA, try MLX-accelerated rSVD first.
    U, S, Vh = None, None, None
    svd_success = False

    if _mlx_available() and not _has_cuda():
        mlx_result = _mlx_svd(x, rank, n_oversamples=5, n_iter=2)
        if mlx_result is not None:
            U, S, Vh = mlx_result
            svd_success = True

    if not svd_success:
        try:
            # Oversampling parameter and power iterations
            n_oversamples = 5
            n_iter = 2
            r_proj = min(rank + n_oversamples, n, d)
            
            # 1. Generate random Gaussian projection matrix
            Omega = torch.randn(d, r_proj, dtype=torch.float32)
            
            # 2. Form sample matrix Y with power iterations for stable subspace capture
            Y = x @ Omega
            for _ in range(n_iter):
                Y = x @ (x.T @ Y)
                
            # 3. Orthogonalize Y to find orthonormal basis Q
            Q, _ = torch.linalg.qr(Y, mode="reduced")
            
            # 4. Project original matrix onto low-rank subspace Q
            B = Q.T @ x
            
            # 5. Perform standard SVD on the much smaller matrix B
            U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Q @ U_b
            svd_success = True
        except Exception:
            # Graceful fallback to standard CPU SVD if randomized step fails
            pass

    if not svd_success:
        try:
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        except Exception:
            return LowRankDelta(
                U=torch.zeros(n, rank, dtype=torch.float16, device=device),
                V=torch.zeros(rank, d, dtype=torch.float16, device=device),
                shape=(n, d), rank=rank, scale=scale, energy_retained=0.0, dynamic_rank=rank
            )

    # --- Phase 36 Energy-Preserving Dynamic Rank Selection ---
    total_energy = (S ** 2).sum().item()
    k = rank
    if total_energy > 1e-9:
        cum = torch.cumsum(S ** 2, dim=0)
        threshold = 0.999 * total_energy
        idx = torch.where(cum >= threshold)[0]
        if idx.numel() > 0:
            k = max(4, min(int(idx[0].item() + 1), rank))

    # Slice SVD outputs to dynamic rank k — NO zero-padding.
    # Storing zeros for unused rank slots (rank-k columns) wastes memory proportional
    # to (rank-k)/rank — typically 50-87% of U/V storage for k=4..16, rank=32.
    # NativeBlockPool.write_block() and the GEMM reconstruction path both handle
    # variable-rank tensors correctly via min(U.shape[1], pool_rank) guards.
    U_k = U[:, :k] * S[:k].unsqueeze(0)  # [n, k]
    Vh_k = Vh[:k, :]                      # [k, d]

    # Convert to FP16
    U_k_fp16 = U_k.to(torch.float16)
    Vh_k_fp16 = Vh_k.to(torch.float16)

    # Sanitize outputs against NaNs/Infs
    if not torch.isfinite(U_k_fp16).all():
        U_k_fp16 = torch.nan_to_num(U_k_fp16, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(Vh_k_fp16).all():
        Vh_k_fp16 = torch.nan_to_num(Vh_k_fp16, nan=0.0, posinf=0.0, neginf=0.0)

    retained = (S[:k]**2).sum().item() / (total_energy + 1e-12)
    
    # Calculate reconstruction metrics on CPU using actual (unpadded) k-rank matrices
    recon = (U_k_fp16.float() @ Vh_k_fp16.float()) * scale
    
    orig_norm = deltas_cpu.norm().item()
    recon_norm = recon.norm().item()
    norm_drift = abs(orig_norm - recon_norm) / (orig_norm + 1e-12)
    
    # Flatten for cosine similarity
    orig_flat = deltas_cpu.reshape(-1)
    recon_flat = recon.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)).item()

    # Move unpadded U and V to the original device
    U_out = U_k_fp16.to(device)   # [n, k]
    V_out = Vh_k_fp16.to(device)  # [k, d]

    return LowRankDelta(U=U_out, V=V_out, shape=(n, d),
                        rank=rank, scale=scale, energy_retained=float(retained),
                        cosine_sim=cos_sim, norm_drift=norm_drift, dynamic_rank=k)


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


def compress_layer_blocks_gpu(blocks_list, rank: int, manager = None) -> bool:
    """
    Compress a list of StreamingKVBlock objects batched together entirely on the GPU.
    Uses randomized SVD for efficiency and maintains correct K/V layout.
    """
    if not blocks_list:
        return True

    T_active = blocks_list[0].active_k.shape[2]
    heads = blocks_list[0].active_k.shape[1]
    head_dim = blocks_list[0].active_k.shape[3]
    feat_dim = 2 * heads * head_dim
    gpu_device = blocks_list[0].active_k.device
    N_blocks = len(blocks_list)

    # 1. Properly concatenate K and V along feat dimension [N, T, 2*H*D]
    stacked_k = torch.cat([b.active_k.permute(0, 2, 1, 3).reshape(1, T_active, -1) for b in blocks_list], dim=0)
    stacked_v = torch.cat([b.active_v.permute(0, 2, 1, 3).reshape(1, T_active, -1) for b in blocks_list], dim=0)
    flat_batch = torch.cat([stacked_k, stacked_v], dim=2)

    # 2. Compute GPU deltas
    stacked_anchors = torch.cat([b.anchor_kv.reshape(1, -1) for b in blocks_list], dim=0)
    deltas = (flat_batch.float() - stacked_anchors.unsqueeze(1).float())
    if not torch.isfinite(deltas).all():
        deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)

    # Vectorized scale computation and normalization on GPU (Phase 41)
    # This prevents the scale factor from being applied twice (which caused deviation on long contexts),
    # and reduces N_blocks CUDA syncs (.item() in loop) down to exactly 1 CPU copy.
    scales_t = deltas.abs().max(dim=-1)[0].max(dim=-1)[0]  # [N_blocks]
    scales_t = torch.clamp(scales_t, min=1e-9)
    deltas_normalized = deltas / scales_t.view(N_blocks, 1, 1)
    scales_cpu = scales_t.cpu()

    # 3. Batched Randomized SVD — O(T × rank × feat) instead of O(T² × feat)
    #    This is ~30x faster than full SVD for typical rank=8, T=256, feat=256.
    n_oversamples = 5
    r_proj = min(rank + n_oversamples, T_active, feat_dim)
    if r_proj < 1:
        return False

    try:
        Omega = torch.randn(N_blocks, feat_dim, r_proj, device=gpu_device, dtype=torch.float32)
        Y = torch.matmul(deltas_normalized, Omega)                         # [N, T, r_proj]
        # One power iteration for better subspace capture
        Y = torch.matmul(deltas_normalized, torch.matmul(deltas_normalized.transpose(1, 2), Y))
        Q, _ = torch.linalg.qr(Y, mode="reduced")              # [N, T, r_proj]
        B = torch.matmul(Q.transpose(1, 2), deltas_normalized)            # [N, r_proj, feat_dim]
        U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)  # tiny matrix — fast!
        U = torch.matmul(Q, U_b)                               # [N, T, r_proj]
    except Exception as e:
        print(f"[DiffKV GPU-rSVD] Batched randomized SVD failed: {e}. Falling back to CPU SVD.")
        return False

    # 4. Extract dynamic rank using S
    S_cpu = S.cpu()
    ranks = []
    for i in range(N_blocks):
        tot = (S_cpu[i] ** 2).sum().item()
        k = rank
        if tot > 1e-9:
            cum = torch.cumsum(S_cpu[i] ** 2, dim=0)
            threshold = 0.999 * tot
            idx = torch.where(cum >= threshold)[0]
            if idx.numel() > 0:
                k = max(4, min(int(idx[0].item() + 1), rank))
        ranks.append(k)

    # 5. Conversion and Sanitization
    U_fp16 = U.to(torch.float16)
    Vh_fp16 = Vh.to(torch.float16)
    S_fp16 = S.to(torch.float16)

    # Sanitize outputs against NaNs/Infs (Phase 41 - safety first)
    if not torch.isfinite(U_fp16).all():
        U_fp16 = torch.nan_to_num(U_fp16, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(Vh_fp16).all():
        Vh_fp16 = torch.nan_to_num(Vh_fp16, nan=0.0, posinf=0.0, neginf=0.0)

    pool = getattr(manager, "native_pool", None) if manager is not None else None
    for i, block in enumerate(blocks_list):
        k = ranks[i]
        # Unpadded U/V — shape (T_active, k) and (k, feat_dim).
        # No zero-padding: pool.write_block() handles variable-rank writes via
        # min(U.shape[1], pool_rank) guards, and GEMM reconstruction slices
        # stacked_U[:, :, :max_k] / stacked_V[:, :max_k, :] from dynamic_rank.
        u_k = U_fp16[i, :, :k] * S_fp16[i, :k].unsqueeze(0)  # [T_active, k]
        v_k = Vh_fp16[i, :k, :]                                # [k, feat_dim]

        block.U = u_k.contiguous()
        block.V = v_k.contiguous()
        block.scale = scales_cpu[i].item()  # local CPU read - zero CUDA sync!
        block.dynamic_rank = k
        block.active_k = None
        block.active_v = None
        block.state = "COMPRESSED"
        block.dirty = True

        if pool is not None:
            if getattr(block, 'pool_idx', None) is None:
                block.pool_idx = pool.allocate_block()
            pool.write_block(block.pool_idx, block.U, block.V, block.anchor_kv[0,0], block.anchor_kv[0,1], block.scale, T_active)

        if manager is not None and getattr(manager, "_streaming_mgr", None) is not None:
            manager._streaming_mgr.update_metadata_state(block.session_id, block.layer_idx, block)

    return True
