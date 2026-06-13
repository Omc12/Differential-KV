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


def pack_int4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack float32 U_semantic of shape [S, R] into int8 U_sem_packed of shape [S // 2, R].
    S does not need to be even; if odd, we pad with a row of zeros before packing.
    Quantizes values to the range [-8, 7].
    """
    S, R = x.shape
    device = x.device
    
    # Pad to even S if odd
    if S % 2 != 0:
        x_padded = torch.cat([x, torch.zeros((1, R), device=device, dtype=x.dtype)], dim=0)
    else:
        x_padded = x
        
    S_pad = x_padded.shape[0]
    
    # Scale per column: [R]
    max_abs = x_padded.abs().max(dim=0).values
    scale = torch.clamp(max_abs / 7.0, min=1e-5)
    
    # Quantize to [-8, 7]
    q = torch.clamp(torch.round(x_padded / scale.unsqueeze(0)), -8, 7).to(torch.int8)
    
    # Shift to unsigned [0, 15] for safe bitwise operation
    q_shifted = (q + 8).to(torch.uint8 if hasattr(torch, "uint8") else torch.int8)
    
    # Pack consecutive rows
    even = q_shifted[0::2, :]
    odd = q_shifted[1::2, :]
    
    # even is in low 4 bits, odd in high 4 bits
    packed = (even & 0x0F) | ((odd & 0x0F) << 4)
    return packed.to(torch.int8), scale


def unpack_int4(packed: torch.Tensor, scale: torch.Tensor, original_S: int) -> torch.Tensor:
    """
    Unpack int8 U_sem_packed of shape [S // 2, R] to float16 U_semantic of shape [S, R].
    Slices the result back to original_S.
    """
    S_half, R = packed.shape
    device = packed.device
    
    # Cast to uint8 to avoid signed bitwise operation bugs (e.g. sign extension on shift)
    packed_u8 = packed.to(torch.uint8)
    
    # Extract low and high 4 bits
    even = packed_u8 & 0x0F
    odd = (packed_u8 >> 4) & 0x0F
    
    # Subtract the offset of 8 that was added during packing to recover signed range [-8, 7]
    even_signed = even.to(torch.float32) - 8.0
    odd_signed = odd.to(torch.float32) - 8.0
    
    # Interleave
    unpacked = torch.zeros((S_half * 2, R), device=device, dtype=torch.float32)
    unpacked[0::2, :] = even_signed
    unpacked[1::2, :] = odd_signed
    
    # Slice to original_S and scale
    unpacked = unpacked[:original_S, :] * scale.unsqueeze(0)
    return unpacked.to(torch.float16)


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
    residual_K_positions: Optional[torch.Tensor] = None
    residual_K_values:    Optional[torch.Tensor] = None
    residual_V_positions: Optional[torch.Tensor] = None
    residual_V_values:    Optional[torch.Tensor] = None
    U_sem_int4:           Optional[torch.Tensor] = None
    U_sem_scale:          Optional[torch.Tensor] = None
    U_fact_fp16:          Optional[torch.Tensor] = None
    n_semantic:           int = 0

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


def compress_lowrank(
    deltas: torch.Tensor,
    rank: int,
    error_threshold: float = 0.08,
    max_residual_frac: float = 0.15,
    token_norms: Optional[torch.Tensor] = None,
) -> LowRankDelta:
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

    # ── Singular Value Stratified Quantization (Solution 2) ──
    s_vals = S[:k]
    max_s = s_vals.max().item() if s_vals.numel() > 0 else 0.0
    s_threshold = max_s * 0.05
    n_semantic = int((s_vals > s_threshold).sum().item())
    n_semantic = max(1, min(n_semantic, k))
    
    U_semantic = U_k_fp16[:, :n_semantic]
    U_factual = U_k_fp16[:, n_semantic:]
    
    U_sem_int4, U_sem_scale = pack_int4(U_semantic)
    U_fact_fp16 = U_factual.clone()

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

    # ── Post-SVD Sparse Residual Storage (Solution 1) ──
    half_d = d // 2
    delta_K = deltas_cpu[:, :half_d]
    delta_V = deltas_cpu[:, half_d:]
    recon_K = recon[:, :half_d]
    recon_V = recon[:, half_d:]

    error_K = (delta_K - recon_K).norm(dim=1)
    error_V = (delta_V - recon_V).norm(dim=1)

    norm_K = delta_K.norm(dim=1).clamp(min=1e-8)
    norm_V = delta_V.norm(dim=1).clamp(min=1e-8)

    rel_error_K = error_K / norm_K
    rel_error_V = error_V / norm_V

    n_max_residual = int(n * max_residual_frac)
    
    fact_positions_K = None
    residual_K_vals = None
    fact_positions_V = None
    residual_V_vals = None

    if n > 0 and n_max_residual > 0:
        top_k_K = torch.topk(rel_error_K, k=min(n_max_residual, n))
        top_k_V = torch.topk(rel_error_V, k=min(n_max_residual, n))
        
        mask_K = (top_k_K.values > error_threshold) & (error_K[top_k_K.indices] > 1e-4)
        fact_positions_K = top_k_K.indices[mask_K]
        
        mask_V = (top_k_V.values > error_threshold) & (error_V[top_k_V.indices] > 1e-4)
        fact_positions_V = top_k_V.indices[mask_V]
        
        if fact_positions_K.numel() > 0:
            res_K_vals = (delta_K - recon_K)[fact_positions_K]
            if token_norms is not None:
                res_K_vals = res_K_vals * token_norms.cpu()[fact_positions_K.cpu()].unsqueeze(1)
            residual_K_vals = res_K_vals.to(torch.float16).to(device)
            fact_positions_K = fact_positions_K.to(torch.int16).to(device)
        else:
            fact_positions_K = None
            residual_K_vals = None
            
        if fact_positions_V.numel() > 0:
            res_V_vals = (delta_V - recon_V)[fact_positions_V]
            if token_norms is not None:
                res_V_vals = res_V_vals * token_norms.cpu()[fact_positions_V.cpu()].unsqueeze(1)
            residual_V_vals = res_V_vals.to(torch.float16).to(device)
            fact_positions_V = fact_positions_V.to(torch.int16).to(device)
        else:
            fact_positions_V = None
            residual_V_vals = None

    return LowRankDelta(U=U_out, V=V_out, shape=(n, d),
                        rank=rank, scale=scale, energy_retained=float(retained),
                        cosine_sim=cos_sim, norm_drift=norm_drift, dynamic_rank=k,
                        residual_K_positions=fact_positions_K,
                        residual_K_values=residual_K_vals,
                        residual_V_positions=fact_positions_V,
                        residual_V_values=residual_V_vals,
                        U_sem_int4=U_sem_int4.to(device) if U_sem_int4 is not None else None,
                        U_sem_scale=U_sem_scale.to(device) if U_sem_scale is not None else None,
                        U_fact_fp16=U_fact_fp16.to(device) if U_fact_fp16 is not None else None,
                        n_semantic=n_semantic)


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
    import os as _local_os
    if _local_os.environ.get("DIFFKV_DIAGNOSTICS", "0") == "1":
        if not torch.isfinite(deltas).all():
            deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)

    # Token-wise Norm-Normalization (row-wise) on GPU (Phase 41)
    token_norms = deltas.norm(dim=2)  # [N_blocks, T_active]
    token_norms = torch.clamp(token_norms, min=1e-5)
    deltas_normalized = deltas / token_norms.unsqueeze(2)

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
    if _local_os.environ.get("DIFFKV_DIAGNOSTICS", "0") == "1":
        if not torch.isfinite(U_fp16).all():
            U_fp16 = torch.nan_to_num(U_fp16, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(Vh_fp16).all():
            Vh_fp16 = torch.nan_to_num(Vh_fp16, nan=0.0, posinf=0.0, neginf=0.0)

    pool = getattr(manager, "native_pool", None) if manager is not None else None
    for i, block in enumerate(blocks_list):
        k = ranks[i]
        u_k = U_fp16[i, :, :k] * S_fp16[i, :k].unsqueeze(0)  # [T_active, k]
        u_k = u_k * token_norms[i].unsqueeze(1)               # Scale U by token norms
        v_k = Vh_fp16[i, :k, :]                                # [k, feat_dim]

        # ── Singular Value Stratified Quantization (Solution 2) ──
        s_vals = S_fp16[i, :k]
        max_s = s_vals.max().item() if s_vals.numel() > 0 else 0.0
        s_threshold = max_s * 0.05
        n_sem = int((s_vals > s_threshold).sum().item())
        n_sem = max(1, min(n_sem, k))
        
        u_semantic = u_k[:, :n_sem]
        u_factual = u_k[:, n_sem:]
        
        u_sem_int4, u_sem_scale = pack_int4(u_semantic)
        u_fact_fp16 = u_factual.clone()

        block.U = u_k.contiguous()
        block.V = v_k.contiguous()
        block.U_sem_int4 = u_sem_int4.to(gpu_device)
        block.U_sem_scale = u_sem_scale.to(gpu_device)
        block.U_fact_fp16 = u_fact_fp16.to(gpu_device)
        block.n_semantic = n_sem
        block.scale = 1.0
        block.dynamic_rank = k
        block.active_k = None
        block.active_v = None
        block._active_buf_k = None
        block._active_buf_v = None
        block.state = "COMPRESSED"
        block.dirty = True

        # ── Post-SVD Sparse Residual Storage on GPU ──
        recon = u_k.float() @ v_k.float()  # [T_active, feat_dim]
        half_d = feat_dim // 2
        delta_K = deltas[i, :, :half_d]
        delta_V = deltas[i, :, half_d:]
        recon_K = recon[:, :half_d]
        recon_V = recon[:, half_d:]

        error_K = (delta_K - recon_K).norm(dim=1)
        error_V = (delta_V - recon_V).norm(dim=1)

        norm_K = delta_K.norm(dim=1).clamp(min=1e-8)
        norm_V = delta_V.norm(dim=1).clamp(min=1e-8)

        rel_error_K = error_K / norm_K
        rel_error_V = error_V / norm_V

        n_max_residual = int(T_active * 0.15)
        
        fact_positions_K = None
        residual_K_vals = None
        fact_positions_V = None
        residual_V_vals = None

        if T_active > 0 and n_max_residual > 0:
            top_k_K = torch.topk(rel_error_K, k=min(n_max_residual, T_active))
            top_k_V = torch.topk(rel_error_V, k=min(n_max_residual, T_active))
            
            mask_K = (top_k_K.values > 0.08) & (error_K[top_k_K.indices] > 1e-4)
            fact_positions_K = top_k_K.indices[mask_K]
            
            mask_V = (top_k_V.values > 0.08) & (error_V[top_k_V.indices] > 1e-4)
            fact_positions_V = top_k_V.indices[mask_V]
            
            if fact_positions_K.numel() > 0:
                residual_K_vals = (delta_K - recon_K)[fact_positions_K].to(torch.float16).to(gpu_device)
                fact_positions_K = fact_positions_K.to(torch.int16).to(gpu_device)
            else:
                fact_positions_K = None
                residual_K_vals = None
                
            if fact_positions_V.numel() > 0:
                residual_V_vals = (delta_V - recon_V)[fact_positions_V].to(torch.float16).to(gpu_device)
                fact_positions_V = fact_positions_V.to(torch.int16).to(gpu_device)
            else:
                fact_positions_V = None
                residual_V_vals = None

        block.residual_K_positions = fact_positions_K
        block.residual_K_values = residual_K_vals
        block.residual_V_positions = fact_positions_V
        block.residual_V_values = residual_V_vals

        if pool is not None:
            if getattr(block, 'pool_idx', None) is None:
                block.pool_idx = pool.allocate_block()
            block.pool = pool
            pool.write_block(
                pool_idx=block.pool_idx,
                U=block.U,
                V=block.V,
                anchor_K=block.anchor_kv[0,0],
                anchor_V=block.anchor_kv[0,1],
                scale=block.scale,
                seq_len=T_active,
                residual_K_positions=block.residual_K_positions,
                residual_K_values=block.residual_K_values,
                residual_V_positions=block.residual_V_positions,
                residual_V_values=block.residual_V_values
            )
            # Clear local GPU tensors on block to prevent VRAM leak
            block.U = None
            block.V = None
            block.residual_K_positions = None
            block.residual_K_values = None
            block.residual_V_positions = None
            block.residual_V_values = None

        if manager is not None and getattr(manager, "_streaming_mgr", None) is not None:
            manager._streaming_mgr.update_metadata_state(block.session_id, block.layer_idx, block)

    return True


def reconstruct_batch_U(pool, idx: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct U of shape [N, max_seq_len, rank] from stratified components:
    pool.U_sem [n_blocks, max_seq_len // 2, rank]
    pool.U_sem_scale [n_blocks, rank]
    pool.U_fact [n_blocks, max_seq_len, rank]
    pool.n_semantic [n_blocks]
    Falls back to pool.U and pool.U_scale if stratified components are not present/used.
    """
    N = idx.shape[0]
    device = pool.device
    dtype = pool.dtype
    
    max_seq_len = pool.U.shape[1]
    rank = pool.U.shape[2]
    
    U_recon = torch.zeros((N, max_seq_len, rank), device=device, dtype=dtype)
    
    idx_list = idx.tolist() if hasattr(idx, 'tolist') else list(idx)
    for i, pool_idx in enumerate(idx_list):
        n_sem = int(pool.n_semantic[pool_idx].item()) if hasattr(pool, "n_semantic") else 0
        seq_len = int(pool.seq_lens[pool_idx].item())
        
        # If stratified quantization is not used (n_sem == 0), fallback to standard pool.U
        if n_sem == 0:
            scale_val = pool.U_scale[pool_idx].to(dtype)
            U_recon[i] = pool.U[pool_idx].to(dtype) * scale_val.view(1, 1)
            continue
            
        # 1. Unpack semantic part
        half_seq = (seq_len + 1) // 2
        packed_sem = pool.U_sem[pool_idx, :half_seq, :n_sem]
        scale_sem = pool.U_sem_scale[pool_idx, :n_sem]
        U_sem = unpack_int4(packed_sem, scale_sem, seq_len).to(dtype)
        U_recon[i, :seq_len, :n_sem] = U_sem
        
        # 2. Copy factual part
        n_fact = rank - n_sem
        if n_fact > 0:
            U_fact = pool.U_fact[pool_idx, :seq_len, :n_fact]
            U_recon[i, :seq_len, n_sem:n_sem+n_fact] = U_fact.to(dtype)
            
    return U_recon

