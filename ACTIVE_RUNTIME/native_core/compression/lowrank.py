"""
compression/lowrank.py — Phase 3 Stage A

Low-rank delta representation for KV cache compression.
ΔKV ≈ U @ V.T  where U=[n_deltas, rank], V=[rank, feat_dim]

Mac/MPS: when MLX is installed, uses mlx_svd_lowrank() for rSVD on the
Apple Neural Engine / GPU via unified memory — significantly faster than
PyTorch CPU for large blocks.  Falls back to PyTorch rSVD if MLX fails.
"""

import math
import os
import re
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
    
    # Track F2: Adaptive Residual Budget
    # Check median relative reconstruction error to classify block complexity
    median_err_K = torch.median(rel_error_K).item() if rel_error_K.numel() > 0 else 0.0
    median_err_V = torch.median(rel_error_V).item() if rel_error_V.numel() > 0 else 0.0
    max_median_err = max(median_err_K, median_err_V)
    
    if max_median_err < 0.05:
        # Easy block (prose filler/redundant text): limit to at most 8 residuals
        n_max_residual = min(8, n_max_residual)
    elif max_median_err < 0.15:
        # Medium complexity block: limit to at most 16 residuals
        n_max_residual = min(16, n_max_residual)
    # Otherwise: keep full budget for high-complexity blocks (numbers, factual data, code)

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


_RE_MATH_BOOST = re.compile(
    r'[\+\-\*\/=]|\$\$|\\\[|\\\(|\\begin\{|\\alpha|\\beta|\\gamma|\\delta|\\sum|\\int|\\frac|\\sqrt|_\{|\^'
)
_RE_DEFINITIONS_BOOST = re.compile(
    r'\b(?:is|are|we)\s+(?:defined|referred|called|known)\s+(?:as|by)\b|\brefers?\s+to\b'
    r'|\b(?:denotes?|stands\s+for|represents?)\b|\bwe\s+define\b|\b(?:let\s+us|let)\s+define\b',
    re.IGNORECASE,
)


def _gather_block_token_ids(block, manager) -> Optional[List[int]]:
    """Return this block's token IDs in order, or None if unavailable.

    Shared by _block_boost_rank (SVD rank boost, below) and
    compress_layer_blocks_gpu's residual-capture finalization loop (content-
    aware residual boost) — both need the same "look up this block's actual
    token ids" step, just for different downstream purposes.  One vectorised
    gather + tolist() replaces a per-position .item() loop (256 Python
    round-trips per block).
    """
    if manager is None:
        return None
    sid = getattr(block, "session_id", None)
    all_tids = None
    if getattr(manager, "_session_token_ids", None) is not None:
        all_tids = manager._session_token_ids.get(sid)
    if all_tids is None:
        return None
    positions = [p for p in getattr(block, "token_indices", []) if 0 <= p < len(all_tids)]
    if not positions:
        return None
    idx = torch.tensor(positions, dtype=torch.long, device=all_tids.device)
    return all_tids[idx].tolist()


def _block_boost_rank(block, rank: int, manager) -> int:
    """Return the SVD rank for one block, boosting content-bearing blocks by 1.5x.

    The boost depends only on the block's token IDs, so the result is identical
    for every layer.  Recomputing it per layer meant a full tokenizer.decode()
    plus three regex scans for each of (num_blocks x num_layers) calls — 2,352
    times for a 13K prompt on a 48-layer model.  MLX hit the same wall and
    solved it the same way (mlx_diffkv_wrapper.py:2424): compute once per
    block, then tile the result across layers.

    Cached on the manager as _block_rank_cache[sid][anchor_idx] -> int, and
    dropped by KVRuntimeManager.clear_session.  The text still comes from one
    whole-block tokenizer.decode() so the matched string is byte-for-byte what
    the uncached path produced; joining per-token decodes would differ whenever
    a multi-byte character is split across two tokens.

    ── The predicate is NOT selective in practice ──────────────────────────────
    Measured on the eval corpus (ACTIVE_RUNTIME/nat_paper.txt, ~256-token
    blocks): any-digit fires on 100% of blocks, _RE_MATH_BOOST on 98%, and the
    union on 100%.  _RE_MATH_BOOST's first alternative is the character class
    [\\+\\-\\*\\/=], so a single hyphen ("self-attention", "state-of-the-art")
    matches, and any citation year or section number satisfies the digit test.
    On technical prose the "boost" is therefore unconditional: every block
    compresses at ceil(rank*1.5), i.e. rank 48 when the preset says 32.

    That is not free.  pool_rank is sized ceil(max_rank*1.5) to cover this
    (kv_runtime_manager ~line 519), so the pool's largest tensor V_KV is
    allocated at 48 ranks and fully used, and the rSVD runs at
    r_proj = 48 + oversamples instead of 32 + oversamples.  MLX has no SVD-rank
    boost at all — it runs a flat rank 32 — so this is a CUDA-only divergence
    that costs ~1.5x pool bytes and ~1.5x compression work.

    DIFFKV_RANK_BOOST:
      auto (default) — current behaviour, boost on the predicate above.
      off            — flat `rank` for every block (MLX parity).
    A/B `off` on the GPU before trusting either: it lowers pool VRAM and
    compress time, and its accuracy cost has never been measured.
    """
    if manager is None:
        return rank

    mode = os.environ.get("DIFFKV_RANK_BOOST", "auto").lower()
    if mode == "off":
        return rank

    sid = getattr(block, "session_id", None)
    anchor = getattr(block, "anchor_idx", None)

    rank_cache = None
    if sid is not None and anchor is not None:
        rank_cache = getattr(manager, "_block_rank_cache", None)
        if rank_cache is None:
            rank_cache = manager._block_rank_cache = {}
        cached = rank_cache.setdefault(sid, {}).get(anchor)
        if cached is not None:
            return cached

    tokenizer = getattr(manager, "tokenizer", None)
    block_rank = rank
    if tokenizer is not None:
        block_token_ids = _gather_block_token_ids(block, manager)
        if block_token_ids:
            try:
                block_text = tokenizer.decode(block_token_ids)
                boost = (
                    any(c.isdigit() for c in block_text)
                    or _RE_MATH_BOOST.search(block_text) is not None
                    or _RE_DEFINITIONS_BOOST.search(block_text) is not None
                )
                if boost:
                    block_rank = int(math.ceil(rank * 1.5))
            except Exception:
                pass

    # Telemetry: the boost rate is the whole question — a rate near 100% means
    # this is a flat rank multiplier wearing a heuristic's clothes.  Counted per
    # session (blocks are cached, so each block is decided exactly once).
    if sid is not None:
        _stats = getattr(manager, "_rank_boost_stats", None)
        if _stats is None:
            _stats = manager._rank_boost_stats = {}
        s = _stats.setdefault(sid, {"boosted": 0, "total": 0})
        s["total"] += 1
        if block_rank > rank:
            s["boosted"] += 1

    if rank_cache is not None:
        rank_cache[sid][anchor] = block_rank
    return block_rank


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
    # A full finiteness reduction scans the entire [N, T, 2*H*D] batch and
    # adds a device-wide synchronization point.  Keep it available for
    # diagnosis; CUDA runs opt out by default, while non-CUDA paths retain the
    # defensive behavior.
    _validate_finite = _local_os.environ.get(
        "DIFFKV_COMPRESS_VALIDATE_FINITE",
        "0" if gpu_device.type == "cuda" else "1",
    ) == "1"
    if _validate_finite and not torch.isfinite(deltas).all():
        deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)

    # ── V-side rebalancing for the joint K|V SVD (MLX parity: DIFFKV_V_SCALE) ──
    # When a block's V-delta energy is much smaller than its K-delta energy, the
    # joint SVD spends its rank budget on K and under-represents V.  MLX scales V
    # up by sqrt(eK/eV) before the SVD so V competes for rank, then undoes it on
    # the V factor (mlx_diffkv_wrapper.compress_deferred_prefill_blocks,
    # v_scale_on).  Ported here: keep the ORIGINAL `deltas` for residual/error/
    # storage, feed a V-scaled copy to the SVD only, and divide the factor's V
    # columns back by the same gain after the SVD (below).  Self-consistent —
    # recon from the unscaled factor is original-space, so nothing else changes.
    # Default ON to match MLX; DIFFKV_V_SCALE=0 restores the old CUDA behavior.
    _half_d = feat_dim // 2
    _v_scale_on = _local_os.environ.get("DIFFKV_V_SCALE", "1") != "0"
    _v_gain = None
    if _v_scale_on:
        _eK = (deltas[:, :, :_half_d] ** 2).sum(dim=(1, 2))                    # [N]
        _eV = (deltas[:, :, _half_d:] ** 2).sum(dim=(1, 2))                    # [N]
        _v_gain = torch.sqrt(_eK / _eV.clamp(min=1e-12)).clamp(1.0, 10000.0)   # [N]
        deltas_svd = torch.cat([
            deltas[:, :, :_half_d],
            deltas[:, :, _half_d:] * _v_gain.view(-1, 1, 1),
        ], dim=2)
    else:
        deltas_svd = deltas

    # Token-wise Norm-Normalization (row-wise) on GPU (Phase 41).  Uses the
    # V-scaled deltas so U's per-token scaling matches the SVD input (MLX does
    # the same — token_norms is taken on the scaled batch).
    token_norms = deltas_svd.norm(dim=2)  # [N_blocks, T_active]
    token_norms = torch.clamp(token_norms, min=1e-5)
    deltas_normalized = deltas_svd / token_norms.unsqueeze(2)

    # 3. Batched Randomized SVD — O(T × rank × feat) instead of O(T² × feat)
    #    This is ~30x faster than full SVD for typical rank=8, T=256, feat=256.
    max_rank_for_batch = rank
    block_ranks = []
    for block in blocks_list:
        block_rank = _block_boost_rank(block, rank, manager)
        block_ranks.append(block_rank)
        if block_rank > max_rank_for_batch:
            max_rank_for_batch = block_rank

    n_oversamples = 5
    r_proj = min(max_rank_for_batch + n_oversamples, T_active, feat_dim)
    if r_proj < 1:
        return False

    try:
        Omega = torch.randn(N_blocks, feat_dim, r_proj, device=gpu_device, dtype=torch.float32)
        Y = torch.matmul(deltas_normalized, Omega)                         # [N, T, r_proj]
        # Issue 3 fix: two power iterations instead of one — matches MLX rSVD (n_iter=2)
        # and the single-block CPU path.  Tighter subspace capture improves energy_retained
        # for rank 4-8 blocks (later layers) by 2-5% with negligible extra cost.
        for _ in range(2):
            Y = torch.matmul(deltas_normalized, torch.matmul(deltas_normalized.transpose(1, 2), Y))
        Q, _ = torch.linalg.qr(Y, mode="reduced")              # [N, T, r_proj]
        B = torch.matmul(Q.transpose(1, 2), deltas_normalized)            # [N, r_proj, feat_dim]
        U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)  # tiny matrix — fast!
        U = torch.matmul(Q, U_b)                               # [N, T, r_proj]
    except Exception as e:
        print(f"[DiffKV GPU-rSVD] Batched randomized SVD failed: {e}. Falling back to CPU SVD.")
        return False

    # 4. Extract dynamic rank using S — vectorized over the whole block batch.
    # Was a per-block Python loop with a per-block .item() (N syncs even though
    # S is already on CPU here).  Equivalent batched form: keep 99.9% of the
    # energy, clamp into [4, b_rank] on the truncation branch, else keep b_rank.
    S_cpu = S.cpu()
    block_ranks_t = torch.tensor(block_ranks, dtype=torch.long)          # [N]
    S_sq = S_cpu.float() ** 2                                            # [N, r_proj]
    tot = S_sq.sum(dim=1)                                                # [N]
    cum = torch.cumsum(S_sq, dim=1)                                      # [N, r_proj]
    ge = cum >= (0.999 * tot).unsqueeze(1)                               # [N, r_proj]
    has_idx = ge.any(dim=1)                                              # [N]
    first_idx = ge.float().argmax(dim=1) + 1                             # first True col + 1
    # Truncation branch: tot > 1e-9 AND an index crossed the threshold.
    trunc = has_idx & (tot > 1e-9)
    k_trunc = torch.clamp(torch.minimum(first_idx, block_ranks_t), min=4)
    k = torch.where(trunc, k_trunc, block_ranks_t)
    k = torch.minimum(k, torch.full_like(k, int(T_active)))
    ranks = k.tolist()

    # 5. Conversion and Sanitization
    U_fp16 = U.to(torch.float16)
    Vh_fp16 = Vh.to(torch.float16)
    S_fp16 = S.to(torch.float16)

    # Undo the V-side rebalancing on the FACTOR (not the deltas): divide the V
    # columns of Vh by the same per-block gain so U @ Vh reconstructs original-
    # space V.  Everything downstream (recon, residual, block.V storage) then
    # operates in original space with no further changes.  See the DIFFKV_V_SCALE
    # note above.
    if _v_gain is not None:
        _vg = _v_gain.view(-1, 1, 1).to(Vh_fp16.dtype)
        Vh_fp16 = torch.cat([Vh_fp16[:, :, :_half_d], Vh_fp16[:, :, _half_d:] / _vg], dim=2)

    # Match the input guard: these scans are diagnostic protection, not part
    # of the compression algorithm.  Re-enable them when validating a new
    # model/checkpoint with DIFFKV_COMPRESS_VALIDATE_FINITE=1.
    if _validate_finite:
        if not torch.isfinite(U_fp16).all():
            U_fp16 = torch.nan_to_num(U_fp16, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(Vh_fp16).all():
            Vh_fp16 = torch.nan_to_num(Vh_fp16, nan=0.0, posinf=0.0, neginf=0.0)

    pool = getattr(manager, "native_pool", None) if manager is not None else None

    # OPT-B: Hoist coverage-bonus statics out of the per-block loop — reading
    # the env var and defining the helper once is enough for the whole batch.
    import os as _os_b
    _cov_frac_batch = 0.0
    try:
        _cov_frac_batch = float(_os_b.environ.get("DIFFKV_RESIDUAL_COVERAGE_FRAC", "0"))
    except ValueError:
        pass
    from collections import namedtuple as _namedtuple
    _TopKCov = _namedtuple("_TopKCov", ["indices", "values"])

    def _topk_with_coverage(rel_err_vec, n_budget, cov_frac):
        """topk with optional stride-stratified coverage bonus (GPU tensors)."""
        if cov_frac <= 0.0 or n_budget <= 0:
            return torch.topk(rel_err_vec, k=min(n_budget, rel_err_vec.shape[0]))
        import numpy as _np
        n_cov = min(n_budget, max(1, int(round(cov_frac * n_budget))))
        n_rank = max(0, n_budget - n_cov)
        T = rel_err_vec.shape[0]
        # Stride-sampled coverage indices (CPU arithmetic only, no GPU sync)
        cov_idx_np = _np.unique(_np.round(_np.linspace(0, T - 1, n_cov)).astype(int))
        cov_idx = torch.from_numpy(cov_idx_np).long().to(rel_err_vec.device)
        # Exclude coverage positions from the error-ranked selection
        errs_for_rank = rel_err_vec.clone()
        errs_for_rank[cov_idx] = -1.0
        if n_rank > 0:
            ranked = torch.topk(errs_for_rank, k=min(n_rank, T))
            valid_rank = ranked.indices[ranked.values > 0.0]
            combined = torch.cat([cov_idx, valid_rank])
        else:
            combined = cov_idx
        combined = torch.unique(combined)
        vals = rel_err_vec[combined]
        order = torch.argsort(vals, descending=True)
        combined = combined[order]
        return _TopKCov(indices=combined, values=vals[order])

    # ── Host-sync batching ────────────────────────────────────────────────
    # The per-block finalization used to force FOUR device→host reads per
    # block (max_s .item(), n_sem .item(), median_K .item(), median_V .item()).
    # At 49 blocks × 48 layers that is ~9,400 stream stalls per 13K-token
    # prefill — measured as ~9s of "compress" wall time, i.e. comparable to
    # the entire forward pass.  Restructured into two passes:
    #   Pass 1: all GPU work enqueued asynchronously; the S-derived scalars
    #           are read from ONE tiny S16_cpu snapshot (identical fp16 bits,
    #           identical expressions → identical values), and the medians
    #           stay on-device as 0-d tensors.
    #   One batched transfer moves all medians to the host together.
    #   Pass 2: residual selection consumes the CPU scalars.  The remaining
    #           per-block syncs are the two boolean-mask compactions
    #           (fact_positions indexing), which resolve against a mostly
    #           drained stream.
    # torch.median selects an element (no arithmetic), so deferring its
    # transfer cannot change the value.
    # S_fp16 is the exact representation used by the quantization path.  A
    # CPU cast from the already-transferred FP32 S values has identical FP16
    # bits, so reuse the one D2H transfer for both rank and semantic-width
    # selection.
    S16_cpu = S_cpu.to(torch.float16)
    half_d = feat_dim // 2

    # ── Batched finalization (MLX-parity) ────────────────────────────────────
    # The old code ran TWO Python loops over blocks_list (~49 blocks × 48 layers
    # ≈ 2,352 iterations per 13K prefill), each launching a per-block recon
    # matmul, an int4 pack, norms and medians.  On discrete CUDA memory with
    # eager execution those launches — NOT the SVD — dominated "compress"
    # (~9s, ~51% of prefill; MLX keeps the equivalent batched finalization at
    # ~13% by doing it over the whole batch at once, see
    # mlx_diffkv_wrapper.compress_deferred_prefill_blocks).  Here every
    # GPU-heavy step runs ONCE over the whole [N, ...] batch:
    #   • U_scaled = U·S·token_norms, columns ≥ dynamic-rank[i] zeroed per block
    #   • recon    = bmm(U_scaled, V)              (one batched matmul)
    #   • errors / rel-errors / medians            (batched reductions)
    # Zeroing columns ≥ k[i] is numerically exact vs the old per-block [:k]
    # slice (the dropped columns contribute 0 to the matmul and the singular
    # values are sorted descending), so the residual selection and pool bytes
    # are identical to the per-block path — verified by test_batched_compress_parity.
    #
    # The per-block int4 pack that used to run here was DEAD on this path:
    # compress_layer_blocks_gpu's own write_block() call (below) never forwards
    # U_sem_int4/n_semantic, so write_block re-quantizes U to int8 and
    # n_semantic stays 0 (the stratified store is only fed by the CPU compress
    # and the finalize_compressed_blocks upload).  It is dropped rather than
    # wired up: full int8 U is higher precision than int4-semantic + fp16-factual.
    ranks_t = torch.tensor(ranks, device=gpu_device, dtype=torch.long)          # [N]
    _col_idx = torch.arange(r_proj, device=gpu_device).unsqueeze(0)             # [1, r_proj]
    rank_mask = (_col_idx < ranks_t.unsqueeze(1)).to(U_fp16.dtype)             # [N, r_proj]

    # U scaled by singular values and per-token norms (matches the old u_k).
    U_scaled = U_fp16 * S_fp16.unsqueeze(1)                # [N, T, r_proj]
    U_scaled = U_scaled * token_norms.unsqueeze(2)         # [N, T, r_proj]
    U_masked = U_scaled * rank_mask.unsqueeze(1)           # zero cols ≥ k[i]
    V_masked = Vh_fp16 * rank_mask.unsqueeze(2)            # [N, r_proj, feat]

    # One batched recon over all blocks (replaces N per-block matmuls).
    recon_all = torch.bmm(U_masked.float(), V_masked.float())   # [N, T, feat]
    delta_K_all = deltas[:, :, :half_d]
    delta_V_all = deltas[:, :, half_d:]
    recon_K_all = recon_all[:, :, :half_d]
    recon_V_all = recon_all[:, :, half_d:]

    error_K_all = (delta_K_all - recon_K_all).norm(dim=2)      # [N, T]
    error_V_all = (delta_V_all - recon_V_all).norm(dim=2)
    norm_K_all = delta_K_all.norm(dim=2).clamp(min=1e-8)
    norm_V_all = delta_V_all.norm(dim=2).clamp(min=1e-8)
    rel_error_K_all = error_K_all / norm_K_all
    rel_error_V_all = error_V_all / norm_V_all

    if T_active > 0:
        med_K_all = torch.median(rel_error_K_all, dim=1).values   # [N]
        med_V_all = torch.median(rel_error_V_all, dim=1).values
    else:
        med_K_all = torch.zeros(N_blocks, device=gpu_device)
        med_V_all = torch.zeros(N_blocks, device=gpu_device)
    # One batched device→host transfer for every block's medians (was N).
    _meds_cpu = torch.stack((med_K_all, med_V_all), dim=1).float().cpu()   # [N, 2]

    # n_semantic per block (metadata parity only — dead on this path).  Computed
    # batched on the already-transferred CPU singular values: count values above
    # 5% of the per-block max.  Sorted-descending S makes this equal to the old
    # per-block "count over [:k] then min(k)".
    _maxs_cpu = S16_cpu.max(dim=1, keepdim=True).values                        # [N,1]
    _nsem_cpu = (S16_cpu > (_maxs_cpu * 0.05)).sum(dim=1).clamp(min=1)          # [N]
    _ranks_cpu = torch.tensor(ranks, dtype=_nsem_cpu.dtype)
    _nsem_cpu = torch.minimum(_nsem_cpu, _ranks_cpu).tolist()

    # Lightweight per-block attribute assignment — no GPU compute in this loop.
    for i, block in enumerate(blocks_list):
        k = ranks[i]
        block.U = U_scaled[i, :, :k].contiguous()
        block.V = Vh_fp16[i, :k, :].contiguous()
        block.n_semantic = int(_nsem_cpu[i]) if k > 0 else 0
        block.scale = 1.0
        block.dynamic_rank = k
        block.active_k = None
        block.active_v = None
        block._active_buf_k = None
        block._active_buf_v = None
        block.state = "COMPRESSED"
        block.dirty = True

    # Collect per-block residual selections for one batched pool write below.
    _rk_pos, _rk_val, _rv_pos, _rv_val = [], [], [], []
    for i, block in enumerate(blocks_list):
        rel_error_K = rel_error_K_all[i]
        rel_error_V = rel_error_V_all[i]
        error_K = error_K_all[i]
        error_V = error_V_all[i]
        recon_K = recon_K_all[i]
        recon_V = recon_V_all[i]
        delta_K = delta_K_all[i]
        delta_V = delta_V_all[i]

        # Content-aware residual capture (C10 remediation): the same token
        # boost + owner capture + table capture the MLX wrapper and lowrank.cpp
        # apply, so the CUDA path stops selecting residuals blind. Boosts
        # multiply the rel-error ranking; under the pool's max_residual_tokens
        # truncation the boosted (value/owner/table) rows sort first and are
        # what survives. Requires the tokenizer + session ids already fetched
        # above for the block-rank heuristic; silently skipped when absent.
        # OPT-A: Adaptive residual budget — 3-tier block classifier by median reconstruction error.
        # Mirrors compress_lowrank() CPU path (lines 304-315). Easy prose blocks waste far fewer
        # residual slots; hard factual/code blocks keep the full budget.
        # (Medians read the UNBOOSTED rel errors, matching the MLX wrapper.)
        n_max_residual = int(T_active * 0.15)
        median_err_K = float(_meds_cpu[i, 0])
        median_err_V = float(_meds_cpu[i, 1])
        max_median_err = max(median_err_K, median_err_V)
        if max_median_err < 0.05:
            # Easy block (prose filler / repeated text): cap at 8 residuals.
            n_max_residual = min(8, n_max_residual)
        elif max_median_err < 0.15:
            # Medium complexity: cap at 16 residuals.
            n_max_residual = min(16, n_max_residual)
        # Hard block (factual / code / numbers): keep full budget unchanged.

        # Content-aware residual capture (C10 remediation): the same token
        # boost + owner capture + table capture the MLX wrapper and lowrank.cpp
        # apply, so the CUDA path stops selecting residuals blind. Boosts
        # multiply the rel-error RANKING (after the tier medians above);
        # under the pool's max_residual_tokens truncation the boosted
        # (value/owner/table) rows sort first and are what survives. Budget
        # floor mirrors the other impls: boosted rows + margin, capped at
        # T_active.
        #
        # This block's own token ids — NOT the rank-boost loop's variable
        # above, which only ever held the LAST block of blocks_list by the
        # time execution reaches here (they are two separate loops over the
        # same list).  Silently applying one block's tokens to every other
        # block's residual selection was a pre-existing bug; look them up
        # fresh per block instead of relying on Python for-loop leakage.
        block_token_ids = _gather_block_token_ids(block, manager)
        if block_token_ids and getattr(manager, "tokenizer", None) is not None \
                and len(block_token_ids) == T_active:
            try:
                _sid = getattr(block, "session_id", None)
                _cached_boost = None
                if _sid is not None:
                    _boost_cache = getattr(manager, "_res_capture_boost_rows", None)
                    if _boost_cache is None:
                        _boost_cache = manager._res_capture_boost_rows = {}
                    _session_boosts = _boost_cache.setdefault(_sid, {})
                    _cached_boost = _session_boosts.get(block.anchor_idx)

                if _cached_boost is not None:
                    boost_row, n_boosted = _cached_boost
                else:
                    from native_core.compression.residual_capture import compute_boost_multipliers
                    _tok = manager.tokenizer
                    _cache = getattr(manager, "_res_capture_decode_cache", None)
                    if _cache is None:
                        _cache = manager._res_capture_decode_cache = {}
                    tok_strs = []
                    for _tid in block_token_ids:
                        _s = _cache.get(_tid)
                        if _s is None:
                            _s = _cache[_tid] = _tok.decode([_tid])
                        tok_strs.append(_s)
                    _all = manager._session_token_ids.get(_sid) if getattr(
                        manager, "_session_token_ids", None) is not None else None
                    _total = int(_all.numel()) if _all is not None else len(block_token_ids)
                    _ckey = (_sid, _total)
                    _counts_cache = getattr(manager, "_res_capture_counts", None)
                    if _counts_cache is None:
                        _counts_cache = manager._res_capture_counts = {}
                    _counts = _counts_cache.get(_ckey)
                    if _counts is None and _all is not None:
                        _counts = {}
                        for _t in _all.tolist():
                            _counts[_t] = _counts.get(_t, 0) + 1
                        _counts_cache.clear()   # keep only the latest session state
                        _counts_cache[_ckey] = _counts
                    boost_row, n_boosted = compute_boost_multipliers(
                        tok_strs, block_token_ids, _counts or {}, _total)
                    if _sid is not None:
                        _session_boosts[block.anchor_idx] = (boost_row, n_boosted)

                if boost_row is not None and n_boosted > 0:
                    _bt = torch.tensor(boost_row, device=rel_error_K.device,
                                       dtype=rel_error_K.dtype)
                    rel_error_K = rel_error_K * _bt
                    rel_error_V = rel_error_V * _bt
                    try:
                        _margin = int(os.environ.get("DIFFKV_RESIDUAL_FLOOR_MARGIN", "4"))
                    except ValueError:
                        _margin = 4
                    n_max_residual = max(n_max_residual,
                                         min(T_active, n_boosted + _margin))
            except Exception:
                pass

        fact_positions_K = None
        residual_K_vals = None
        fact_positions_V = None
        residual_V_vals = None

        if T_active > 0 and n_max_residual > 0:
            top_k_K = _topk_with_coverage(rel_error_K, n_max_residual, _cov_frac_batch)
            top_k_V = _topk_with_coverage(rel_error_V, n_max_residual, _cov_frac_batch)

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
        _rk_pos.append(fact_positions_K); _rk_val.append(residual_K_vals)
        _rv_pos.append(fact_positions_V); _rv_val.append(residual_V_vals)

    # ── One batched pool write (replaces N per-block write_block calls) ──
    # The per-block write_block used to launch ~15 kernels each (int8 U quant,
    # V split, anchor/scale/residual writes, SRL descriptor) — ~2,352 calls per
    # 13K prefill.  Here every block in the call is scattered at once.  Reuses
    # U_masked / V_masked (already built for the recon; columns ≥ each block's
    # dynamic rank are zeroed, exactly what the per-block path leaves in the
    # slot), so the pool bytes are bit-identical — proven by
    # test_write_blocks_batched_parity + test_compress_gpu_smoke.
    if pool is not None:
        _pool_rank = pool.U.shape[2]
        _need_alloc = [b for b in blocks_list if getattr(b, 'pool_idx', None) is None]
        if _need_alloc:
            _slots = pool.allocate_blocks(len(_need_alloc))
            for _b, _s in zip(_need_alloc, _slots):
                _b.pool_idx = _s
        for _b in blocks_list:
            _b.pool = pool

        _N = len(blocks_list)
        _max_res = pool.max_residual_tokens
        _rk_pos_pad = torch.full((_N, _max_res), -1, device=gpu_device, dtype=torch.int16)
        _rk_val_pad = torch.zeros((_N, _max_res, heads, head_dim), device=gpu_device, dtype=torch.float16)
        _rv_pos_pad = torch.full((_N, _max_res), -1, device=gpu_device, dtype=torch.int16)
        _rv_val_pad = torch.zeros((_N, _max_res, heads, head_dim), device=gpu_device, dtype=torch.float16)
        for _i in range(_N):
            _fp = _rk_pos[_i]
            if _fp is not None and _fp.numel() > 0:
                _n = min(int(_fp.numel()), _max_res)
                _rk_pos_pad[_i, :_n] = _fp[:_n]
                _rk_val_pad[_i, :_n] = _rk_val[_i][:_n].view(_n, heads, head_dim)
            _fpv = _rv_pos[_i]
            if _fpv is not None and _fpv.numel() > 0:
                _n = min(int(_fpv.numel()), _max_res)
                _rv_pos_pad[_i, :_n] = _fpv[:_n]
                _rv_val_pad[_i, :_n] = _rv_val[_i][:_n].view(_n, heads, head_dim)

        pool.write_blocks_batched(
            pool_indices=torch.tensor([b.pool_idx for b in blocks_list], device=gpu_device, dtype=torch.long),
            U=U_masked[:, :, :_pool_rank].contiguous(),
            V=V_masked[:, :_pool_rank, :].contiguous(),
            anchor_K=torch.stack([b.anchor_kv[0, 0] for b in blocks_list], dim=0),
            anchor_V=torch.stack([b.anchor_kv[0, 1] for b in blocks_list], dim=0),
            scales=torch.tensor([float(getattr(b, 'scale', 1.0)) for b in blocks_list], device=gpu_device),
            seq_len=T_active,
            res_K_positions=_rk_pos_pad, res_K_values=_rk_val_pad,
            res_V_positions=_rv_pos_pad, res_V_values=_rv_val_pad,
        )
        # Clear local GPU tensors on blocks to prevent VRAM leak.
        for _b in blocks_list:
            _b.U = None
            _b.V = None
            _b.residual_K_positions = None
            _b.residual_K_values = None
            _b.residual_V_positions = None
            _b.residual_V_values = None

    if manager is not None and getattr(manager, "_streaming_mgr", None) is not None:
        for _b in blocks_list:
            manager._streaming_mgr.update_metadata_state(_b.session_id, _b.layer_idx, _b)

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


def compress_lowrank_batch(
    deltas: torch.Tensor,  # [B, n, d]
    rank: int,
):
    """
    Compress a batch of [n, feat_dim] float32 delta matrices to rank-r approximations.
    Uses batched Randomized SVD (rSVD) for maximum efficiency on CPU/GPU.
    """
    assert deltas.dim() == 3
    B, n, d = deltas.shape
    rank = min(rank, n, d)
    device = deltas.device
    
    # Perform operations on CPU if not CUDA to avoid syncs
    deltas_cpu = deltas.cpu() if device.type != "cpu" else deltas
    
    # Compute scale per batch item
    scale = deltas_cpu.abs().view(B, -1).max(dim=-1).values  # [B]
    scale = torch.clamp(scale, min=1e-9)
    
    x = deltas_cpu / scale.view(B, 1, 1)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Batched Randomized SVD
    svd_success = False
    U, S, Vh = None, None, None
    
    try:
        n_oversamples = 5
        n_iter = 2
        r_proj = min(rank + n_oversamples, n, d)
        
        # 1. Generate random Gaussian projection matrix
        Omega = torch.randn(d, r_proj, dtype=torch.float32, device=x.device)
        
        # 2. Form sample matrix Y with power iterations
        # x @ Omega: [B, n, d] @ [d, r_proj] -> [B, n, r_proj]
        Y = torch.matmul(x, Omega)
        for _ in range(n_iter):
            Y = torch.matmul(x, torch.matmul(x.transpose(1, 2), Y))
            
        # 3. Orthogonalize Y
        Q, _ = torch.linalg.qr(Y, mode="reduced") # [B, n, r_proj]
        
        # 4. Project original matrix onto low-rank subspace Q
        B_mat = torch.matmul(Q.transpose(1, 2), x)
        
        # 5. Standard SVD on the much smaller matrix B_mat
        U_b, S, Vh = torch.linalg.svd(B_mat, full_matrices=False)
        U = torch.matmul(Q, U_b)
        svd_success = True
    except Exception:
        # Fallback to standard SVD
        pass
        
    if not svd_success:
        try:
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        except Exception:
            # Complete failure fallback
            U = torch.zeros((B, n, rank), dtype=torch.float32, device=device)
            S = torch.zeros((B, rank), dtype=torch.float32, device=device)
            Vh = torch.zeros((B, rank, d), dtype=torch.float32, device=device)
            
    return U, S, Vh, scale
