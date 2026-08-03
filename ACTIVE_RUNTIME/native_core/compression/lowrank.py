"""
compression/lowrank.py — Phase 3 Stage A

Low-rank delta representation for KV cache compression.
ΔKV ≈ U @ V.T  where U=[n_deltas, rank], V=[rank, feat_dim]

Mac/MPS: when MLX is installed, uses mlx_svd_lowrank() for rSVD on the
Apple Neural Engine / GPU via unified memory — significantly faster than
PyTorch CPU for large blocks.  Falls back to PyTorch rSVD if MLX fails.
"""

import contextlib
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


def _rsvd_omega(*shape, device=None, dtype=torch.float32) -> torch.Tensor:
    """Deterministic random projection for the randomized SVD.

    The rSVD sketches B = A @ Omega with a RANDOM Omega. Every call site used a
    bare `torch.randn`, which draws from the global RNG -- so two runs of the same
    build on the same prompt got different projections, hence a different
    truncated U/V, hence a different reconstruction and different output tokens.

    That was THE source of this runtime's nondeterminism at temperature 0.
    Measured signature: across two identical runs the pool's `anchors_K`,
    `anchors_V`, `scales`, `U_scale`, `seq_lens`, slot mapping and dense/compressed
    split were all IDENTICAL, while `U`, `V_K`, `V_V` and the residuals derived
    from them differed -- i.e. exactly the tensors that depend on Omega. Layer 3's
    attention output then differed at cos~0.25 on the FIRST decode step.

    (Sign flips alone would be harmless -- they cancel between U and Vh. The
    damage comes from rank TRUNCATION: with a near-degenerate spectrum, which
    directions land inside the kept rank-r subspace depends on the sketch, so the
    truncated reconstruction genuinely changes.)

    Uses a dedicated generator rather than torch.manual_seed so it never perturbs
    or depends on global RNG state (sampling, dropout, anything else). Generated
    on CPU so the draw is identical regardless of device backend, then moved.
    Set DKV_RSVD_SEED to change it; there is no reason to make it random again.
    """
    try:
        # MLX calls this DKV_SVD_SEED; accept both.
        seed = int(os.environ.get("DKV_RSVD_SEED",
                                  os.environ.get("DKV_SVD_SEED", "0")))
    except ValueError:
        seed = 0
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    out = torch.randn(*shape, generator=g, dtype=dtype, device="cpu")
    return out.to(device) if device is not None and str(device) != "cpu" else out


def _residual_error_threshold() -> float:
    """Relative-error floor a token must clear to be given an exact residual.

    MLX applies NO such floor: it takes the top-n_res by capture score outright
    (`top_k_indices = mx.argsort(capture_scores)[-n_res:]`) -- the string "0.08"
    does not appear anywhere in mlx_dkv_wrapper.py.

    This side filtered the top-k afterwards with `values > 0.08`, so a token the
    SVD got only MODERATELY wrong never received a residual -- even when that
    error is enough to flip a digit, which is exactly the failure residuals exist
    to prevent. It also meant a raised max_residual could not actually be spent:
    the budget grew, then the filter threw the extra slots away.

    Default 0.0 = MLX behaviour (spend the whole budget on the worst rows).
    Set DKV_RESIDUAL_ERR_THRESHOLD to restore a floor.
    """
    try:
        return float(os.environ.get("DKV_RESIDUAL_ERR_THRESHOLD", "0.0"))
    except ValueError:
        return 0.0


def _exact_keys_enabled(device=None) -> bool:
    """DKV_RESIDUAL_EXACT_KEYS — MLX-parity residual semantics.

    ON (default on every device):
    residual_{K,V}_values hold the anchor-relative EXACT value, so
    `anchor + residual` is the true K/V, and the decode kernel SUBSTITUTES it — replacing the token's score and removing its lossy twin from
    the block's value accumulation. This is what MLX does (it masks the twin and
    re-attends the exact row) and what dkv_decode.metal does.

    OFF: they hold a CORRECTION to the low-rank reconstruction and the kernel ADDS
    them, leaving the twin in. Approximate; kept as an escape hatch.

    Read via a function, not a module constant, because the tests and the
    per-block compress paths toggle it per-process after import.

    The STORAGE FORMAT has to satisfy every decoder that reads it. All of them
    now substitute (see the list at the return), so CUDA no longer runs a
    lower-fidelity residual algorithm than Metal and MLX on identical inputs.

    Pass the target `device`; an explicit DKV_RESIDUAL_EXACT_KEYS (or MLX's
    DKV_RESIDUAL_EXCLUDE_SVD) overrides the default either way.

    Measured on Qwen3.5-2B / MPS, per-layer attention cos vs dense (8217-token
    needle prompt, reproducible runtime). EVERY full_attention layer improves:

        layer   3      7      11      15     19     23
        off   0.093  0.074  -0.049  0.707  0.878  0.872
        on    0.116  0.782   0.048  0.892  0.910  0.926

    Set DKV_RESIDUAL_EXACT_KEYS=0 to revert to the correction form.
    """
    # MLX gates the same behaviour on DKV_RESIDUAL_EXCLUDE_SVD (default "1").
    # Accept either name so a config written against MLX works here.
    v = os.environ.get("DKV_RESIDUAL_EXACT_KEYS")
    if v is None:
        v = os.environ.get("DKV_RESIDUAL_EXCLUDE_SVD")
    if v is not None:
        return str(v).strip().lower() not in ("0", "off", "false", "no", "")

    # No explicit setting: the storage format MUST match EVERY decoder that reads
    # it back. As of the substitution port below, they all agree.
    #
    # SUBSTITUTES (wants EXACT form) -- every reader:
    #   dkv_decode.metal
    #   mlx_dkv_wrapper           (DKV_RESIDUAL_EXCLUDE_SVD default "1")
    #   triton_fused_decode._fused_sparse_decode_kernel      via EXACT_RESIDUAL
    #   triton_fused_decode._fused_decode_combined_kernel    via EXACT_RESIDUAL
    #   remat_cache._scatter_residuals
    #   triton_fused_decode._prefill_fused_history_attend_compiled  (exact_residual arg)
    #   triton_fused_decode.fused_decode_mps
    #   triton_fused_decode._pytorch_vectorized_sparse_attn_decode
    #
    # The last three used to do a bare `scatter_add_` of the residual onto the
    # low-rank row, so exact-form values landed ON TOP of the twin instead of
    # replacing it -- adding nearly the whole key a second time. That made
    # residuals actively HARMFUL: test_sparse_residual_correctness measured
    # attention error 0.047 WITH residuals against 0.014 without. With
    # substitution ported to all three it is 0.000013 with vs 0.014 without, so
    # residuals now help by ~1000x instead of hurting.
    #
    # CUDA therefore no longer needs a lower-fidelity default than Metal/MLX.
    # Set DKV_RESIDUAL_EXACT_KEYS=0 to fall back to correction form.
    return True

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


from collections import namedtuple as _namedtuple

_TopKCov = _namedtuple("_TopKCov", ["indices", "values"])


def _topk_with_coverage(rel_err_vec: torch.Tensor, n_budget: int, cov_frac: float):
    """topk with optional stride-stratified coverage bonus (MLX parity:
    DKV_RESIDUAL_COVERAGE_FRAC). Reserves round(cov_frac * n_budget) slots for
    evenly-spaced positions regardless of their individual error, so residual
    capture is never fully zero-sum on error alone — MLX's own docstring for
    this documents the concrete failure mode it fixes (boosting one high-error
    row evicting an adjacent low-error-but-load-bearing row). Single-block
    granularity — shared by compress_lowrank and the GPU-batched compress path.
    """
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


def compress_lowrank(
    deltas: torch.Tensor,
    rank: int,
    error_threshold: float = None,   # None -> _residual_error_threshold() (MLX: no floor)
    max_residual_frac: float = 0.15,
    token_norms: Optional[torch.Tensor] = None,
    force_exact: bool = False,
    max_residual: Optional[int] = None,
) -> LowRankDelta:
    """
    Compress [n, feat_dim] float32 delta matrix to rank-r approximation.
    Uses Phase 36 Randomized SVD (rSVD) and Energy-Preserving Dynamic Rank.
    """
    if error_threshold is None:
        error_threshold = _residual_error_threshold()
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

    # ── V-side rebalancing for the joint K|V SVD (MLX parity: DKV_V_SCALE) ──
    # Ported from _compress_layer_blocks_gpu_inner (the only one of CUDA's three
    # compress paths that had this) so the single-block sync path also gets it.
    # When a block's V-delta energy is much smaller than its K-delta energy, the
    # joint SVD under-represents V; scale V up before the SVD only, then divide
    # the gain back out of the V factor afterward (below) so decode reconstructs
    # original-space V with no other changes. Default ON to match MLX/the GPU
    # path; DKV_V_SCALE=0 restores the old unscaled behavior.
    half_d = d // 2
    v_scale_on = os.environ.get("DKV_V_SCALE", "1") != "0"
    v_gain = None
    if v_scale_on:
        eK = (deltas_cpu[:, :half_d].float() ** 2).sum()
        eV = (deltas_cpu[:, half_d:].float() ** 2).sum()
        v_gain = torch.sqrt(eK / eV.clamp(min=1e-12)).clamp(1.0, 10000.0)
        deltas_for_svd = torch.cat([
            deltas_cpu[:, :half_d],
            deltas_cpu[:, half_d:] * v_gain,
        ], dim=1)
    else:
        deltas_for_svd = deltas_cpu

    scale = deltas_for_svd.abs().max().item()
    if scale < 1e-9:
        return LowRankDelta(
            U=torch.zeros(n, rank, dtype=torch.float16, device=device),
            V=torch.zeros(rank, d, dtype=torch.float16, device=device),
            shape=(n, d), rank=rank, scale=1.0, energy_retained=0.0, dynamic_rank=rank
        )

    x = deltas_for_svd / scale
    
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
            Omega = _rsvd_omega(d, r_proj, dtype=torch.float32)
            
            # 2. Form sample matrix Y with power iterations for stable subspace capture
            Y = x @ Omega
            for _ in range(n_iter):
                Y = x @ (x.T @ Y)
                
            # 3. Orthogonalize Y to find orthonormal basis Q
            Q, _ = torch.linalg.qr(Y, mode="reduced")
            
            # 4. Project original matrix onto low-rank subspace Q
            B = Q.T @ x

            # 5. Decompose the much smaller matrix B (MLX/GPU-batched-path parity:
            # DKV_COMPRESS_GRAM_SVD). Default ON: eigendecompose the small
            # [r_proj, r_proj] Gram matrix instead of a wide SVD on B -- same
            # reconstruction (sign flips in U_b cancel against Vh), numerically
            # equivalent to the exact SVD (see _compress_layer_blocks_gpu_inner's
            # A/B-validated version of this same trick), just cheaper. Any
            # failure falls through to the exact SVD.
            _gram_ok = False
            if os.environ.get("DKV_COMPRESS_GRAM_SVD", "1") != "0":
                try:
                    G = B @ B.T                                  # [r_proj, r_proj]
                    evals, evecs = torch.linalg.eigh(G)          # ascending
                    evals = evals.flip(-1).clamp(min=0.0)        # descending S^2
                    U_b = evecs.flip(-1)                         # [r_proj, r_proj]
                    S = evals.sqrt()                             # [r_proj] desc
                    Vh = (U_b.T @ B) / S.clamp(min=1e-8).unsqueeze(-1)
                    _gram_ok = True
                except Exception as _ge:
                    print(f"[DKV rSVD] Gram eigh path failed ({_ge}); using exact SVD.")
            if not _gram_ok:
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

    # Undo the V-side rebalancing on the FACTOR (not the deltas): divide the V
    # columns of Vh by the same gain so U @ Vh reconstructs original-space V.
    # Everything downstream (recon, residual, returned V) then operates in
    # original space with no further changes — same as the GPU-batched path.
    if v_gain is not None:
        vg = v_gain.to(Vh_k_fp16.dtype)
        Vh_k_fp16 = torch.cat([Vh_k_fp16[:, :half_d], Vh_k_fp16[:, half_d:] / vg], dim=1)

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

    # Starting budget. MLX uses the POOL's max_residual directly
    # (`b_res = self.max_residual`, then clamps down for easy blocks); this side
    # started from `int(n * 0.15)` = 38 at n=256, so a HARD block -- numbers,
    # codes, exactly the content residuals exist for -- could never receive more
    # than 38 exact tokens no matter how large max_residual was set.
    #
    # That silently defeated raising max_residual to MLX's 128: only force-exact
    # blocks bypassed this cap, so ordinary factual blocks kept the old ceiling.
    # A CUDA-only invention with no MLX counterpart.
    #
    # When the caller passes the real budget, use it (MLX behaviour). The
    # fraction remains as the fallback for callers that do not, so nothing that
    # relied on the old signature changes silently.
    if max_residual is not None and max_residual > 0:
        n_max_residual = min(int(max_residual), n)
    else:
        n_max_residual = int(n * max_residual_frac)

    # Track F2: Adaptive Residual Budget -- matches MLX's ladder exactly
    # (val < 0.05 -> min(8, b); val < 0.15 -> min(16, b); else full budget).
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

    # Residual capacity the caller's pool will actually keep. Selecting past it
    # is not "storing more", it is deciding by position which tokens get dropped.
    _res_cap = int(max_residual) if (max_residual is not None and max_residual > 0) else n

    if n > 0 and (n_max_residual > 0 or force_exact):
        if force_exact and n <= _res_cap:
            # This block was flagged skip_compression because SVD cannot reproduce
            # its exact values (a passcode / formula / table cell — Rule 1 \d{5,}
            # etc), and a LOW SVD error is still not exact ('847291' -> '84').
            # Every token fits the budget, so store them all.
            fact_positions_K = torch.arange(n, device=rel_error_K.device)
            fact_positions_V = torch.arange(n, device=rel_error_V.device)
        else:
            if force_exact:
                # n EXCEEDS the budget. This used to select torch.arange(n) and
                # let the pool keep the leading slice, defended as "keeping the
                # earliest positions, where the exact content sits" -- an
                # assumption about where in the block a code lands, false for
                # anything past offset max_residual. MLX never selects by
                # position; it ranks boosted joint error and takes the top
                # max_residual (mlx_dkv_wrapper.py:2852). Same fix as the batched
                # GPU path above; see the long note there.
                n_max_residual = _res_cap
            # Residual coverage quota (MLX parity: DKV_RESIDUAL_COVERAGE_FRAC) —
            # ported from the GPU-batched compress path so the single-block sync
            # path also reserves evenly-spaced coverage slots instead of pure
            # error-ranked selection. Default 0 (off) preserves prior behavior.
            try:
                cov_frac = float(os.environ.get("DKV_RESIDUAL_COVERAGE_FRAC", "0"))
            except ValueError:
                cov_frac = 0.0
            # ── MLX-parity residual selection ────────────────────────────────
            # This path still ranked rel_error_K and rel_error_V SEPARATELY and
            # kept two different index sets:
            #
            #     top_k_K = _topk_with_coverage(rel_error_K, ...)
            #     top_k_V = _topk_with_coverage(rel_error_V, ...)
            #     fact_positions_K = top_k_K.indices[mask_K]
            #     fact_positions_V = top_k_V.indices[mask_V]
            #
            # which is exactly the form 42eb66c removed from the batched GPU
            # path, left behind here. Two independent defects:
            #
            #  1. SPLIT SETS. A token can be selected for K and not for V. Its
            #     score is then made exact while its VALUE stays the lossy
            #     low-rank reconstruction, so attention locates the token
            #     correctly and reads the wrong content back. That is the
            #     ZEBRA-447 / partial-code signature, and it is a V-side failure
            #     that no amount of K-side work can reach.
            #
            #  2. RELATIVE vs ABSOLUTE. Dividing by each token's own norm ranks
            #     by FRACTIONAL error, so a low-norm token with a small absolute
            #     error outranks a high-norm needle with a large one. Attention
            #     error is absolute -- q.k does not care what fraction of the key
            #     was lost -- so MLX ranks absolute (mlx_dkv_wrapper.py:2547,
            #     errors_v_balanced = errors_v * v_gain; joint = sqrt(eK^2+eV^2)).
            #
            # rel_error_* stays as it is above: MLX uses it for the median tier
            # (:3822) and the budget ladder here mirrors that, unchanged.
            _err_V_bal = error_V if v_gain is None else error_V * v_gain
            joint_err = torch.sqrt(error_K.float() ** 2 + _err_V_bal.float() ** 2)
            top_k_J = _topk_with_coverage(joint_err, min(n_max_residual, n), cov_frac)

            # NOTE error_threshold is now compared against an ABSOLUTE joint
            # score rather than a relative one. That is the same change the
            # batched path made (:1600 compares _residual_error_threshold()
            # against joint_err), so both paths agree, and the default is 0.0 --
            # MLX applies no floor at all -- so nothing is filtered unless
            # DKV_RESIDUAL_ERR_THRESHOLD is set deliberately.
            #
            # A token qualifies if EITHER half is non-degenerate; with one shared
            # index set, dropping it on K alone would strand its V.
            _idx = top_k_J.indices
            mask_J = (top_k_J.values > error_threshold) & (
                (error_K[_idx] > 1e-4) | (error_V[_idx] > 1e-4))
            fact_positions_K = _idx[mask_J]
            fact_positions_V = fact_positions_K
        
        if fact_positions_K.numel() > 0:
            # DKV_RESIDUAL_EXACT_KEYS (MLX parity) — store the residual K as the
            # FULL anchor-relative delta (exact_K - anchor_K) instead of the
            # correction-to-the-reconstruction (delta_K - recon_K).
            #
            # Why: the decode kernels rotate a compressed block's base score at
            # the block's ANCHOR position (that anchor-rotation is exactly what
            # makes Project-Then-Attend cheap -- rank ops per token instead of
            # rebuilding D-dim keys). A residual stored as a correction can only
            # be ADDED on top of that anchor-rotated base, so the token's score
            # keeps an irreducible RoPE phase error that grows with its distance
            # from the anchor -- which corrupts precisely the position-sensitive
            # digit/code tokens residuals exist to preserve verbatim.
            #
            # MLX (the reference implementation, which recalls these codes
            # correctly) never forms that hybrid: it masks the lossy SVD twin of
            # a residual position out of the compressed pool entirely and
            # attends the EXACT row as a real token at its true position. DKV
            # already does the same thing for `fact` anchors (they store
            # k_orig[pos] and the kernel REPLACES the score) -- but facts are
            # capped at 3 slots/block while residuals cover up to max_residual.
            #
            # Storing (exact - anchor) lets the kernel reconstruct the exact key
            # as anchor + residual and REPLACE the score, rotating both terms at
            # the token's true position. Exact, and cheap: two D-dim dots, no
            # D*rank reconstruction, and identical tensor shapes/memory.
            #
            # Default OFF: this changes the stored residual SEMANTICS, so every
            # decode path that consumes residual_K_values must agree.
            #
            # MUST be the same gate as the V half below. This line used to read
            # the environment directly with a default of "0" while the V half
            # called _exact_keys_enabled() (default ON), so K was written in
            # CORRECTION form and V in EXACT form in the same block -- while the
            # Metal kernel, which reads its own flag, substituted both. That is
            # precisely the mixed-semantics state the batch path's comment warns
            # is "strictly worse than not enabling the mode at all".
            _exact_keys = _exact_keys_enabled(device)
            if _exact_keys:
                res_K_vals = delta_K[fact_positions_K]
            else:
                res_K_vals = (delta_K - recon_K)[fact_positions_K]
            if token_norms is not None:
                res_K_vals = res_K_vals * token_norms.cpu()[fact_positions_K.cpu()].unsqueeze(1)
            residual_K_vals = res_K_vals.to(torch.float16).to(device)
            if _exact_keys:
                # MLX uses ONE index set for both halves (`res_k_active` and
                # `res_v_active` are both `take(..., top_k_indices + 1)`), and
                # exact-keys mode REQUIRES that: the kernel removes a residual
                # token's lossy twin from the block's V accumulation keyed on the
                # V positions, and replaces its score keyed on the K positions.
                # If the two sets disagree, a token can lose its approximate V
                # without gaining an exact one (or keep both and double-count).
                fact_positions_V = fact_positions_K
            fact_positions_K = fact_positions_K.to(torch.int16).to(device)
        else:
            fact_positions_K = None
            residual_K_vals = None
            if _exact_keys_enabled(device):
                fact_positions_V = fact_positions_V[:0]

        if fact_positions_V.numel() > 0:
            # Exact-keys mode stores the anchor-relative EXACT value (anchor + res
            # == the true V), mirroring the K half, so the kernel can substitute
            # it for the token's low-rank estimate rather than nudge it.
            if _exact_keys_enabled(device):
                res_V_vals = delta_V[fact_positions_V]
            else:
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
    """Compare memory for FP16 / INT8-DKV / LowRank-DKV."""
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
    solved it the same way (mlx_dkv_wrapper.py:2424): compute once per
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

    DKV_RANK_BOOST:
      off (default, MLX parity) — flat `rank` for every block, matching MLX's
        behavior exactly (MLX has no rank-boost concept at all).
      auto           — the predicate-based 1.5x boost described above. Was the
        default; flip this back on to A/B it against the MLX-parity default —
        its accuracy cost was never conclusively measured either way.
    """
    if manager is None:
        return rank

    mode = os.environ.get("DKV_RANK_BOOST", "off").lower()
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


@contextlib.contextmanager
def _tf32_matmul():
    """Enable TF32 for fp32 matmul inside this block only, then restore.

    torch.backends.cuda.matmul.allow_tf32 is PROCESS-GLOBAL.  Setting it at
    startup (as an earlier version of this work did, from
    hf_dkv_wrapper._configure_cuda_allocator) also changed the fp32 math in
    the decode reconstruction and the block router, which shifted generated
    output across every preset.  Scope it to the compression math so the
    accuracy of decode is untouched.

    Measured worth on A100 (colab/profile_compress_stages.py): compress 4.635 s
    -> 4.453 s, i.e. ~4%.  Small, because 92% of compress is cuSOLVER
    (qr + svd), which does not use TF32 at all.  Kept because it is free and
    confined; DKV_TF32=0 disables it.
    """
    if not (torch.cuda.is_available() and os.environ.get("DKV_TF32", "1") != "0"):
        yield
        return
    _prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _prev


# DKV_ROUTE_TRACE anchor fingerprint: (layer_idx, anchor_idx) -> (|anchor|, slot).
# Detects the SAME logical block being handed a DIFFERENT anchor on a later
# generation, which separates "prefill recomputed it" from "storage moved it".
_ANCHOR_FP: dict = {}
_ANCHOR_FP_SHOWN = 0
_ANCHOR_FP_CASE = None   # DKV_ROUTE_TRACE_TOKEN of the prompt being compared


def compress_layer_blocks_gpu(blocks_list, rank: int, manager = None) -> bool:
    """
    Compress a list of StreamingKVBlock objects batched together entirely on the GPU.
    Uses randomized SVD for efficiency and maintains correct K/V layout.
    """
    if not blocks_list:
        return True
    with _tf32_matmul():
        return _compress_layer_blocks_gpu_inner(blocks_list, rank, manager)


def _compress_layer_blocks_gpu_inner(blocks_list, rank: int, manager = None) -> bool:

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
        "DKV_COMPRESS_VALIDATE_FINITE",
        "0" if gpu_device.type == "cuda" else "1",
    ) == "1"
    if _validate_finite and not torch.isfinite(deltas).all():
        deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)

    # ── V-side rebalancing for the joint K|V SVD (MLX parity: DKV_V_SCALE) ──
    # When a block's V-delta energy is much smaller than its K-delta energy, the
    # joint SVD spends its rank budget on K and under-represents V.  MLX scales V
    # up by sqrt(eK/eV) before the SVD so V competes for rank, then undoes it on
    # the V factor (mlx_dkv_wrapper.compress_deferred_prefill_blocks,
    # v_scale_on).  Ported here: keep the ORIGINAL `deltas` for residual/error/
    # storage, feed a V-scaled copy to the SVD only, and divide the factor's V
    # columns back by the same gain after the SVD (below).  Self-consistent —
    # recon from the unscaled factor is original-space, so nothing else changes.
    # Default ON to match MLX; DKV_V_SCALE=0 restores the old CUDA behavior.
    _half_d = feat_dim // 2
    _v_scale_on = _local_os.environ.get("DKV_V_SCALE", "1") != "0"
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

    # r_proj is the randomized-SVD projection width; it sets the size of every
    # cuSOLVER call below (qr on [T, r_proj], svd/eigh on [r_proj, ...]).
    #
    # ── The batched-kernel cliff (measured, A100, colab/profile_compress_stages) ──
    # cuSOLVER's genuinely batched Jacobi solvers (syevjBatched for eigh,
    # gesvdjBatched for svd) cap at 32x32.  Above that, PyTorch loops per matrix:
    #   eigh [N,r,r]:  r=32 -> 0.006 ms/call   r=33 -> 0.812 ms/call  (~130x cliff)
    #   svd  [N,r,f]:  r=32 -> 0.884 ms/call   r=33 -> 1.348 ms/call
    # With N_blocks x num_layers = 2,352 calls/prefill, keeping r_proj <= 32 takes
    # the Gram eigh from ~2.1 s to ~0.015 s — the single biggest compress lever,
    # far past the 1.9x the Gram swap gives at r_proj=53.
    #
    # BUT r_proj = max_rank + oversamples: even DKV_RANK_BOOST=off (rank 32)
    # gives 32+5 = 37, still over the cliff.  Two knobs make r_proj<=32 reachable:
    #   DKV_RSVD_OVERSAMPLES (default 5) — randomized-SVD slack.  With the two
    #     power iterations below, 0-2 is usually enough; rank 32 + 0 = 32 batches.
    #   DKV_RSVD_MAX_RPROJ (default 0=off) — hard cap on r_proj.  Blocks that
    #     wanted a higher rank are capped to it (fidelity trade — A/B recall).
    # Recommended batched recipe to A/B:
    #   DKV_COMPRESS_GRAM_SVD=1 DKV_RANK_BOOST=off DKV_RSVD_MAX_RPROJ=32
    try:
        n_oversamples = int(_local_os.environ.get("DKV_RSVD_OVERSAMPLES", "5"))
    except ValueError:
        n_oversamples = 5
    n_oversamples = max(0, n_oversamples)
    r_proj = min(max_rank_for_batch + n_oversamples, T_active, feat_dim)
    try:
        _max_rproj = int(_local_os.environ.get("DKV_RSVD_MAX_RPROJ", "0"))
    except ValueError:
        _max_rproj = 0
    if _max_rproj > 0:
        r_proj = min(r_proj, _max_rproj)
    if r_proj < 1:
        return False

    try:
        Omega = _rsvd_omega(N_blocks, feat_dim, r_proj, device=gpu_device, dtype=torch.float32)
        Y = torch.matmul(deltas_normalized, Omega)                         # [N, T, r_proj]
        # Issue 3 fix: two power iterations instead of one — matches MLX rSVD (n_iter=2)
        # and the single-block CPU path.  Tighter subspace capture improves energy_retained
        # for rank 4-8 blocks (later layers) by 2-5% with negligible extra cost.
        for _ in range(2):
            Y = torch.matmul(deltas_normalized, torch.matmul(deltas_normalized.transpose(1, 2), Y))
        Q, _ = torch.linalg.qr(Y, mode="reduced")              # [N, T, r_proj]
        B = torch.matmul(Q.transpose(1, 2), deltas_normalized)            # [N, r_proj, feat_dim]

        # ── Small-matrix decomposition of B [N, r_proj, feat_dim] ────────────
        # torch.linalg.svd(B) is the dominant compress cost. B is [r_proj≈53,
        # feat=2048]; cuSOLVER's batched SVD (gesvdjBatched) only covers matrices
        # up to 32x32, so this batch of N falls back to a per-matrix loop of
        # N_blocks x num_layers = 2,352 sequential decompositions per 13K prompt.
        # Profiled on A100 (colab/profile_compress_stages.py, 2026-07-17):
        # svd = 3.9s of a 4.6s compress (84.5%), ~1.67 ms/block. The rSVD
        # MATMULS around it total ~0.04s — the cost is entirely cuSOLVER.
        #
        # DKV_COMPRESS_GRAM_SVD=1 replaces the wide SVD with an eigendecomp of
        # the small [r_proj, r_proj] Gram matrix G = B Bᵀ:
        #     G = U_b diag(S²) U_bᵀ            (eigh, ascending → flip to desc)
        #     S  = sqrt(clamp(eigvals, 0))
        #     Vh = diag(1/S) U_bᵀ B
        # This reconstructs B identically to the SVD (sign flips in U_b cancel
        # against Vh, and U = Q U_b feeds block.U while Vh feeds block.V, so the
        # product U_scaled·Vh is sign-invariant). Measured on the SAME A100:
        # 2.1s vs 3.9s → 1.9x on the SVD, compress ~4.6s → ~2.8s, prefill
        # 14.9s → ~13s. Reconstruction error 8.2e-6 vs the SVD's 8.5e-6 (equal;
        # the pool's downstream int8 U quantization is ~9.2e-3, ~1000x larger).
        # Squaring cond(B) costs the SMALLEST singular values precision — exactly
        # the tail the 99.9% energy truncation below discards.
        #
        # DEFAULT ON (A/B-validated on A100, 2026-07-20, colab/gram_eigh_decision.py
        # --gpu-ab @8k): compress 2.26s→1.51s (1.50x) with recall IDENTICAL to the
        # exact SVD baseline (both 33% at that harness's samples; dense ceiling 100%)
        # — it changes SPEED, not quality, because the factorisation is numerically
        # equivalent (Part 1 recon error ~6e-7 << the int8-U quant floor 9.2e-3).
        # Set DKV_COMPRESS_GRAM_SVD=0 to force the exact SVD. Any runtime failure
        # falls through to the exact SVD below. NOTE: the r_proj<=32 recipe
        # (DKV_RSVD_MAX_RPROJ=32) is a SEPARATE fidelity trade that DROPPED recall
        # to 0% in the same A/B — it stays OFF by default; do not enable it blindly.
        _gram_ok = False
        if _local_os.environ.get("DKV_COMPRESS_GRAM_SVD", "1") != "0":
            try:
                G = torch.matmul(B, B.transpose(1, 2))                     # [N, r, r]
                evals, evecs = torch.linalg.eigh(G)                        # ascending
                evals = evals.flip(-1).clamp(min=0.0)                      # descending S²
                U_b = evecs.flip(-1)                                       # [N, r, r]
                S = evals.sqrt()                                           # [N, r] desc
                Vh = torch.matmul(U_b.transpose(1, 2), B) / S.clamp(min=1e-8).unsqueeze(-1)
                _gram_ok = True
            except Exception as _ge:
                print(f"[DKV GPU-rSVD] Gram eigh path failed ({_ge}); using exact SVD.")
        if not _gram_ok:
            U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)  # tiny matrix — fast!
        U = torch.matmul(Q, U_b)                               # [N, T, r_proj]
    except Exception as e:
        print(f"[DKV GPU-rSVD] Batched randomized SVD failed: {e}. Falling back to CPU SVD.")
        return False

    # 4. Extract dynamic rank using S — vectorized over the whole block batch.
    # Was a per-block Python loop with a per-block .item() (N syncs even though
    # S is already on CPU here).  Equivalent batched form: keep 99.9% of the
    # energy, clamp into [4, b_rank] on the truncation branch, else keep b_rank.
    S_cpu = S.cpu()
    # Cap the per-block target rank at r_proj: U/S/Vh only have r_proj columns,
    # so a block_rank above r_proj (possible when DKV_RSVD_MAX_RPROJ caps
    # r_proj below the boosted rank) would otherwise record a dynamic_rank larger
    # than the number of factor columns actually stored, and block.U =
    # U_scaled[:, :, :k] would silently keep only r_proj of them — a rank/data
    # mismatch the decode path then trusts.  Without the cap r_proj is always
    # >= max block_rank, so this clamp is a no-op on the default path.
    block_ranks_t = torch.tensor(block_ranks, dtype=torch.long).clamp(max=r_proj)  # [N]
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
    # operates in original space with no further changes.  See the DKV_V_SCALE
    # note above.
    if _v_gain is not None:
        _vg = _v_gain.view(-1, 1, 1).to(Vh_fp16.dtype)
        Vh_fp16 = torch.cat([Vh_fp16[:, :, :_half_d], Vh_fp16[:, :, _half_d:] / _vg], dim=2)

    # Match the input guard: these scans are diagnostic protection, not part
    # of the compression algorithm.  Re-enable them when validating a new
    # model/checkpoint with DKV_COMPRESS_VALIDATE_FINITE=1.
    if _validate_finite:
        if not torch.isfinite(U_fp16).all():
            U_fp16 = torch.nan_to_num(U_fp16, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(Vh_fp16).all():
            Vh_fp16 = torch.nan_to_num(Vh_fp16, nan=0.0, posinf=0.0, neginf=0.0)

    pool = getattr(manager, "native_pool", None) if manager is not None else None

    # OPT-B: Hoist coverage-bonus statics out of the per-block loop — reading
    # the env var once is enough for the whole batch. _topk_with_coverage is
    # now module-level (shared with compress_lowrank's sync path).
    _cov_frac_batch = 0.0
    try:
        _cov_frac_batch = float(os.environ.get("DKV_RESIDUAL_COVERAGE_FRAC", "0"))
    except ValueError:
        pass

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
    # mlx_dkv_wrapper.compress_deferred_prefill_blocks).  Here every
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
        # Same MLX alignment as the per-block path: start from the POOL's real
        # budget, not int(0.15 * T_active). The comment above says "hard blocks
        # keep the full budget" -- they did not, because the starting value was
        # already capped at 38 for T_active=256 regardless of max_residual.
        _pool_max_res = getattr(pool, "max_residual_tokens", None) if pool is not None else None
        if _pool_max_res:
            n_max_residual = min(int(_pool_max_res), T_active)
        else:
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
        # Reset PER BLOCK. Assigning this only inside the boost branch would let
        # one block's multipliers leak into the next block's ranking through the
        # loop variable — the same failure already called out above for
        # block_token_ids.
        _boost_vec = None

        block_token_ids = _gather_block_token_ids(block, manager)

        # ALIGN TO THE ACTIVE ROWS. _gather_block_token_ids returns
        # block.token_indices, which is [anchor, anchor+1, ... ] -- 1 + T_active
        # entries, because the anchor occupies block-local slot 0 and active_k is
        # k[..., 1:] (streaming_sparse_ingest.py:1246 vs :1216). The gate below
        # demanded len == T_active exactly, so for a FULL block it compared
        # 257 == 256 and the whole content-aware boost was skipped -- silently,
        # with no else branch, on every full block there has ever been.
        #
        # That boost is CUDA's counterpart to MLX's is_core capture
        # (mlx_dkv_wrapper.py:2711 flags digits, all-caps runs of length >= 2,
        # '-' and '_', and :2794 multiplies joint_errors by the result so those
        # tokens win the top-k). Without it the residual set is chosen on
        # reconstruction error alone, and a code like ZEBRA-4471-QUARTZ has to
        # out-error 250 tokens of prose filler to earn a slot -- in a block whose
        # low median error puts it in the "easy" tier with a budget of EIGHT.
        #
        # _boost_vec is multiplied against joint_err, which is indexed by DELTA
        # row j == active token j == token_indices[1 + j]. So the row must be
        # built from the tokens AFTER the anchor; using the raw list would also
        # shift every boost one position off even where the length matched.
        _ids_active = None
        if block_token_ids:
            if len(block_token_ids) == T_active + 1:
                _ids_active = block_token_ids[1:]      # drop the anchor
            elif len(block_token_ids) == T_active:
                _ids_active = block_token_ids          # already active-aligned
        block_token_ids = _ids_active

        if block_token_ids and getattr(manager, "tokenizer", None) is not None:
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
                    _boost_vec = _bt
                    try:
                        _margin = int(os.environ.get("DKV_RESIDUAL_FLOOR_MARGIN", "4"))
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

        _force_exact = bool(getattr(block, "skip_compression", False))

        # Residuals are stored as (delta - recon).  For non-exact blocks `recon`
        # is the fp16-U reconstruction (recon_K/recon_V above) and the lossy
        # residual just captures the worst-reconstructed rows.  For force_exact
        # blocks we need decode to rebuild `delta` bit-for-bit — but the pool
        # stores U as int8 (write_blocks_batched:632-638) and the Triton decode
        # reads that int8 U (the stratified fp16-U proxy is never populated on
        # this GPU compress path, n_semantic=0).  A residual computed against the
        # fp16 recon therefore leaves the int8-U quant error UNABSORBED at decode:
        # int8's single per-block scale pushes low-norm rows (a passcode embedded
        # in filler is exactly that) into a sliver of the range -> up to ~17%
        # per-token error, enough to flip marginal digit tokens ('4657'->'46577').
        # Recompute recon against the SAME int8-dequant U decode reads so the
        # residual absorbs that error and reconstruction is exact.
        recon_K_for_res, recon_V_for_res = recon_K, recon_V
        # EVERY block, not just force-exact ones. The comment above states the
        # invariant — "the residual absorbs that error and reconstruction is
        # exact" — but the `_force_exact and` gate applied it only to skip blocks.
        # For an ordinary compressed block the residual was
        #     delta - recon_fp16
        # while decode reconstructs recon_int8, so the corrected value came out
        #     recon_int8 + (delta - recon_fp16) = delta + (recon_int8 - recon_fp16)
        # i.e. the ENTIRE int8-U quantisation error survived, uncorrected, on
        # exactly the tokens residuals exist to make exact.
        #
        # The magnitude is not marginal. The `absorbed` telemetry below is that
        # same quantity, and on a Qwen3.5-2B 8k run it reports 86.4%, 90.5%,
        # 96.9% and 100.9% of the per-token delta norm — above 100% meaning the
        # int8 reconstruction differs from the fp16 one by more than the delta it
        # is approximating. Skip blocks had this corrected and passed; ordinary
        # blocks did not, which is consistent with depth 0.0 (uncompressed, below
        # short_context_threshold) and skip-rule blocks recalling the needle while
        # ordinary compressed blocks return it with the digits wrong.
        #
        # Costs one int8 quantise + [T, wr] @ [wr, feat] matmul per block at
        # prefill — the same work already paid for skip blocks, now for all.
        if pool is not None and T_active > 0:
            _wr = min(U_masked.shape[2], pool.U.shape[2])          # write_rank
            _U_w = U_masked[i, :T_active, :_wr]                    # [T, wr] fp16, as written
            _V_w = V_masked[i, :_wr, :]                            # [wr, feat] fp16, as written
            _max_abs = _U_w.detach().abs().amax()
            # fp16-rounded scale, matching pool store + decode dequant exactly.
            _scale_u = torch.clamp(_max_abs / 127.0, min=1e-5).to(torch.float16).float()
            _U_q = torch.clamp(torch.round(_U_w.float() / _scale_u), -127, 127)
            _recon_exact = (_U_q * _scale_u) @ _V_w.float()       # int8-dequant recon
            recon_K_for_res = _recon_exact[:, :half_d]
            recon_V_for_res = _recon_exact[:, half_d:]
            if os.environ.get("DKV_TELEMETRY", "0") == "1":
                _dn = delta_K.norm(dim=1).clamp(min=1e-6)
                _absorbed = ((recon_K_for_res - recon_K).norm(dim=1) / _dn).max().item()
                print(f"[DKV DEBUG] int8-exact residual for skip block "
                      f"anchor={getattr(block, 'anchor_idx', '?')}: absorbed up to "
                      f"{_absorbed*100:.1f}% per-token int8-U error", flush=True)

        # Residual capacity the pool will actually keep for this block; anything
        # selected beyond it is silently dropped by write_blocks_batched.
        _res_cap = int(_pool_max_res) if _pool_max_res else T_active

        if T_active > 0 and (n_max_residual > 0 or _force_exact):
            if _force_exact and T_active <= _res_cap:
                # Every token FITS in the pool's residual budget, so store them
                # all: no ranking decision to make, nothing gets truncated.
                fact_positions_K = torch.arange(T_active, device=rel_error_K.device)
                fact_positions_V = torch.arange(T_active, device=rel_error_V.device)
            else:
                # T_active EXCEEDS the budget (or this is an ordinary block), so
                # the choice of WHICH tokens get
                # an exact residual is a real decision -- and this used to make it
                # by position:
                #
                #     fact_positions_K = torch.arange(T_active)   # 0,1,2,...,255
                #
                # write_blocks_batched then keeps only the leading slice
                # (native_block_pool.py:684, `res_K_positions[:, :mr]`), so with
                # T_active=256 and max_residual=128 a skip block got exact
                # residuals for block-local offsets 0-127 and NOTHING for 128-255.
                # Those tokens fell back to the rank-32 low-rank reconstruction --
                # in a block that was flagged skip_compression precisely because it
                # holds digits the SVD must not smear.
                #
                # The old comment defended this as "earliest positions, where the
                # exact content sits". That is an assumption about where in a
                # 257-token block a code happens to land, and it is false half the
                # time. Whether a needle survived came down to
                # needle_position % 257 < 128, which is why recall has looked
                # erratic rather than monotone in depth, and why codes come back
                # TRUNCATED at a token boundary ('ZEBRA-4471-QUARTZ' ->
                # 'ZEBRA-447', -> 'ZEBRA-47-QUARTZ') -- the signature of a
                # multi-token code straddling offset 127, exact before it and
                # low-rank after.
                #
                # MLX has no positional path at all. It ALWAYS ranks:
                #     top_k = argsort(capture_scores)[:, -max_residual:][:, ::-1]
                # (mlx_dkv_wrapper.py:2852) with is_core tokens -- digits,
                # all-caps runs, '-', '_' (:2711) -- boosted so they win that
                # ranking. So the MLX-parity behaviour when the budget binds is to
                # RANK, not to slice. Fall through to the shared ranked selection
                # below with the full budget: it scores by the same joint absolute
                # V-balanced error and already applies _boost_vec, which is where
                # CUDA's equivalent of is_core lives.
                if _force_exact:
                    # A skip block keeps the FULL budget: the adaptive median
                    # tiering above may have capped it at 8 or 16, which is a
                    # policy for easy prose blocks, not for one flagged as
                    # holding verbatim content.
                    n_max_residual = _res_cap

                # ── MLX-parity residual selection ───────────────────────────
                # MLX ranks ONE joint score and takes ONE index set:
                #     errors_v_balanced = errors_v * v_gain
                #     joint_errors      = sqrt(errors_k**2 + errors_v_balanced**2)
                #     top_k_indices     = argsort(joint_errors)[-n_res:]
                # (mlx_dkv_wrapper.py:2630 and :3853, boost applied to
                # joint_errors at :2794.)
                #
                # This side ranked rel_error_K and rel_error_V SEPARATELY and kept
                # two different index sets. Two independent consequences, both
                # wrong, and both invisible in the 2k tests:
                #
                #  1. SPLIT SETS. A token could be selected for K and not for V.
                #     Its score then becomes exact while the value attended stays
                #     the lossy low-rank estimate — attention lands on the right
                #     token and reads the wrong content. That is the observed
                #     failure shape exactly: ZEBRA-447 / ZEBRA-474-QUARTZ, the
                #     needle located but its digits wrong. (compress_lowrank
                #     already forced fact_positions_V = fact_positions_K, but only
                #     under exact-keys, so the default CUDA path kept the split.)
                #
                #  2. RELATIVE vs ABSOLUTE. Dividing by each token's own norm
                #     ranks by *fractional* error, so a low-magnitude token with a
                #     small absolute error outranks a high-magnitude needle with a
                #     large one. Attention error is absolute — q·k does not care
                #     what fraction of the key was lost — so MLX ranks absolute.
                #     The relative form spends the budget on tokens that barely
                #     move the logits.
                #
                # rel_error_* stays as-is above: MLX uses it for the median tier
                # (:3822) and this path mirrors that, unchanged.
                _err_V_bal = error_V
                if _v_gain is not None:
                    _err_V_bal = error_V * _v_gain[i].to(error_V.dtype)
                joint_err = torch.sqrt(error_K.float() ** 2 + _err_V_bal.float() ** 2)
                if _boost_vec is not None:
                    joint_err = joint_err * _boost_vec.to(joint_err.dtype)

                top_k_J = _topk_with_coverage(joint_err, n_max_residual, _cov_frac_batch)

                _err_thr = _residual_error_threshold()   # MLX: 0.0, see resolver
                # A token qualifies if EITHER half is non-degenerate; with one
                # shared index set, dropping it on K alone would strand its V.
                _idx = top_k_J.indices
                mask_J = (top_k_J.values > _err_thr) & (
                    (error_K[_idx] > 1e-4) | (error_V[_idx] > 1e-4))
                fact_positions_K = _idx[mask_J]
                fact_positions_V = fact_positions_K

            if fact_positions_K.numel() > 0:
                # DKV_RESIDUAL_EXACT_KEYS — must match compress_lowrank's form
                # EXACTLY (see the long comment there). The decode kernel decides
                # add-vs-REPLACE from the same env var, so if this path kept the
                # correction form while that one stored the anchor-relative
                # delta, blocks compressed here would have their scores REPLACED
                # by a wrong value -- strictly worse than not enabling the mode
                # at all. That mixed-semantics state is exactly what regressed
                # when only compress_lowrank was converted.
                if _exact_keys_enabled(gpu_device):
                    residual_K_vals = delta_K[fact_positions_K].to(torch.float16).to(gpu_device)
                    # Single index set for both halves, as MLX does — see the
                    # matching note in compress_lowrank.
                    fact_positions_V = fact_positions_K
                else:
                    residual_K_vals = (delta_K - recon_K_for_res)[fact_positions_K].to(torch.float16).to(gpu_device)
                fact_positions_K = fact_positions_K.to(torch.int16).to(gpu_device)
            else:
                fact_positions_K = None
                residual_K_vals = None
                if _exact_keys_enabled(gpu_device):
                    fact_positions_V = fact_positions_V[:0]

            if fact_positions_V.numel() > 0:
                if _exact_keys_enabled(gpu_device):
                    residual_V_vals = delta_V[fact_positions_V].to(torch.float16).to(gpu_device)
                else:
                    residual_V_vals = (delta_V - recon_V_for_res)[fact_positions_V].to(torch.float16).to(gpu_device)
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

        # ── DKV_ROUTE_TRACE — is the ANCHOR ITSELF different this generation? ──
        #
        # The route trace showed the same logical block (same layer, same
        # anchor_idx), resolving to the SAME slot, reading back a different
        # |anchors_K| on repeat 2 of an identical prompt. tests/
        # test_pool_recycle_aliasing.py then proved the pool stores faithfully:
        # allocate/free/grow and both write paths (including batched writes into
        # LIFO-recycled non-monotonic slots) all keep each block's bytes in its
        # own slot. So storage is innocent and the value HANDED to it must
        # already differ.
        #
        # This is the last place that value exists before it becomes pool bytes.
        # Keyed on (layer_idx, anchor_idx) -- anchor_idx alone repeats across
        # layers and would compare unrelated blocks. Reports only MISMATCHES,
        # so a clean run is silent and a divergent one names the exact block.
        #
        # If this fires, the anchor is being COMPUTED differently on the second
        # generation and the defect is in prefill, not in the KV store. If it
        # stays silent while the router still reads a changed |anc|, the change
        # happens between this write and that read.
        if os.environ.get("DKV_ROUTE_TRACE", "0") == "1":
            global _ANCHOR_FP, _ANCHOR_FP_SHOWN, _ANCHOR_FP_CASE
            try:
                # THE CASE MUST BE PART OF THE KEY. The first version keyed only
                # (layer_idx, anchor_idx) and compared GLOBALLY across the whole
                # suite, so 2k@depth0.0's block at (layer 3, anchor 257) was
                # compared against 2k@depth0.5's -- a DIFFERENT PROMPT, which of
                # course has different keys at the same position. Every line it
                # printed was a false positive, and the line cap was exhausted
                # inside that burst so it never reached the within-case repeats
                # it exists to compare.
                #
                # DKV_ROUTE_TRACE_TOKEN is set by the validator once per case and
                # is identical across that case's repeats, so it is exactly the
                # discriminator needed: same value => same prompt => the anchor
                # for a given (layer, anchor_idx) MUST be identical.
                _case = os.environ.get("DKV_ROUTE_TRACE_TOKEN", "?")
                if _case != _ANCHOR_FP_CASE:
                    _ANCHOR_FP_CASE = _case
                    _ANCHOR_FP = {}          # new prompt: nothing to compare to
                    _ANCHOR_FP_SHOWN = 0     # budget is per case, not per suite
                _ak = torch.stack([b.anchor_kv[0, 0] for b in blocks_list], dim=0)
                _nrm = _ak.float().flatten(1).norm(dim=1).tolist()
                for _b, _n in zip(blocks_list, _nrm):
                    _key = (getattr(_b, "layer_idx", None), int(_b.anchor_idx))
                    _prev = _ANCHOR_FP.get(_key)
                    if _prev is None:
                        _ANCHOR_FP[_key] = (_n, _b.pool_idx)
                    elif abs(_prev[0] - _n) > 1e-3 and _ANCHOR_FP_SHOWN < 16:
                        _ANCHOR_FP_SHOWN += 1
                        print(f"[DKV] ANCHOR FP MISMATCH case={_case} "
                              f"layer={_key[0]} anchor={_key[1]} "
                              f"|anc| {_prev[0]:.4f} -> {_n:.4f} "
                              f"slot {_prev[1]} -> {_b.pool_idx} "
                              f"(SAME prompt, recomputed differently "
                              f"BEFORE storage)", flush=True)
                        _ANCHOR_FP[_key] = (_n, _b.pool_idx)
            except Exception as _fe:                             # noqa: BLE001
                if _ANCHOR_FP_SHOWN < 1:
                    _ANCHOR_FP_SHOWN += 1
                    print(f"[DKV] ANCHOR FP failed: {_fe}", flush=True)

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

    # ── V-side rebalancing for the joint K|V SVD (MLX parity: DKV_V_SCALE) ──
    # Same reasoning/formula as compress_lowrank and _compress_layer_blocks_gpu_inner:
    # scale V up before the SVD only (per batch item), divide the gain back out
    # of the V half of Vh afterward so the returned factors reconstruct
    # original-space V — this function's contract (raw SVD factors, no residual
    # logic) is unchanged for callers. Default ON; DKV_V_SCALE=0 to disable.
    half_d = d // 2
    v_scale_on = os.environ.get("DKV_V_SCALE", "1") != "0"
    v_gain = None
    if v_scale_on:
        eK = (deltas_cpu[:, :, :half_d].float() ** 2).sum(dim=(1, 2))  # [B]
        eV = (deltas_cpu[:, :, half_d:].float() ** 2).sum(dim=(1, 2))  # [B]
        v_gain = torch.sqrt(eK / eV.clamp(min=1e-12)).clamp(1.0, 10000.0)  # [B]
        deltas_for_svd = torch.cat([
            deltas_cpu[:, :, :half_d],
            deltas_cpu[:, :, half_d:] * v_gain.view(B, 1, 1),
        ], dim=2)
    else:
        deltas_for_svd = deltas_cpu

    # Compute scale per batch item
    scale = deltas_for_svd.abs().view(B, -1).max(dim=-1).values  # [B]
    scale = torch.clamp(scale, min=1e-9)

    x = deltas_for_svd / scale.view(B, 1, 1)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Batched Randomized SVD
    svd_success = False
    U, S, Vh = None, None, None
    
    try:
        n_oversamples = 5
        n_iter = 2
        r_proj = min(rank + n_oversamples, n, d)
        
        # 1. Generate random Gaussian projection matrix
        Omega = _rsvd_omega(d, r_proj, dtype=torch.float32, device=x.device)
        
        # 2. Form sample matrix Y with power iterations
        # x @ Omega: [B, n, d] @ [d, r_proj] -> [B, n, r_proj]
        Y = torch.matmul(x, Omega)
        for _ in range(n_iter):
            Y = torch.matmul(x, torch.matmul(x.transpose(1, 2), Y))
            
        # 3. Orthogonalize Y
        Q, _ = torch.linalg.qr(Y, mode="reduced") # [B, n, r_proj]
        
        # 4. Project original matrix onto low-rank subspace Q
        B_mat = torch.matmul(Q.transpose(1, 2), x)

        # 5. Decompose the much smaller matrix B_mat (MLX/GPU-batched-path
        # parity: DKV_COMPRESS_GRAM_SVD). Default ON, same batched Gram-eigh
        # trick as _compress_layer_blocks_gpu_inner (A/B-validated there,
        # numerically equivalent to the exact SVD). Falls through on failure.
        _gram_ok = False
        if os.environ.get("DKV_COMPRESS_GRAM_SVD", "1") != "0":
            try:
                G = torch.matmul(B_mat, B_mat.transpose(1, 2))      # [B, r, r]
                evals, evecs = torch.linalg.eigh(G)                 # ascending
                evals = evals.flip(-1).clamp(min=0.0)               # descending S^2
                U_b = evecs.flip(-1)                                # [B, r, r]
                S = evals.sqrt()                                    # [B, r] desc
                Vh = torch.matmul(U_b.transpose(1, 2), B_mat) / S.clamp(min=1e-8).unsqueeze(-1)
                _gram_ok = True
            except Exception as _ge:
                print(f"[DKV rSVD] Gram eigh path failed ({_ge}); using exact SVD.")
        if not _gram_ok:
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

    # Undo the V-side rebalancing on the FACTOR: divide Vh's V columns by the
    # same per-batch-item gain so downstream reconstruction (U @ Vh) is in
    # original space, matching what a caller that never knew about v_gain expects.
    if v_gain is not None:
        vg = v_gain.view(B, 1, 1).to(Vh.dtype)
        Vh = torch.cat([Vh[:, :, :half_d], Vh[:, :, half_d:] / vg], dim=2)

    return U, S, Vh, scale
