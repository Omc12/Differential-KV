"""
native_core/sparse_decode/triton_fused_decode.py

Triton-optimized fused reconstruction & sparse attention kernels for Differential KV.
Provides maximum memory bandwidth efficiency for DeltaKV = U @ V.T + anchor.
Falls back to pure-PyTorch on any system where Triton is unavailable.

Mac/MPS: Triton is CUDA-only; the PyTorch fallback is always used on Apple Silicon.
"""

import torch
import math
import os
import threading
from collections import OrderedDict
from typing import Optional, Tuple, List, Any

try:
    from native_core.mac_utils import nvtx_push as _nvtx_push, nvtx_pop as _nvtx_pop, has_cuda as _has_cuda
except ImportError:
    def _nvtx_push(label, device=None): pass
    def _nvtx_pop(device=None): pass
    def _has_cuda(): return torch.cuda.is_available()

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ── 1. Fused Triton Kernels ───────────────────────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def diffkv_fused_decode_kernel(
        # Inputs already in pool (indexed by slot)
        Q_ptr,           # [num_q_heads, head_dim]
        U_ptr,           # [pool_size, MAX_S, RANK]          INT8
        U_scale_ptr,     # [pool_size]                        FP16
        VK_ptr,          # [pool_size, RANK, head_dim_kv]    INT8 (after V quant)
        VK_scale_ptr,    # [pool_size, RANK]                  FP16
        AncK_ptr,        # [pool_size, kv_heads, head_dim]   INT8
        AncK_scale_ptr,  # [pool_size]                        FP16
        VV_ptr,          # [pool_size, RANK, head_dim_kv]    INT8
        VV_scale_ptr,    # [pool_size, RANK]                  FP16
        AncV_ptr,        # [pool_size, kv_heads, head_dim]   INT8
        AncV_scale_ptr,  # [pool_size]                        FP16
        SlotIdx_ptr,     # [n_active]  which slots are live
        BlkSz_ptr,       # [n_active]  actual token count per block
        PartOut_ptr,     # [n_chunks, num_q_heads, head_dim] output accumulator
        PartLSE_ptr,     # [n_chunks, num_q_heads]            log-sum-exp
        n_active,
        scale,           # 1/sqrt(head_dim)
        q_per_kv,        # GQA ratio
        num_q_heads,
        HEAD_DIM:         tl.constexpr,
        RANK:             tl.constexpr,
        TILE_S:           tl.constexpr,     # tokens per inner tile
        BLOCKS_PER_CHUNK: tl.constexpr,     # KV blocks per thread block
        MAX_S:            tl.constexpr,
    ):
        q_head  = tl.program_id(0)
        chunk   = tl.program_id(1)
        kv_head = q_head // q_per_kv

        # ── Load Q into registers — never leaves registers ──────────────────
        q = tl.load(Q_ptr + q_head * HEAD_DIM + tl.arange(0, HEAD_DIM)).to(tl.float32)

        # ── Load V_K for this KV head into SRAM — shared across all blocks ──
        # KEY INSIGHT: V_K is loaded ONCE here and reused for ALL blocks.
        vk_base  = tl.load(VK_scale_ptr + tl.arange(0, RANK),   # [RANK] scales
                           mask=tl.arange(0, RANK) < RANK)
        # V_K dequant: inline in registers during q_proj computation
        vk_data  = tl.load(
            VK_ptr + kv_head * RANK * HEAD_DIM +
            tl.arange(0, RANK)[:, None] * HEAD_DIM +
            tl.arange(0, HEAD_DIM)[None, :]
        ).to(tl.float32) * vk_base[:, None]                        # [RANK, HEAD_DIM]

        # Same for V_V
        vv_base  = tl.load(VV_scale_ptr + tl.arange(0, RANK))
        vv_data  = tl.load(
            VV_ptr + kv_head * RANK * HEAD_DIM +
            tl.arange(0, RANK)[:, None] * HEAD_DIM +
            tl.arange(0, HEAD_DIM)[None, :]
        ).to(tl.float32) * vv_base[:, None]                        # [RANK, HEAD_DIM]

        # ── Q projection — computed once for this entire chunk ─────────────
        q_proj_k = tl.sum(q[None, :] * vk_data, axis=1) * scale   # [RANK]

        # ── Online softmax state in registers ──────────────────────────────
        m   = float('-inf')
        l   = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for b in range(BLOCKS_PER_CHUNK):
            b_abs = chunk * BLOCKS_PER_CHUNK + b
            if b_abs >= n_active: break

            slot     = tl.load(SlotIdx_ptr + b_abs)
            blk_sz   = tl.load(BlkSz_ptr   + b_abs)
            u_scale  = tl.load(U_scale_ptr  + slot).to(tl.float32)
            ak_scale = tl.load(AncK_scale_ptr + slot).to(tl.float32)

            # Anchor score (post-RoPE anchor, stored exactly at position)
            anc_k = tl.load(AncK_ptr + slot * tl.constexpr(2) * HEAD_DIM +
                            kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
                           ).to(tl.float32) * ak_scale
            s_anc = tl.sum(q * anc_k) * scale

            # Anchor value
            av_scale = tl.load(AncV_scale_ptr + slot).to(tl.float32)
            anc_v    = tl.load(AncV_ptr + slot * tl.constexpr(2) * HEAD_DIM +
                               kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
                              ).to(tl.float32) * av_scale

            # Online softmax: anchor
            m_new = tl.maximum(m, s_anc)
            l_new = l * tl.exp(m - m_new) + tl.exp(s_anc - m_new)
            acc   = acc * (l / l_new) * tl.exp(m - m_new) + \
                    anc_v * (tl.exp(s_anc - m_new) / l_new)
            m, l  = m_new, l_new

            # Delta tokens: Project-Then-Attend, tiled
            u_base = slot * MAX_S * RANK

            for t in range(0, MAX_S - 1, TILE_S):
                valid   = (tl.arange(0, TILE_S) + t) < (blk_sz - 1)
                u_tile  = tl.load(
                    U_ptr + u_base + (t + tl.arange(0, TILE_S))[:, None] * RANK +
                    tl.arange(0, RANK)[None, :],
                    mask=valid[:, None], other=0
                ).to(tl.float32) * u_scale                          # [TILE_S, RANK]

                d_scores = tl.sum(u_tile * q_proj_k[None, :], axis=1)
                d_scores = tl.where(valid, d_scores, float('-inf'))

                t_max  = tl.max(d_scores, axis=0)
                m_new  = tl.maximum(m, t_max)
                exp_d  = tl.exp(d_scores - m_new) * valid.to(tl.float32)
                l_new  = l * tl.exp(m - m_new) + tl.sum(exp_d)

                w_d    = exp_d / l_new
                w_proj = tl.sum(w_d[:, None] * u_tile, axis=0)
                v_c    = tl.sum(w_proj[:, None] * vv_data, axis=0)

                acc = acc * (l / l_new) * tl.exp(m - m_new) + v_c
                m, l = m_new, l_new

        out_off = (chunk * num_q_heads + q_head) * HEAD_DIM
        tl.store(PartOut_ptr + out_off + tl.arange(0, HEAD_DIM), acc)
        tl.store(PartLSE_ptr + chunk * num_q_heads + q_head,
                 m + tl.log(tl.where(l > 0.0, l, 1e-9)))

    # Working Triton kernel matching the actual NativeBlockPool layout
    @triton.jit
    def _fused_sparse_decode_kernel(
        q_ptr, block_indices_ptr, pool_ak_ptr, pool_av_ptr, pool_vk_ptr, pool_vv_ptr,
        pool_u_ptr, pool_u_scale_ptr, pool_scales_ptr, pool_seq_lens_ptr,
        out_ptr, m_ptr, l_ptr,
        stride_q_h, stride_q_d,
        stride_ak_n, stride_ak_h, stride_ak_d,
        stride_av_n, stride_av_h, stride_av_d,
        stride_vk_n, stride_vk_r, stride_vk_h, stride_vk_d,
        stride_vv_n, stride_vv_r, stride_vv_h, stride_vv_d,
        stride_u_n, stride_u_s, stride_u_r,
        stride_out_h, stride_out_d,
        N: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr, KV_GRP: tl.constexpr, D: tl.constexpr,
        R: tl.constexpr, S_MAX: tl.constexpr, INV_SCALE: tl.constexpr, BLOCKS_PER_CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr,
    ):
        h_q = tl.program_id(0)
        chunk_id = tl.program_id(1)
        h_kv = h_q // KV_GRP
        
        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, R)
        offs_s = tl.arange(0, S_MAX)
        
        q_ptrs = q_ptr + h_q * stride_q_h + offs_d * stride_q_d
        q = tl.load(q_ptrs).to(tl.float32)
        
        m_i = -float("inf")
        l_i = 0.0
        O_i = tl.zeros([D], dtype=tl.float32)
        
        start_block = chunk_id * BLOCKS_PER_CHUNK
        end_block = start_block + BLOCKS_PER_CHUNK
        if end_block > N:
            end_block = N
            
        for n in range(start_block, end_block):
            pool_idx = tl.load(block_indices_ptr + n)
            scale = tl.load(pool_scales_ptr + pool_idx).to(tl.float32)
            actual_s = tl.load(pool_seq_lens_ptr + pool_idx)
            
            ak_ptrs = pool_ak_ptr + pool_idx * stride_ak_n + h_kv * stride_ak_h + offs_d * stride_ak_d
            av_ptrs = pool_av_ptr + pool_idx * stride_av_n + h_kv * stride_av_h + offs_d * stride_av_d
            ak = tl.load(ak_ptrs).to(tl.float32)
            av = tl.load(av_ptrs).to(tl.float32)
            
            vk_ptrs = pool_vk_ptr + pool_idx * stride_vk_n + h_kv * stride_vk_h + offs_r[:, None] * stride_vk_r + offs_d[None, :] * stride_vk_d
            vv_ptrs = pool_vv_ptr + pool_idx * stride_vv_n + h_kv * stride_vv_h + offs_r[:, None] * stride_vv_r + offs_d[None, :] * stride_vv_d
            vk = tl.load(vk_ptrs).to(tl.float32)
            vv = tl.load(vv_ptrs).to(tl.float32)
            
            u_ptrs = pool_u_ptr + pool_idx * stride_u_n + offs_s[:, None] * stride_u_s + offs_r[None, :] * stride_u_r
            s_mask = offs_s[:, None] < actual_s
            u = tl.load(u_ptrs, mask=s_mask, other=0.0).to(tl.float32)
            
            u_scale_ptr = pool_u_scale_ptr + pool_idx
            u_scale = tl.load(u_scale_ptr)
            u = u * u_scale
            
            s_anchor = tl.sum(q * ak) * INV_SCALE
            q_proj = tl.sum(q[None, :] * vk, axis=1) * INV_SCALE
            delta_scores = tl.sum(u * q_proj[None, :], axis=1) * scale
            s = s_anchor + delta_scores
            
            s = tl.where(offs_s < actual_s, s, -float("inf"))
            m_b_delta = tl.max(s, axis=0)
            m_b = tl.maximum(s_anchor, m_b_delta)
            
            m_new = tl.maximum(m_i, m_b)
            alpha = tl.exp(m_i - m_new)
            p_anchor = tl.exp(s_anchor - m_new)
            p_delta = tl.exp(s - m_new)
            p_delta = tl.where(offs_s < actual_s, p_delta, 0.0)
            p_delta_sum = tl.sum(p_delta, axis=0)
            
            l_i = l_i * alpha + p_anchor + p_delta_sum
            
            p_u = tl.sum(p_delta[:, None] * u, axis=0)
            o_delta = tl.sum(p_u[:, None] * vv, axis=0) * scale
            
            O_i = O_i * alpha + (p_anchor + p_delta_sum) * av + o_delta
            m_i = m_new

        if NUM_CHUNKS == 1:
            O_i = O_i / l_i
            out_ptrs = out_ptr + h_q * stride_out_h + offs_d * stride_out_d
            tl.store(out_ptrs, O_i)
            if m_ptr is not None:
                tl.store(m_ptr + h_q, m_i)
            if l_ptr is not None:
                tl.store(l_ptr + h_q, l_i)
        else:
            out_work_ptrs = out_ptr + h_q * (NUM_CHUNKS * D) + chunk_id * D + offs_d
            tl.store(out_work_ptrs, O_i)
            if m_ptr is not None:
                tl.store(m_ptr + h_q * NUM_CHUNKS + chunk_id, m_i)
            if l_ptr is not None:
                tl.store(l_ptr + h_q * NUM_CHUNKS + chunk_id, l_i)

    @triton.jit
    def _fused_sparse_decode_reduction_kernel(
        out_workspace_ptr, m_workspace_ptr, l_workspace_ptr, out_ptr, m_final_ptr, l_final_ptr,
        NUM_CHUNKS: tl.constexpr, D: tl.constexpr,
    ):
        h_q = tl.program_id(0)
        offs_d = tl.arange(0, D)
        
        m_i = -float("inf")
        l_i = 0.0
        O_i = tl.zeros([D], dtype=tl.float32)
        
        for c in range(NUM_CHUNKS):
            m_c = tl.load(m_workspace_ptr + h_q * NUM_CHUNKS + c)
            l_c = tl.load(l_workspace_ptr + h_q * NUM_CHUNKS + c)
            out_c_ptrs = out_workspace_ptr + h_q * (NUM_CHUNKS * D) + c * D + offs_d
            O_c = tl.load(out_c_ptrs).to(tl.float32)
            
            m_new = tl.maximum(m_i, m_c)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_c - m_new)
            
            l_i = l_i * alpha + l_c * beta
            O_i = O_i * alpha + O_c * beta
            m_i = m_new
            
        O_i = O_i / l_i
        out_ptrs = out_ptr + h_q * D + offs_d
        tl.store(out_ptrs, O_i)
        if m_final_ptr is not None:
            tl.store(m_final_ptr + h_q, m_i)
        if l_final_ptr is not None:
            tl.store(l_final_ptr + h_q, l_i)

    @triton.jit
    def lowrank_recon_kernel(
        U_ptr, V_ptr, anchor_ptr, out_ptr,
        stride_un, stride_uk,
        stride_vk, stride_vd,
        stride_ad,
        stride_on, stride_od,
        n_tokens, rank, feat_dim, scale,
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)

        mask_n = offs_n < n_tokens
        mask_d = offs_d < feat_dim

        anchor = tl.load(anchor_ptr + offs_d, mask=mask_d, other=0.0)
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=tl.float32)

        for k_start in range(0, rank, BLOCK_SIZE_K):
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
            mask_k = offs_k < rank

            u = tl.load(
                U_ptr + offs_n[:, None] * stride_un + offs_k[None, :] * stride_uk,
                mask=mask_n[:, None] & mask_k[None, :], other=0.0,
            )
            v = tl.load(
                V_ptr + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=mask_k[:, None] & mask_d[None, :], other=0.0,
            )
            acc += tl.dot(u, v)

        if scale != 1.0:
            acc *= scale

        acc += anchor[None, :]
        out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc, mask=mask_n[:, None] & mask_d[None, :])


# ── 2. PyTorch JIT Helpers for Compilation ─────────────────────────────────────

def _reconstruct_and_score_compiled(
    U: torch.Tensor,
    V_K: torch.Tensor,
    anchors_K: torch.Tensor,
    scales: torch.Tensor,
    cos_sliced: torch.Tensor,
    sin_sliced: torch.Tensor,
    q_sq: torch.Tensor,
    inv_scale: float,
) -> torch.Tensor:
    N = U.shape[0]
    S = U.shape[1]
    R = U.shape[2]
    H = q_sq.shape[0]
    D = q_sq.shape[1]
    
    deltas_k_flat = torch.bmm(U.float(), V_K.float().reshape(N, R, H * D))
    deltas_k = deltas_k_flat.reshape(N, S, H, D).to(U.dtype) * scales.unsqueeze(-1)
    
    zeros_pad = torch.zeros((N, 1, H, D), dtype=U.dtype, device=U.device)
    deltas_k_full = torch.cat([zeros_pad, deltas_k], dim=1)
    K_unrot_full = anchors_K.unsqueeze(1) + deltas_k_full
    
    half_d = D // 2
    K_unrot_half1 = K_unrot_full[..., :half_d]
    K_unrot_half2 = K_unrot_full[..., half_d:]
    K_unrot_rotated = torch.cat([-K_unrot_half2, K_unrot_half1], dim=-1)
    K_rot_full = K_unrot_full * cos_sliced + K_unrot_rotated * sin_sliced
    
    q_expanded = q_sq.view(1, 1, H, D)
    scores = torch.sum(q_expanded * K_rot_full, dim=-1) * inv_scale
    return scores

def _attend_and_reconstruct_v_compiled(
    P_anchor: torch.Tensor,
    P_comp: torch.Tensor,
    P_dense: torch.Tensor,
    U: torch.Tensor,
    V_V: torch.Tensor,
    anchors_V: torch.Tensor,
    scales: torch.Tensor,
    v_dense_rep: torch.Tensor,
    H_q: int,
    N: int,
    block_capacity: int,
    R: int,
    D: int,
    S_dense: int,
    V_V_perm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    O_final = torch.zeros((H_q, D), device=P_anchor.device, dtype=P_anchor.dtype)
    if N > 0:
        P_comp_reshaped = P_comp.view(H_q, N, block_capacity).permute(1, 0, 2)
        P_U = torch.bmm(P_comp_reshaped.float(), U.float())

        p_total_anchor = P_anchor.transpose(0, 1) + P_comp_reshaped.sum(dim=-1)
        O_anchor_fused = torch.sum(p_total_anchor.unsqueeze(-1) * anchors_V.float(), dim=0)
        O_final = O_final + O_anchor_fused.to(P_anchor.dtype)

        P_U_flat = P_U.reshape(N * H_q, 1, R)
        if V_V_perm is None:
            V_V_perm = V_V.float().permute(0, 2, 1, 3).contiguous().reshape(N * H_q, R, D)
        O_delta = torch.bmm(P_U_flat, V_V_perm).reshape(N, H_q, D) * scales.float()
        O_final = O_final + O_delta.sum(0).to(P_anchor.dtype)

    if S_dense > 0:
        O_dense_total = torch.sum(P_dense.unsqueeze(-1) * v_dense_rep.squeeze(0), dim=1)
        O_final = O_final + O_dense_total.to(P_anchor.dtype)

    return O_final

_IS_MPS_AVAILABLE = (hasattr(torch, "backends") and
                     hasattr(torch.backends, "mps") and
                     torch.backends.mps.is_available())
_IS_CUDA_AVAILABLE = torch.cuda.is_available()

use_compile = os.environ.get("DIFFKV_USE_TORCH_COMPILE", "auto")
if use_compile == "auto":
    use_compile = "1" if _IS_CUDA_AVAILABLE else "0"
elif _IS_MPS_AVAILABLE and not _IS_CUDA_AVAILABLE:
    use_compile = "0"

if use_compile == "1":
    try:
        _backend = "inductor"
        _mode = "reduce-overhead" if _IS_CUDA_AVAILABLE else "default"
        print(f"[DiffKV JIT] Compiling _reconstruct_and_score with backend={_backend}, mode={_mode} (dynamic=True) ...")
        _reconstruct_and_score = torch.compile(
            _reconstruct_and_score_compiled,
            backend=_backend,
            mode=_mode,
            fullgraph=False,
            dynamic=True,
        )
    except Exception as e:
        print(f"[DiffKV JIT] torch.compile of _reconstruct_and_score failed ({e}). Falling back to eager.")
        _reconstruct_and_score = _reconstruct_and_score_compiled
        
    try:
        _backend = "inductor"
        _mode = "reduce-overhead" if _IS_CUDA_AVAILABLE else "default"
        print(f"[DiffKV JIT] Compiling _attend_and_reconstruct_v with backend={_backend}, mode={_mode} (dynamic=True) ...")
        _attend_and_reconstruct_v = torch.compile(
            _attend_and_reconstruct_v_compiled,
            backend=_backend,
            mode=_mode,
            fullgraph=False,
            dynamic=True,
        )
    except Exception as e:
        print(f"[DiffKV JIT] torch.compile of _attend_and_reconstruct_v failed ({e}). Falling back to eager.")
        _attend_and_reconstruct_v = _attend_and_reconstruct_v_compiled
else:
    _reconstruct_and_score = _reconstruct_and_score_compiled
    _attend_and_reconstruct_v = _attend_and_reconstruct_v_compiled


@torch.jit.script
def _prefill_fused_history_attend(
    U: torch.Tensor,
    V_K: torch.Tensor,
    V_V: torch.Tensor,
    anchors_K: torch.Tensor,
    anchors_V: torch.Tensor,
    scales: torch.Tensor,
    cos_sliced: torch.Tensor,
    sin_sliced: torch.Tensor,
    q: torch.Tensor,
    seq_lens: torch.Tensor,
    inv_scale: float,
) -> torch.Tensor:
    N = U.shape[0]
    S = U.shape[1]
    R = U.shape[2]
    H = q.shape[1]
    Q = q.shape[2]
    D = q.shape[3]

    V_K_flat = V_K.float().reshape(N, R, H * D)
    deltas_k_flat = torch.bmm(U.float(), V_K_flat)
    deltas_k = (deltas_k_flat.reshape(N, S, H, D)
                * scales.view(N, 1, 1, 1).float()).to(q.dtype)

    K_unrot_full = torch.cat(
        [anchors_K.unsqueeze(1), anchors_K.unsqueeze(1) + deltas_k], dim=1
    )

    half_d = D // 2
    K_half1 = K_unrot_full[..., :half_d]
    K_half2 = K_unrot_full[..., half_d:]
    K_rotated = torch.cat([-K_half2, K_half1], dim=-1)
    cos_s = cos_sliced.squeeze(2)
    sin_s = sin_sliced.squeeze(2)
    K_rot_full = (K_unrot_full * cos_s.unsqueeze(2)
                  + K_rotated  * sin_s.unsqueeze(2))

    q_hqd = q.float().squeeze(0).reshape(H, Q, D)
    K_hnd = K_rot_full.float().permute(2, 0, 1, 3).reshape(H, N * (1 + S), D)
    scores_flat = torch.bmm(q_hqd, K_hnd.transpose(1, 2)) * inv_scale
    scores = scores_flat.reshape(H, Q, N, 1 + S)

    col = torch.arange(1 + S, device=U.device, dtype=torch.long).unsqueeze(0)
    valid = col <= seq_lens.unsqueeze(1).long()
    scores = scores.masked_fill(
        (~valid).unsqueeze(0).unsqueeze(0), float('-inf')
    )

    scores_f = scores.reshape(H, Q, N * (1 + S))
    lse_hist  = torch.logsumexp(scores_f, dim=-1)
    weights_f = torch.softmax(scores_f, dim=-1)
    weights   = weights_f.reshape(H, Q, N, 1 + S)

    w_anchor = weights[:, :, :, 0]
    w_delta  = weights[:, :, :, 1:]
    p_total  = w_anchor + w_delta.sum(dim=-1)

    anc_v_hnd = anchors_V.permute(1, 0, 2).float()
    out_anchor = torch.bmm(p_total.float(), anc_v_hnd)

    w_delta_perm = w_delta.float().permute(2, 0, 1, 3)
    w_delta_flat = w_delta_perm.reshape(N, H * Q, S)
    W_proj_flat = torch.bmm(w_delta_flat, U.float())
    W_proj_flat = W_proj_flat * scales.float().view(N, 1, 1)
    W_proj = W_proj_flat.reshape(N, H, Q, R).permute(1, 2, 0, 3)

    V_V_t  = V_V.float().permute(2, 0, 1, 3)
    W_proj_flat2 = W_proj.reshape(H, Q, N * R)
    V_V_t_flat2 = V_V_t.contiguous().reshape(H, N * R, D)
    out_delta = torch.bmm(W_proj_flat2, V_V_t_flat2)

    out_hist = (out_anchor + out_delta).to(q.dtype).unsqueeze(0)
    lse_out  = lse_hist.to(q.dtype).unsqueeze(0)

    lse_padded = lse_out.unsqueeze(-1).expand(1, H, Q, D)
    return torch.stack([out_hist, lse_padded], dim=0)


# ── 3. PyTorch fallbacks / MPS decoders ───────────────────────────────────────

def fused_decode_attention_mps(
    Q:        torch.Tensor,
    U:        torch.Tensor,
    U_scale:  torch.Tensor,
    VK:       torch.Tensor,
    VV:       torch.Tensor,
    AncK:     torch.Tensor,
    AncV:     torch.Tensor,
    slot_idx: torch.Tensor,
    blk_sizes: torch.Tensor,
) -> torch.Tensor:
    N  = slot_idx.shape[0]
    if N == 0:
        return torch.zeros(Q.shape, dtype=Q.dtype, device=Q.device)

    H_q, D  = Q.shape
    H_kv    = VK.shape[0]
    gpk     = H_q // H_kv
    scale   = D ** -0.5
    q       = Q.float()

    U_a     = U[slot_idx].float() * U_scale[slot_idx].view(N, 1, 1).float()
    AncK_a  = AncK[slot_idx].float()
    AncV_a  = AncV[slot_idx].float()

    AncK_e  = AncK_a.repeat_interleave(gpk, dim=1)
    AncV_e  = AncV_a.repeat_interleave(gpk, dim=1)
    VK_e    = VK.float().repeat_interleave(gpk, dim=0)
    VV_e    = VV.float().repeat_interleave(gpk, dim=0)

    score_anc = torch.einsum('hd,nhd->hn', q, AncK_e) * scale
    q_proj    = torch.einsum('hd,hrd->hr', q, VK_e) * scale
    delta_s   = torch.einsum('hr,nsr->hns', q_proj, U_a)

    s_range   = torch.arange(U_a.shape[1], device=Q.device).view(1, 1, -1)
    valid_mask = s_range < blk_sizes.view(1, N, 1).long()
    delta_s    = delta_s.masked_fill(~valid_mask, float('-inf'))

    all_scores = torch.cat(
        [score_anc.unsqueeze(-1), delta_s], dim=-1
    ).reshape(H_q, -1)

    w = torch.softmax(all_scores, dim=-1).reshape(H_q, N, 1 + U_a.shape[1])
    w_anc = w[:, :, 0]
    w_d   = w[:, :, 1:]

    out_anc  = torch.einsum('hn,nhd->hd', w_anc, AncV_e)
    w_proj   = torch.einsum('hns,nsr->hr', w_d, U_a)
    out_d    = torch.einsum('hr,hrd->hd', w_proj, VV_e)

    return (out_anc + out_d).to(Q.dtype)


def fused_decode_mps(
    Q:                    torch.Tensor,
    pool:                 object,
    block_indices:        Optional[torch.Tensor],
    blk_sizes:            Optional[torch.Tensor],
    num_key_value_groups: int,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    H_q, D   = Q.shape
    gpk      = num_key_value_groups
    scale    = D ** -0.5
    q        = Q.float()

    if block_indices is None or block_indices.numel() == 0:
        return torch.zeros((H_q, D), dtype=Q.dtype, device=Q.device), torch.full((H_q,), float('-inf'), dtype=Q.dtype, device=Q.device)

    N   = block_indices.shape[0]
    idx = block_indices.long()

    U_a    = pool.U[idx].float() * pool.U_scale[idx].view(N, 1, 1).float()
    AncK_a = pool.anchors_K[idx].float()
    AncV_a = pool.anchors_V[idx].float()
    VK_a   = pool.V_K[idx].float()
    VV_a   = pool.V_V[idx].float()

    if anchor_indices is not None and cos is not None and sin is not None:
        cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
        sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
        
        cos_anc = cos_flat[anchor_indices].to(device=VK_a.device, dtype=VK_a.dtype).unsqueeze(1).unsqueeze(2)
        sin_anc = sin_flat[anchor_indices].to(device=VK_a.device, dtype=VK_a.dtype).unsqueeze(1).unsqueeze(2)
        
        cos_anc_2d = cos_flat[anchor_indices].to(device=AncK_a.device, dtype=AncK_a.dtype).unsqueeze(1)
        sin_anc_2d = sin_flat[anchor_indices].to(device=AncK_a.device, dtype=AncK_a.dtype).unsqueeze(1)
        
        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)
            
        VK_a = VK_a * cos_anc + rotate_half(VK_a) * sin_anc
        AncK_a = AncK_a * cos_anc_2d + rotate_half(AncK_a) * sin_anc_2d

    S_comp = U_a.shape[1]
    R      = U_a.shape[2]

    AncK_e = AncK_a.repeat_interleave(gpk, dim=1)
    AncV_e = AncV_a.repeat_interleave(gpk, dim=1)
    VK_e   = VK_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
    VV_e   = VV_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()

    s_anc = torch.einsum('hd,nhd->hn', q, AncK_e) * scale
    q_proj_n = torch.einsum('hd,nhrd->nhr', q, VK_e) * scale
    delta_s = torch.einsum('nhr,nsr->hns', q_proj_n, U_a)
    delta_s = delta_s * pool.scales[idx].float().view(1, N, 1)
    delta_s = delta_s + s_anc.unsqueeze(-1)

    if blk_sizes is not None:
        s_range   = torch.arange(S_comp, device=Q.device).view(1, 1, -1)
        valid_msk = s_range < blk_sizes.view(1, N, 1).long()
        delta_s   = delta_s.masked_fill(~valid_msk, float('-inf'))

    scores = torch.cat(
        [s_anc.unsqueeze(-1), delta_s], dim=-1
    ).reshape(H_q, N * (1 + S_comp))

    lse = torch.logsumexp(scores, dim=-1)
    w = torch.softmax(scores, dim=-1)

    W_comp = w.reshape(H_q, N, 1 + S_comp)
    w_anc  = W_comp[:, :, 0]
    w_d    = W_comp[:, :, 1:]

    O = torch.zeros((H_q, D), device=Q.device, dtype=torch.float32)
    O = O + torch.einsum('hn,nhd->hd', w_anc, AncV_e)

    w_proj = torch.einsum('hns,nsr->nhr', w_d, U_a)
    w_proj = w_proj * pool.scales[idx].float().view(N, 1, 1)
    O = O + torch.einsum('nhr,nhrd->hd', w_proj, VV_e)

    return O.to(Q.dtype), lse.to(Q.dtype)


def _pytorch_vectorized_sparse_attn_decode(
    q:                    torch.Tensor,
    block_indices:        torch.Tensor,
    pool:                 object,
    dense_blocks:         list,            
    active_k:             torch.Tensor,
    active_v:             torch.Tensor,
    num_key_value_groups: int,
    R:                    int = 16,
    S_MAX:                int = 64,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
    total_seq_len:        int = 0,
    max_valid_len:        Optional[int] = None,
    cos_sliced:           Optional[torch.Tensor] = None,
    sin_sliced:           Optional[torch.Tensor] = None,
    session_id:           Optional[str] = None,
    layer_idx:            Optional[int] = None,
    decode_workspace:     Optional[dict] = None,
) -> torch.Tensor:
    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1
    inv_scale = 1.0 / math.sqrt(D)
    q_sq = q.view(H_q, D)
    
    def repeat_kv_at_dim(t, n_rep, dim):
        if n_rep == 1:
            return t
        if dim < 0:
            dim = t.dim() + dim
        shape = list(t.shape)
        val = shape[dim]
        t = t.unsqueeze(dim + 1)
        expand_shape = list(t.shape)
        expand_shape[dim + 1] = n_rep
        t = t.expand(*expand_shape)
        new_shape = shape[:dim] + [val * n_rep] + shape[dim + 1:]
        return t.reshape(*new_shape)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    N = block_indices.shape[0] if block_indices is not None else 0
    block_capacity = 0
    diagnostics = (os.environ.get("DIFFKV_DIAGNOSTICS", "0") == "1")

    U = torch.empty((0,), device=q.device, dtype=q.dtype)
    V_K = torch.empty((0,), device=q.device, dtype=q.dtype)
    V_V = torch.empty((0,), device=q.device, dtype=q.dtype)
    anchors_K = torch.empty((0,), device=q.device, dtype=q.dtype)
    anchors_V = torch.empty((0,), device=q.device, dtype=q.dtype)
    scales = torch.empty((0,), device=q.device, dtype=q.dtype)
    seq_lens_t = torch.empty((0,), device=q.device, dtype=torch.int32)

    # ── Check configuration-driven caching limits ───────────────────────
    config = getattr(pool, "config", None)
    decode_cache_enabled = config.decode_cache_enabled if config is not None else True
    decode_cache_max_tokens = config.decode_cache_max_tokens if config is not None else 4096

    use_workspace_cache = decode_cache_enabled
    if decode_cache_max_tokens > 0 and total_seq_len > decode_cache_max_tokens:
        use_workspace_cache = False

    session_dict = None
    if use_workspace_cache and decode_workspace is not None and session_id is not None:
        session_dict = decode_workspace.setdefault(session_id, {})
    elif decode_workspace is not None and session_id is not None:
        session_dict = decode_workspace.get(session_id)

    if not use_workspace_cache and decode_workspace is not None and session_id is not None and layer_idx is not None:
        # Clear existing cached tensors in O(1) immediately to reclaim VRAM/RAM
        if session_dict is not None:
            session_dict.get("gathered_kv", {}).pop(layer_idx, None)
            is_empty = True
            for val in session_dict.values():
                if isinstance(val, dict) and len(val) > 0:
                    is_empty = False
                    break
                elif not isinstance(val, dict) and val is not None:
                    is_empty = False
                    break
            if is_empty:
                decode_workspace.pop(session_id, None)

    if N > 0:
        current_version = session_dict.get("routing_version", 0) if session_dict is not None else 0

        cached_gathered = None
        gathered_cache = None
        if use_workspace_cache and session_dict is not None and layer_idx is not None:
            gathered_cache = session_dict.setdefault("gathered_kv", {})
            cached_val = gathered_cache.get(layer_idx)
            if cached_val is not None and cached_val[0] == current_version:
                cached_gathered = cached_val[1]

        if cached_gathered is not None:
            U, V_K, V_V, anchors_K, anchors_V, scales, seq_lens_t = cached_gathered
        else:
            indices = block_indices.long()
            U = pool.U[indices].to(q.dtype) * pool.U_scale[indices].view(-1, 1, 1)
            V_K = repeat_kv_at_dim(pool.V_K[indices], num_key_value_groups, dim=2)
            V_V = repeat_kv_at_dim(pool.V_V[indices], num_key_value_groups, dim=2)
            anchors_K = repeat_kv_at_dim(pool.anchors_K[indices], num_key_value_groups, dim=1)
            anchors_V = repeat_kv_at_dim(pool.anchors_V[indices], num_key_value_groups, dim=1)
            scales = pool.scales[indices].view(N, 1, 1)
            seq_lens_t = pool.seq_lens[indices]
            
            if use_workspace_cache and gathered_cache is not None:
                gathered_cache[layer_idx] = (current_version, (U, V_K, V_V, anchors_K, anchors_V, scales, seq_lens_t))
        
        block_capacity = U.shape[1]

        if max_valid_len is None:
            max_valid_len = int(seq_lens_t.max().item())

        # ── Project-Then-Attend formulation (zero-reconstruction, zero dense VRAM) ──
        # exact anchor score: [H_q, N]
        scores_anchor = torch.einsum('hd,nhd->hn', q_sq, anchors_K) * inv_scale
        
        # Project query to V_K: [N, H_q, R]
        q_proj = torch.einsum('hd,nrhd->nhr', q_sq, V_K) * inv_scale
        
        # Inner product with U: [H_q, N, block_capacity]
        scores_block = torch.einsum('nhr,nsr->hns', q_proj, U) * scales.view(1, N, 1)
        scores_block = scores_block + scores_anchor.unsqueeze(-1)
        
        # Mask out-of-bounds tokens
        s_range = torch.arange(block_capacity, device=q.device).view(1, 1, -1)
        valid_mask = s_range < seq_lens_t.view(1, N, 1)
        scores_block = scores_block.masked_fill(~valid_mask, float('-inf'))
        
        scores_compressed = scores_block.reshape(H_q, N * block_capacity)
    else:
        scores_anchor = torch.empty((H_q, 0), device=q.device, dtype=q.dtype)
        scores_compressed = torch.empty((H_q, 0), device=q.device, dtype=q.dtype)

    dense_k_parts = []
    dense_v_parts = []
    
    if active_k is not None and active_k.shape[2] > 0:
        dense_k_parts.append(active_k)
        dense_v_parts.append(active_v)
    else:
        for blk in (dense_blocks or []):
            dense_k_parts.append(blk.anchor_kv[:, 0].unsqueeze(2))
            dense_v_parts.append(blk.anchor_kv[:, 1].unsqueeze(2))
            if blk.active_k is not None:
                dense_k_parts.append(blk.active_k)
                dense_v_parts.append(blk.active_v)

    if dense_k_parts:
        full_k = torch.cat(dense_k_parts, dim=2)
        full_v = torch.cat(dense_v_parts, dim=2)
        
        S_dense = full_k.shape[2]
        if dense_blocks:
            dense_positions_list = []
            for blk in dense_blocks:
                dense_positions_list.extend(blk.token_indices)
            dense_positions = torch.tensor(dense_positions_list, dtype=torch.long, device=q.device)
        else:
            dense_positions = torch.arange(total_seq_len - S_dense, total_seq_len, device=q.device)
        
        if cos is not None and sin is not None:
            cos_dense = cos[0, dense_positions].unsqueeze(0).unsqueeze(1)
            sin_dense = sin[0, dense_positions].unsqueeze(0).unsqueeze(1)
            full_k_rot = (full_k * cos_dense) + (rotate_half(full_k) * sin_dense)
        else:
            full_k_rot = full_k
        
        k_dense_rep = repeat_kv_at_dim(full_k_rot, num_key_value_groups, dim=1)
        v_dense_rep = repeat_kv_at_dim(full_v, num_key_value_groups, dim=1)
        scores_dense = torch.sum(q * k_dense_rep, dim=-1).squeeze(0) * inv_scale
    else:
        S_dense = 0
        scores_dense = torch.empty((H_q, 0), device=q.device, dtype=q.dtype)

    scores_all = torch.cat([scores_anchor, scores_compressed, scores_dense], dim=-1)
    
    if diagnostics:
        has_nan = torch.isnan(scores_all).any().item()
        has_posinf = (scores_all == float('inf')).any().item()
        if has_nan or has_posinf:
            scores_all = scores_all.clone()
            if has_nan:
                scores_all[torch.isnan(scores_all)] = -1e4
            if has_posinf:
                scores_all[scores_all == float('inf')] = 1e4

    probs_all = torch.nn.functional.softmax(scores_all, dim=-1)
    P_anchor, P_comp, P_dense = torch.split(probs_all, [N, N * block_capacity, S_dense], dim=-1)

    O_final = _attend_and_reconstruct_v(
        P_anchor=P_anchor,
        P_comp=P_comp,
        P_dense=P_dense,
        U=U,
        V_V=V_V,
        anchors_V=anchors_V,
        scales=scales,
        v_dense_rep=v_dense_rep if S_dense > 0 else torch.empty((0,), device=q.device),
        H_q=H_q,
        N=N,
        block_capacity=block_capacity,
        R=R,
        D=D,
        S_dense=S_dense,
        V_V_perm=None,
    )

    return O_final.unsqueeze(0).unsqueeze(2)


def native_triton_sparse_attn_decode(
    q:                    torch.Tensor,
    block_indices:        torch.Tensor,
    pool:                 object,
    dense_blocks:         list,            
    active_k:             torch.Tensor,
    active_v:             torch.Tensor,
    num_key_value_groups: int,
    R:                    int = 16,
    S_MAX:                int = 64,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
    total_seq_len:        int = 0,
    max_valid_len:        Optional[int] = None,
    cos_sliced:           Optional[torch.Tensor] = None,
    sin_sliced:           Optional[torch.Tensor] = None,
    session_id:           Optional[str] = None,
    layer_idx:            Optional[int] = None,
    decode_workspace:     Optional[dict] = None,
) -> torch.Tensor:
    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1
    
    if not HAS_TRITON:
        return _pytorch_vectorized_sparse_attn_decode(
            q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len, max_valid_len=max_valid_len,
            cos_sliced=cos_sliced, sin_sliced=sin_sliced,
            session_id=session_id, layer_idx=layer_idx, decode_workspace=decode_workspace,
        )
        
    inv_scale = 1.0 / math.sqrt(D)
    N = block_indices.shape[0] if block_indices is not None else 0
    
    if N > 0:
        try:
            q_sq = q[0, :, 0, :]
            D_pad = triton.next_power_of_2(D)
            R_pad = triton.next_power_of_2(R)
            S_pad = triton.next_power_of_2(S_MAX)
            
            BLOCKS_PER_CHUNK = 16
            if N > BLOCKS_PER_CHUNK:
                num_chunks = (N + BLOCKS_PER_CHUNK - 1) // BLOCKS_PER_CHUNK
            else:
                num_chunks = 1
                
            grid = (H_q, num_chunks)
            
            if num_chunks > 1:
                cache_key = (H_q, num_chunks, D_pad, q.device)
                if not hasattr(native_triton_sparse_attn_decode, "_workspaces_cache"):
                    native_triton_sparse_attn_decode._workspaces_cache = {}
                
                workspaces = native_triton_sparse_attn_decode._workspaces_cache.get(cache_key)
                if workspaces is None:
                    out_workspace = torch.empty((H_q, num_chunks, D_pad), device=q.device, dtype=torch.float32)
                    m_workspace = torch.empty((H_q, num_chunks), device=q.device, dtype=torch.float32)
                    l_workspace = torch.empty((H_q, num_chunks), device=q.device, dtype=torch.float32)
                    workspaces = (out_workspace, m_workspace, l_workspace)
                    native_triton_sparse_attn_decode._workspaces_cache[cache_key] = workspaces
                else:
                    out_workspace, m_workspace, l_workspace = workspaces
                    
                out = torch.empty((H_q, D), device=q.device, dtype=torch.float32)
                m_out = torch.empty((H_q,), device=q.device, dtype=torch.float32)
                l_out = torch.empty((H_q,), device=q.device, dtype=torch.float32)
            else:
                out = torch.empty((H_q, D), device=q.device, dtype=torch.float32)
                m_out = torch.empty((H_q,), device=q.device, dtype=torch.float32)
                l_out = torch.empty((H_q,), device=q.device, dtype=torch.float32)
                out_workspace = out
                m_workspace = m_out
                l_workspace = l_out
            
            _fused_sparse_decode_kernel[grid](
                q_sq, block_indices, pool.anchors_K, pool.anchors_V, pool.V_K, pool.V_V,
                pool.U, pool.U_scale, pool.scales, pool.seq_lens,
                out_workspace, m_workspace, l_workspace,
                q_sq.stride(0), q_sq.stride(1),
                pool.anchors_K.stride(0), pool.anchors_K.stride(1), pool.anchors_K.stride(2),
                pool.anchors_V.stride(0), pool.anchors_V.stride(1), pool.anchors_V.stride(2),
                pool.V_K.stride(0), pool.V_K.stride(1), pool.V_K.stride(2), pool.V_K.stride(3),
                pool.V_V.stride(0), pool.V_V.stride(1), pool.V_V.stride(2), pool.V_V.stride(3),
                pool.U.stride(0), pool.U.stride(1), pool.U.stride(2),
                out_workspace.stride(0), out_workspace.stride(1),
                N, H_q, pool.anchors_K.shape[1], num_key_value_groups, D_pad,
                R_pad, S_pad, inv_scale, BLOCKS_PER_CHUNK, num_chunks
            )
            
            if num_chunks > 1:
                grid_reduction = (H_q,)
                _fused_sparse_decode_reduction_kernel[grid_reduction](
                    out_workspace, m_workspace, l_workspace, out, m_out, l_out,
                    num_chunks, D_pad
                )
            
            if dense_blocks or (active_k is not None and active_k.shape[2] > 0):
                O_i = out * l_out.unsqueeze(-1)
                m_i = m_out
                l_i = l_out
                
                dense_k_parts = []
                dense_v_parts = []
                for blk in (dense_blocks or []):
                    if blk.anchor_kv is not None:
                        dense_k_parts.append(blk.anchor_kv[:, 0].unsqueeze(2))
                        dense_v_parts.append(blk.anchor_kv[:, 1].unsqueeze(2))
                    if blk.active_k is not None and blk.active_k.shape[2] > 0:
                        dense_k_parts.append(blk.active_k)
                        dense_v_parts.append(blk.active_v)
                if active_k is not None and active_k.shape[2] > 0:
                    dense_k_parts.append(active_k)
                    dense_v_parts.append(active_v)
                    
                if dense_k_parts:
                    k_kv = torch.cat(dense_k_parts, dim=2).float()
                    v_kv = torch.cat(dense_v_parts, dim=2).float()
                    
                    H_q, D = q_sq.shape
                    H_kv = k_kv.shape[1]
                    n_rep = num_key_value_groups
                    
                    q_reshaped = q_sq.float().view(H_kv, n_rep, D)
                    k_permuted = k_kv[0].permute(0, 2, 1)
                    
                    s = torch.bmm(q_reshaped, k_permuted).view(H_q, -1) * inv_scale
                    
                    m_b = s.max(-1).values
                    m_new = torch.maximum(m_i, m_b)
                    a = torch.exp(m_i - m_new)
                    P = torch.exp(s - m_new.unsqueeze(-1))
                    l_i = a * l_i + P.sum(-1)
                    
                    P_reshaped = P.view(H_kv, n_rep, -1)
                    v_permuted = v_kv[0]
                    
                    O_i_delta = torch.bmm(P_reshaped, v_permuted).view(H_q, D)
                    O_i = a.unsqueeze(-1) * O_i + O_i_delta
                    m_i = m_new
                    
                out = O_i / l_i.unsqueeze(-1)
                
            return out.unsqueeze(0).unsqueeze(2).to(q.dtype)
        except Exception as e:
            if not hasattr(native_triton_sparse_attn_decode, "fallback_fired"):
                print(f"[DiffKV] WARNING: Triton compilation failed: {e}. Falling back to PyTorch vectorized decoder.")
                native_triton_sparse_attn_decode.fallback_fired = True
            return _pytorch_vectorized_sparse_attn_decode(
                q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
                anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len, max_valid_len=max_valid_len,
                cos_sliced=cos_sliced, sin_sliced=sin_sliced,
                session_id=session_id, layer_idx=layer_idx, decode_workspace=decode_workspace,
            )
    else:
        return _pytorch_vectorized_sparse_attn_decode(
            q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len, max_valid_len=max_valid_len,
            cos_sliced=cos_sliced, sin_sliced=sin_sliced,
            session_id=session_id, layer_idx=layer_idx, decode_workspace=decode_workspace,
        )

# ── 4. TritonDiffKV Low-Rank Reconstruction ───────────────────────────────────

def triton_fused_reconstruct(
    U: torch.Tensor,
    V: torch.Tensor,
    anchor: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    n_tokens, rank = U.shape
    _, feat_dim = V.shape

    if not HAS_TRITON:
        result = (torch.matmul(U.float(), V.float()) * scale + anchor.float()).to(U.dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if out is None:
        out = torch.empty((n_tokens, feat_dim), device=U.device, dtype=U.dtype)

    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = 64
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(n_tokens, BLOCK_SIZE_N), triton.cdiv(feat_dim, BLOCK_SIZE_D))

    _use_nvtx = _has_cuda()
    if _use_nvtx:
        _nvtx_push("Triton_LowRank_Recon_Kernel_Launch")

    lowrank_recon_kernel[grid](
        U, V, anchor, out,
        U.stride(0), U.stride(1),
        V.stride(0), V.stride(1),
        anchor.stride(0),
        out.stride(0), out.stride(1),
        n_tokens, rank, feat_dim, scale,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    if _use_nvtx:
        _nvtx_pop()

    return out


class TritonDiffKV:
    _recon_buffers = {}

    @classmethod
    def _get_recon_buffer(cls, n_tokens: int, feat_dim: int, device, dtype) -> torch.Tensor:
        key = (device, dtype, feat_dim)
        if key not in cls._recon_buffers or cls._recon_buffers[key].shape[0] < n_tokens:
            alloc_size = max(2048, n_tokens)
            cls._recon_buffers[key] = torch.zeros(
                (alloc_size, feat_dim), device=device, dtype=dtype
            )
        return cls._recon_buffers[key][:n_tokens]

    @staticmethod
    def reconstruct_lowrank(
        U: torch.Tensor,
        V: torch.Tensor,
        anchor: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        out_buf = TritonDiffKV._get_recon_buffer(
            U.shape[0], V.shape[1], U.device, U.dtype
        )
        try:
            out = triton_fused_reconstruct(U, V, anchor, out=out_buf, scale=scale)
            return out.clone()
        except Exception as e:
            return (torch.matmul(U.float(), V.float()) * scale + anchor.float()).to(U.dtype)

    @staticmethod
    def reconstruct_lowrank_sparse(
        U: torch.Tensor,
        V: torch.Tensor,
        anchor: torch.Tensor,
        sparse_indices: Optional[torch.Tensor],
        sparse_values: Optional[torch.Tensor],
        scale: float = 1.0,
    ) -> torch.Tensor:
        out = TritonDiffKV.reconstruct_lowrank(U, V, anchor, scale)
        if sparse_indices is not None and sparse_indices.numel() > 0:
            out.view(-1).index_add_(
                0, sparse_indices.long(), sparse_values.to(out.dtype)
            )
        return out
