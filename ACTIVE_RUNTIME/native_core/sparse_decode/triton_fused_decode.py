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
from native_core.compression.lowrank import reconstruct_batch_U

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

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

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
        # Residual correction pointers (C1). res_pos = K positions, res_pos_v = V positions
        # (they differ — K and V select different worst-reconstructed tokens).
        pool_res_k_ptr, pool_res_v_ptr, pool_res_pos_ptr, pool_res_pos_v_ptr, pool_res_n_ptr,
        # Fact anchor override pointers (C2)
        pool_fact_pos_ptr, pool_fact_ak_ptr, pool_fact_av_ptr,
        out_ptr, m_ptr, l_ptr,
        stride_q_h, stride_q_d,
        stride_ak_n, stride_ak_h, stride_ak_d,
        stride_av_n, stride_av_h, stride_av_d,
        stride_vk_n, stride_vk_r, stride_vk_h, stride_vk_d,
        stride_vv_n, stride_vv_r, stride_vv_h, stride_vv_d,
        stride_u_n, stride_u_s, stride_u_r,
        stride_res_k_n, stride_res_k_s, stride_res_k_h, stride_res_k_d,
        stride_res_v_n, stride_res_v_s, stride_res_v_h, stride_res_v_d,
        stride_res_pos_n, stride_res_pos_v_n,
        stride_fact_pos_n,
        stride_fact_ak_n, stride_fact_ak_f, stride_fact_ak_h, stride_fact_ak_d,
        stride_fact_av_n, stride_fact_av_f, stride_fact_av_h, stride_fact_av_d,
        stride_out_h, stride_out_d,
        N: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr, KV_GRP: tl.constexpr, D: tl.constexpr,
        R: tl.constexpr, S_MAX: tl.constexpr, INV_SCALE: tl.constexpr, BLOCKS_PER_CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr,
        MAX_RESIDUAL: tl.constexpr, MAX_FACT: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr, HAS_FACT: tl.constexpr,
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

            # ── C1: Residual K correction (ALIGNED to Mac reference) ──────────────
            # residual_K_values store (exact - lowrank_recon) at the worst-reconstructed
            # positions (lowrank.py:659). Add q·resK to the delta score AT that position
            # so the token's score becomes exact — one token per position, matching
            # _pytorch_vectorized_sparse_attn_decode's scatter_add_ (line 1164) and
            # fused_decode_mps (806). res_k is pre-rotated (anchor RoPE) in the dispatcher.
            # NOTE: this replaces the old "append residuals as extra softmax tokens" path,
            # which double-counted those positions and was measurably worse than no
            # correction at all (see CUDA_TRITON_AUDIT.md F1).
            if HAS_RESIDUAL:
                for ri in range(MAX_RESIDUAL):
                    r_pos_k = tl.load(pool_res_pos_ptr + pool_idx * stride_res_pos_n + ri)
                    if r_pos_k >= 0:
                        rk = tl.load(pool_res_k_ptr + pool_idx * stride_res_k_n +
                                     ri * stride_res_k_s + h_kv * stride_res_k_h +
                                     offs_d * stride_res_k_d).to(tl.float32)
                        r_corr = tl.sum(q * rk) * INV_SCALE
                        s = tl.where(offs_s == r_pos_k, s + r_corr, s)

            # ── C2: Fact Anchor Override — replace scores at flagged positions ──
            if HAS_FACT:
                for fi in range(MAX_FACT):
                    fact_pos = tl.load(pool_fact_pos_ptr + pool_idx * stride_fact_pos_n + fi)
                    if fact_pos >= 0:
                        fact_k_ptrs = pool_fact_ak_ptr + pool_idx * stride_fact_ak_n + fi * stride_fact_ak_f + h_kv * stride_fact_ak_h + offs_d * stride_fact_ak_d
                        fact_k = tl.load(fact_k_ptrs).to(tl.float32)
                        fact_score = tl.sum(q * fact_k) * INV_SCALE
                        # Override: scatter exact score at the flagged delta position
                        replace_mask = offs_s == fact_pos
                        s = tl.where(replace_mask, fact_score, s)
            
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
            
            # ── C2: Fact Anchor Override Value Correction ──
            O_fact_corr = tl.zeros([D], dtype=tl.float32)
            if HAS_FACT:
                for fi in range(MAX_FACT):
                    fact_pos = tl.load(pool_fact_pos_ptr + pool_idx * stride_fact_pos_n + fi)
                    if fact_pos >= 0:
                        # Get attention weight for this fact token
                        replace_mask = offs_s == fact_pos
                        p_fact = tl.sum(tl.where(replace_mask, p_delta, 0.0), axis=0)
                        
                        # Load exact fact V
                        fact_v_ptrs = pool_fact_av_ptr + pool_idx * stride_fact_av_n + fi * stride_fact_av_f + h_kv * stride_fact_av_h + offs_d * stride_fact_av_d
                        fact_v = tl.load(fact_v_ptrs).to(tl.float32)
                        
                        # Compute low-rank reconstructed V at fact_pos
                        u_val_ptrs = pool_u_ptr + pool_idx * stride_u_n + fact_pos * stride_u_s + offs_r * stride_u_r
                        u_val = tl.load(u_val_ptrs).to(tl.float32) * u_scale
                        v_recon = tl.sum(u_val[:, None] * vv, axis=0) * scale + av
                        
                        # Accumulate correction: p_fact * (fact_v - v_recon)
                        O_fact_corr += p_fact * (fact_v - v_recon)
            
            # ── C1: Residual V correction (ALIGNED to Mac reference) ──────────────
            # O += p_delta[res_pos_v] · resV, with resV = (exact - recon) V
            # (lowrank.py:666). Uses the SAME unnormalized p_delta as o_delta (the K
            # correction above already made p_delta exact at these positions), and is
            # normalized by l_i at the end — matching _pytorch_vectorized_…'s
            # gather(P, res_pos_v)·resV (lines 1291-1298). V residual positions differ
            # from K's, so this loop reads pool_res_pos_v_ptr (not pool_res_pos_ptr).
            O_res_corr = tl.zeros([D], dtype=tl.float32)
            if HAS_RESIDUAL:
                for ri in range(MAX_RESIDUAL):
                    r_pos_v = tl.load(pool_res_pos_v_ptr + pool_idx * stride_res_pos_v_n + ri)
                    if r_pos_v >= 0:
                        p_at = tl.sum(tl.where(offs_s == r_pos_v, p_delta, 0.0), axis=0)
                        rv = tl.load(pool_res_v_ptr + pool_idx * stride_res_v_n +
                                     ri * stride_res_v_s + h_kv * stride_res_v_h +
                                     offs_d * stride_res_v_d).to(tl.float32)
                        O_res_corr += p_at * rv

            O_i = O_i * alpha + (p_anchor + p_delta_sum) * av + o_delta + O_fact_corr + O_res_corr
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

    # OPT-E: Pairwise-merge kernel — one pass of a binary tree reduction.
    # Each program merges a (left, right) chunk pair: accumulator at slot `left`
    # absorbs slot `right` using the standard online-softmax (LSE-safe) formula.
    # Launch ceil(NUM_ACTIVE / 2) programs along axis-1 to run all pairs in parallel.
    # After ceil(log2(NUM_CHUNKS)) such passes only slot 0 remains, which is the
    # final normalised output.  Only used when num_chunks >= 8 (see _dispatch_reduction
    # below); the sequential kernel is faster at smaller chunk counts.
    @triton.jit
    def _fused_reduction_pairwise_kernel(
        workspace_ptr,   # [H_q, NUM_CHUNKS_PAD, D]  in-place
        m_ptr,           # [H_q, NUM_CHUNKS_PAD]      in-place
        l_ptr,           # [H_q, NUM_CHUNKS_PAD]      in-place
        STRIDE: tl.constexpr,   # distance between left and right slot (1, 2, 4, …)
        NUM_CHUNKS: tl.constexpr,
        D: tl.constexpr,
    ):
        h_q   = tl.program_id(0)
        pair  = tl.program_id(1)
        left  = pair * 2 * STRIDE
        right = left + STRIDE
        if right >= NUM_CHUNKS:
            return

        offs_d = tl.arange(0, D)

        m_l = tl.load(m_ptr + h_q * NUM_CHUNKS + left)
        m_r = tl.load(m_ptr + h_q * NUM_CHUNKS + right)
        l_l = tl.load(l_ptr + h_q * NUM_CHUNKS + left)
        l_r = tl.load(l_ptr + h_q * NUM_CHUNKS + right)
        O_l = tl.load(workspace_ptr + h_q * NUM_CHUNKS * D + left  * D + offs_d).to(tl.float32)
        O_r = tl.load(workspace_ptr + h_q * NUM_CHUNKS * D + right * D + offs_d).to(tl.float32)

        m_new = tl.maximum(m_l, m_r)
        alpha = tl.exp(m_l - m_new)
        beta  = tl.exp(m_r - m_new)
        l_new = l_l * alpha + l_r * beta
        O_new = O_l * alpha + O_r * beta

        tl.store(m_ptr + h_q * NUM_CHUNKS + left, m_new)
        tl.store(l_ptr + h_q * NUM_CHUNKS + left, l_new)
        tl.store(workspace_ptr + h_q * NUM_CHUNKS * D + left * D + offs_d, O_new)

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


# ── OPT-E: Reduction dispatcher \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def _dispatch_reduction(
    out_workspace: "torch.Tensor",
    m_workspace:   "torch.Tensor",
    l_workspace:   "torch.Tensor",
    out:           "torch.Tensor",
    m_out:         "torch.Tensor",
    l_out:         "torch.Tensor",
    num_chunks:    int,
    D:             int,
    H_q:           int,
) -> None:
    """
    Dispatch the multi-chunk LSE-safe reduction to either:
      - Sequential kernel   when num_chunks < 8  (low launch overhead dominates)
      - Parallel tree       when num_chunks >= 8  (ceil(log2(C)) passes, each parallel)

    The parallel tree pads num_chunks to the next power of 2 and runs one
    pairwise-merge Triton kernel per level.  After the last pass, slot 0 of the
    workspace holds the un-normalised merged accumulator; we copy it to `out` and
    divide by l.  Levels where a right-hand partner is out of range are no-ops
    (guarded inside _fused_reduction_pairwise_kernel by the `right >= NUM_CHUNKS`
    early-exit).
    """
    if not HAS_TRITON:
        return  # caller handles the no-Triton path

    PARALLEL_THRESHOLD = 8  # Q3 answer: >= 8 chunks use parallel tree

    if num_chunks < PARALLEL_THRESHOLD:
        # ── Sequential path (existing kernel) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        _fused_sparse_decode_reduction_kernel[(H_q,)](
            out_workspace, m_workspace, l_workspace, out, m_out, l_out,
            num_chunks, D,
        )
    else:
        # ── Parallel tree-reduction path \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        import math as _math
        # Pad to next power of 2 so every level has even pair counts.
        nc_pad = 1 << _math.ceil(_math.log2(num_chunks))
        n_levels = _math.ceil(_math.log2(nc_pad))

        stride = 1
        for _ in range(n_levels):
            n_pairs = nc_pad // (2 * stride)
            if n_pairs == 0:
                break
            grid_pw = (H_q, n_pairs)
            _fused_reduction_pairwise_kernel[grid_pw](
                out_workspace, m_workspace, l_workspace,
                STRIDE=stride, NUM_CHUNKS=num_chunks, D=D,
            )
            stride *= 2

        # After all passes, slot 0 holds the merged (unnormalised) accumulator.
        import torch as _torch
        m_final = m_workspace[:, 0]               # [H_q]
        l_final = l_workspace[:, 0]               # [H_q]
        O_final = out_workspace[:, 0, :].float()  # [H_q, D]

        # The pairwise kernel does NOT divide by l — do it here.
        out_f = O_final / l_final.unsqueeze(-1).clamp(min=1e-9)
        out.copy_(out_f)
        if m_out is not None:
            m_out.copy_(m_final)
        if l_out is not None:
            l_out.copy_(l_final)


# ── 2. PyTorch JIT Helpers for Compilation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

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


def _prefill_fused_history_attend_compiled(
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
    residual_K_positions: torch.Tensor,
    residual_K_values: torch.Tensor,
    residual_V_positions: torch.Tensor,
    residual_V_values: torch.Tensor,
) -> torch.Tensor:
    N = U.shape[0]
    S = U.shape[1]
    R = U.shape[2]
    H = q.shape[1]
    Q = q.shape[2]
    D = q.shape[3]

    V_K_flat = V_K.reshape(N, R, H * D)
    deltas_k_flat = torch.bmm(U, V_K_flat)
    deltas_k = deltas_k_flat.reshape(N, S, H, D) * scales.view(N, 1, 1, 1).to(q.dtype)

    K_unrot_full = torch.cat(
        [anchors_K.unsqueeze(1), anchors_K.unsqueeze(1) + deltas_k], dim=1
    )

    # ── Post-SVD Sparse Residual Correction for Key (Prefill History) ──
    if residual_K_positions.numel() > 0:
        res_pos_K_clamped = residual_K_positions.clamp(min=0).long()
        mask_K = (residual_K_positions >= 0).unsqueeze(-1).unsqueeze(-1)
        res_vals_K_masked = residual_K_values.to(K_unrot_full.dtype) * mask_K
        index_K = (res_pos_K_clamped + 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
        K_unrot_full.scatter_add_(dim=1, index=index_K, src=res_vals_K_masked)

    half_d = D // 2
    K_half1 = K_unrot_full[..., :half_d]
    K_half2 = K_unrot_full[..., half_d:]
    K_rotated = torch.cat([-K_half2, K_half1], dim=-1)
    cos_s = cos_sliced.squeeze(2)
    sin_s = sin_sliced.squeeze(2)
    K_rot_full = (K_unrot_full * cos_s.unsqueeze(2)
                  + K_rotated  * sin_s.unsqueeze(2))

    q_hqd = q.squeeze(0).reshape(H, Q, D).float()
    K_hnd = K_rot_full.permute(2, 0, 1, 3).reshape(H, N * (1 + S), D).float()
    scores_flat = torch.bmm(q_hqd, K_hnd.transpose(1, 2)) * inv_scale
    scores = scores_flat.reshape(H, Q, N, 1 + S)

    col = torch.arange(1 + S, device=U.device, dtype=torch.long).unsqueeze(0)
    valid = col <= seq_lens.unsqueeze(1).long()
    scores = scores.masked_fill(
        (~valid).unsqueeze(0).unsqueeze(0), float('-inf')
    )

    scores_f = scores.reshape(H, Q, N * (1 + S)).float()
    lse_hist  = torch.logsumexp(scores_f, dim=-1)
    weights_f = torch.softmax(scores_f, dim=-1)
    weights   = weights_f.reshape(H, Q, N, 1 + S).to(q.dtype)

    w_anchor = weights[:, :, :, 0]
    w_delta  = weights[:, :, :, 1:]
    p_total  = w_anchor + w_delta.sum(dim=-1)

    anc_v_hnd = anchors_V.permute(1, 0, 2)
    out_anchor = torch.bmm(p_total, anc_v_hnd)

    w_delta_perm = w_delta.permute(2, 0, 1, 3)
    w_delta_flat = w_delta_perm.reshape(N, H * Q, S)
    W_proj_flat = torch.bmm(w_delta_flat, U)
    W_proj_flat = W_proj_flat * scales.view(N, 1, 1).to(q.dtype)
    W_proj = W_proj_flat.reshape(N, H, Q, R).permute(1, 2, 0, 3)

    V_V_t  = V_V.permute(2, 0, 1, 3)
    W_proj_flat2 = W_proj.reshape(H, Q, N * R)
    V_V_t_flat2 = V_V_t.contiguous().reshape(H, N * R, D)
    out_delta = torch.bmm(W_proj_flat2, V_V_t_flat2)

    out_hist = (out_anchor + out_delta).unsqueeze(0)

    # ── Post-SVD Sparse Residual Correction for Value (Prefill History) ──
    if residual_V_positions.numel() > 0:
        w_d_perm = w_delta.permute(2, 0, 1, 3)
        res_pos_V_clamped = residual_V_positions.clamp(min=0).long()
        res_pos_V_expanded = res_pos_V_clamped.unsqueeze(1).unsqueeze(2).expand(-1, H, Q, -1)
        w_res_V = torch.gather(w_d_perm, dim=3, index=res_pos_V_expanded)
        
        mask_V = (residual_V_positions >= 0).unsqueeze(1).unsqueeze(2).expand(-1, H, Q, -1)
        w_res_V = w_res_V.masked_fill(~mask_V, 0.0)

        res_val_V_perm = residual_V_values.to(w_res_V.dtype).permute(0, 2, 1, 3)
        O_res = torch.sum(w_res_V.unsqueeze(-1) * res_val_V_perm.unsqueeze(2), dim=(0, 3))
        out_hist = out_hist + O_res.unsqueeze(0).to(out_hist.dtype)

    lse_out  = lse_hist.to(q.dtype).unsqueeze(0)

    lse_padded = lse_out.unsqueeze(-1).expand(1, H, Q, D)
    return torch.stack([out_hist, lse_padded], dim=0)


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
    try:
        _backend = "inductor"
        _mode = "reduce-overhead" if _IS_CUDA_AVAILABLE else "default"
        print(f"[DiffKV JIT] Compiling _prefill_fused_history_attend with backend={_backend}, mode={_mode} (dynamic=True) ...")
        _prefill_fused_history_attend = torch.compile(
            _prefill_fused_history_attend_compiled,
            backend=_backend,
            mode=_mode,
            fullgraph=False,
            dynamic=True,
        )
    except Exception as e:
        print(f"[DiffKV JIT] torch.compile of _prefill_fused_history_attend failed ({e}). Falling back to JIT script.")
        try:
            _prefill_fused_history_attend = torch.jit.script(_prefill_fused_history_attend_compiled)
        except Exception:
            _prefill_fused_history_attend = _prefill_fused_history_attend_compiled
else:
    _reconstruct_and_score = _reconstruct_and_score_compiled
    _attend_and_reconstruct_v = _attend_and_reconstruct_v_compiled
    try:
        _prefill_fused_history_attend = torch.jit.script(_prefill_fused_history_attend_compiled)
    except Exception:
        _prefill_fused_history_attend = _prefill_fused_history_attend_compiled

# ── 2b. Stratified U reconstruction helper (Issue 1 fix) ─────────────────────
# The Triton kernel reads pool.U (int8) + pool.U_scale (scalar).  That path
# bypasses the stratified quantization system (U_sem int4 + U_fact fp16) built
# for accuracy parity with MPS.  This helper reconstructs a full fp16 U tensor
# from the stratified components for ALL active blocks BEFORE kernel dispatch,
# then patches a proxy pool so the kernel sees fp16 values and U_scale=1.0.
#
# Cost: one reconstruct_batch_U call per decode step (~N*S*R fp16 elements).
# This is cheaper than re-running full attention; on CUDA with VRAM headroom
# the intermediate tensor lives and dies within the decode step.

class _StratifiedUProxy:
    """
    Thin wrapper that stands in for pool when passing U data to Triton kernels.
    Exposes pool.U as the full-precision fp16 reconstruction and pool.U_scale
    as a tensor of ones so the kernel applies no additional scaling.
    All other attributes delegate to the original pool object.
    """
    __slots__ = ("_pool", "U", "U_scale")

    def __init__(self, pool, U_fp16: torch.Tensor):
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "U", U_fp16.to(pool.dtype))       # [n_blocks, S, R] fp16
        # Ones so that kernel's  u = u * u_scale  is a no-op
        object.__setattr__(self, "U_scale",
            torch.ones(pool.U_scale.shape, device=pool.U_scale.device, dtype=pool.U_scale.dtype))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_pool"), name)


# ── OPT-D: Generation-keyed module-level U proxy cache ────────────────────────
# Maps (pool_id, pool_generation, active_key_tuple) → _StratifiedUProxy.
# Invalidated automatically when pool._stratified_generation increments
# (NativeBlockPool.write_block does this on every pool write).
# Cache is intentionally small: a new pool_id is rare (one per session) and each
# generation evicts all prior entries, so the dict stays bounded at O(1) entries
# per active pool across typical single-session inference.
_stratified_proxy_cache: dict = {}


def _build_stratified_U_for_triton(
    pool,
    block_indices: torch.Tensor,
) -> "tuple[object, bool]":
    """
    Build a _StratifiedUProxy for the Triton kernel dispatch.

    Resolves two accuracy issues simultaneously:

    Issue 1 — Stratified U bypass:
        The raw Triton path read pool.U (int8) + pool.U_scale (one scalar per
        block), completely skipping the stratified quantization system
        (U_sem int4 for semantic components + U_fact fp16 for factual
        components).  reconstruct_batch_U() correctly combines both, giving
        the same accuracy as the MPS path.

    Issue 2 — Token-norm quantization error in the int8 path:
        During compression, each token's U row is scaled by its L2 norm before
        storage (compress_layer_blocks_gpu:593, lowrank.py).  The int8 path
        then requantizes ALL rows with a SINGLE global U_scale = max_abs/127.
        Tokens with large norms consume most of the int8 range; low-norm tokens
        are pushed into a tiny slice of it, causing ~0.8% relative error.
        By serving full fp16 U (reconstructed from U_sem + U_fact), every token
        is represented at float16 precision regardless of its norm magnitude.
        The U_scale tensor in the proxy is all-ones, so the kernel's
            u = u * u_scale
        becomes a no-op and the fp16 values are used exactly.

    OPT-C — Active-only reconstruction:
        Previous code called reconstruct_batch_U(pool, all_idx) where all_idx
        covered every slot in the pool (up to 256+), running the expensive
        int4-unpack Python loop for all of them even if only 16 are active.
        Now we do a cheap int8 dequant broadcast for all slots and then overwrite
        only the active slots with accurate stratified reconstruction — cutting the
        expensive loop from N_pool to N_active iterations (typically 16×).

    OPT-D — Generation-keyed cache:
        If the pool has not been written to (same _stratified_generation) and the
        same set of active block indices is requested, return the cached proxy
        immediately without any tensor allocation or reconstruction. Generation
        increments in NativeBlockPool.write_block on every pool update.

    Returns (proxy_or_pool, used_stratified).
    If the pool has no stratified components (n_semantic all-zero) the original
    pool is returned unchanged to avoid a pointless VRAM allocation.
    """
    # Fast-path: no stratified components in this pool at all
    n_sem_attr = getattr(pool, "n_semantic", None)
    if n_sem_attr is None or (n_sem_attr == 0).all().item():
        return pool, False

    active_idx = block_indices.long()

    # ── OPT-D: Cache lookup ────────────────────────────────────────────────────
    pool_id  = id(pool)
    pool_gen = getattr(pool, "_stratified_generation", -1)
    # Sort for stable key; routing order may differ but the block set is what matters.
    # tolist() is O(N_active), acceptable for N_active ≤ 256.
    active_key = tuple(sorted(active_idx.tolist()))
    cache_key  = (pool_id, pool_gen, active_key)

    cached = _stratified_proxy_cache.get(cache_key)
    if cached is not None:
        return cached, True   # ── Cache HIT: skip all tensor work ──

    # Evict all stale entries for this pool_id (keeps dict bounded at O(1) entries).
    stale = [k for k in _stratified_proxy_cache if k[0] == pool_id]
    for k in stale:
        del _stratified_proxy_cache[k]

    # ── OPT-C: Reconstruct only active slots ─────────────────────────────────
    # 1. Fast int8 dequant broadcast as the baseline for ALL pool slots (single
    #    GPU multiply — no Python loop). Active-slot overwrite follows.
    U_full = (pool.U.to(pool.dtype)
              * pool.U_scale.view(-1, 1, 1).to(pool.dtype))  # [n_pool, S, R]
    U_full = U_full.clone()  # detach from pool view before scatter-write

    # 2. Accurate int4/fp16 stratified reconstruction for the N_active slots only.
    #    reconstruct_batch_U loops over idx — now N_active (≤ 16 typical) instead
    #    of N_pool (up to 256+), giving ≥16× speedup on the hot per-decode call.
    U_active = reconstruct_batch_U(pool, active_idx)   # [N_active, S, R] fp16
    U_full[active_idx] = U_active

    proxy = _StratifiedUProxy(pool, U_full)

    # ── OPT-D: Populate cache ─────────────────────────────────────────────────
    _stratified_proxy_cache[cache_key] = proxy
    return proxy, True


# ── F2: routed-row gather + anchor-RoPE rotation, cached per routing interval ──
# The Triton kernels address EVERY per-block tensor through block_indices, so they
# only ever read the N routed rows — yet the dispatchers used to `.clone()` the
# ENTIRE pool (anchors_K, V_K, res_k) on EVERY decode token just to scatter N
# rotated rows into it: O(pool_size) memory traffic per token on the exact path
# being optimized (audit finding F2). This helper instead gathers the N routed
# rows, rotates those, and hands the kernel compact [N]-row tensors with
# block_indices remapped to arange(N) — bit-identical kernel inputs (equivalence
# certified on CPU by tests/test_triton_gather_equiv.py).
#
# The gathered set depends only on (pool contents, routing order, anchor
# positions), NOT on q, so it is cached keyed on (pool id, pool generation,
# exact index order) — the same route-interval reuse that made the MLX fused
# decode fast (per-token work O(1), per-route work O(N)). pool generation is
# NativeBlockPool._stratified_generation, bumped on every write_block/reset;
# if the attribute is missing the cache is skipped (gather still wins vs clone).
# NOTE: the block_indices.tolist() key costs one small D2H sync per call — the
# stratified-U cache above already pays an identical sync per call, so this adds
# no NEW sync point on CUDA.
_gathered_rot_cache: dict = {}


def _gather_routed_blocks_for_kernel(pool_for_kernel, block_indices, anchor_indices, cos, sin):
    """Gather the [N] routed rows of every per-block tensor the Triton kernels
    read, pre-rotating the K-side rows (anchors_K, V_K, res_k) by each block
    anchor's RoPE when rotation inputs are provided. Returns a dict of compact
    tensors plus `idx` = arange(N) to pass as the kernel's block_indices."""
    N = block_indices.shape[0]
    device = block_indices.device
    base_pool = object.__getattribute__(pool_for_kernel, "_pool") \
        if isinstance(pool_for_kernel, _StratifiedUProxy) else pool_for_kernel
    pool_gen = getattr(base_pool, "_stratified_generation", None)
    cache_key = None
    if pool_gen is not None:
        cache_key = (id(base_pool), pool_gen, tuple(block_indices.tolist()))
        got = _gathered_rot_cache.get(cache_key)
        if got is not None:
            return got

    indices = block_indices.long()
    g = {}
    g["idx"] = torch.arange(N, device=device, dtype=block_indices.dtype)

    anchors_K = pool_for_kernel.anchors_K[indices]      # [N, H_kv, D]
    V_K       = pool_for_kernel.V_K[indices]            # [N, R, H_kv, D]
    g["anchors_V"] = pool_for_kernel.anchors_V[indices]
    g["V_V"]       = pool_for_kernel.V_V[indices]
    g["U"]         = pool_for_kernel.U[indices]
    g["U_scale"]   = pool_for_kernel.U_scale[indices]
    g["scales"]    = pool_for_kernel.scales[indices]
    g["seq_lens"]  = pool_for_kernel.seq_lens[indices]

    do_rot = (anchor_indices is not None and cos is not None and sin is not None)
    cos_anc = sin_anc = None
    if do_rot:
        cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
        sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
        anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1)
        cos_anc = cos_flat[anchor_indices_clamped].to(device=V_K.device, dtype=V_K.dtype).unsqueeze(1).unsqueeze(2)
        sin_anc = sin_flat[anchor_indices_clamped].to(device=V_K.device, dtype=V_K.dtype).unsqueeze(1).unsqueeze(2)
        cos_anc_2d = cos_anc.squeeze(2)                 # [N, 1, D] for [N, H_kv, D] tensors
        sin_anc_2d = sin_anc.squeeze(2)
        V_K       = V_K * cos_anc + rotate_half(V_K) * sin_anc
        anchors_K = anchors_K * cos_anc_2d + rotate_half(anchors_K) * sin_anc_2d
    g["anchors_K"] = anchors_K
    g["V_K"]       = V_K

    res_k   = getattr(base_pool, "residual_K_values",    None)
    res_v   = getattr(base_pool, "residual_V_values",    None)
    res_pos = getattr(base_pool, "residual_K_positions", None)
    res_pos_v = getattr(base_pool, "residual_V_positions", None)
    g["has_res"] = (res_k is not None and res_v is not None and
                    res_pos is not None and res_pos_v is not None)
    if g["has_res"]:
        res_k_g = res_k[indices]                        # [N, MAX_RES, H_kv, D]
        if do_rot:
            # Pre-rotate residual K by the block anchor's RoPE, exactly like V_K
            # (reference rotates res_val_K identically). res_v is never rotated.
            res_k_g = res_k_g * cos_anc + rotate_half(res_k_g) * sin_anc
        g["res_k"]     = res_k_g
        g["res_v"]     = res_v[indices]
        g["res_pos"]   = res_pos[indices]
        g["res_pos_v"] = res_pos_v[indices]
        g["res_n"]     = (g["res_pos"] >= 0).sum(dim=-1).to(torch.int32)
        g["max_res_pad"] = res_pos.shape[1]
    else:
        g["res_k"]     = torch.empty((0, 0, 0, 0), device=device)
        g["res_v"]     = torch.empty((0, 0, 0, 0), device=device)
        g["res_pos"]   = torch.empty((0, 0), device=device, dtype=torch.int16)
        g["res_pos_v"] = torch.empty((0, 0), device=device, dtype=torch.int16)
        g["res_n"]     = torch.zeros((N,), device=device, dtype=torch.int32)
        g["max_res_pad"] = 1

    fact_pos = getattr(base_pool, "fact_anchor_positions", None)
    fact_ak  = getattr(base_pool, "fact_anchors_K",        None)
    fact_av  = getattr(base_pool, "fact_anchors_V",        None)
    g["has_fact"] = (fact_pos is not None and fact_ak is not None and fact_av is not None)
    if g["has_fact"]:
        g["fact_pos"] = fact_pos[indices]
        g["fact_ak"]  = fact_ak[indices]
        g["fact_av"]  = fact_av[indices]
        g["max_fact"] = fact_pos.shape[1]
    else:
        g["fact_pos"] = torch.empty((0, 0),       device=device, dtype=torch.int16)
        g["fact_ak"]  = torch.empty((0, 0, 0, 0), device=device)
        g["fact_av"]  = torch.empty((0, 0, 0, 0), device=device)
        g["max_fact"] = 1

    if cache_key is not None:
        # Evict stale entries for this pool (older generation or routing) so the
        # cache stays O(1) entries per live pool.
        stale = [k for k in _gathered_rot_cache
                 if k[0] == cache_key[0] and k != cache_key]
        for k in stale:
            del _gathered_rot_cache[k]
        _gathered_rot_cache[cache_key] = g
    return g


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
    if torch.isnan(Q).any():
        print(f"[fused_decode_mps DEBUG] Q has NaN at start! shape={Q.shape}", flush=True)
    q        = Q.float()

    if block_indices is None or block_indices.numel() == 0:
        return torch.zeros((H_q, D), dtype=Q.dtype, device=Q.device), torch.full((H_q,), float('-inf'), dtype=Q.dtype, device=Q.device)

    N   = block_indices.shape[0]
    idx = block_indices.long()

    U_a    = reconstruct_batch_U(pool, idx).float()
    S_comp = U_a.shape[1]
    AncK_a = pool.anchors_K[idx].float()
    AncV_a = pool.anchors_V[idx].float()
    VK_a   = pool.V_K[idx].float()
    VV_a   = pool.V_V[idx].float()

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    if anchor_indices is not None and cos is not None and sin is not None:
        cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
        sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
        
        # 1. Exact RoPE for anchor key
        anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
        cos_anc = cos_flat[anchor_indices_clamped].to(device=VK_a.device, dtype=VK_a.dtype).unsqueeze(1)
        sin_anc = sin_flat[anchor_indices_clamped].to(device=VK_a.device, dtype=VK_a.dtype).unsqueeze(1)
        
        AncK_e = AncK_a.repeat_interleave(gpk, dim=1)
        AncV_e = AncV_a.repeat_interleave(gpk, dim=1)
        VK_e   = VK_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
        VV_e   = VV_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
        
        AncK_e_rot = AncK_e * cos_anc + rotate_half(AncK_e) * sin_anc
        s_anc = torch.einsum('hd,nhd->hn', q, AncK_e_rot) * scale

        # Forwards rotation for VK_e using cos_anc and sin_anc (since keys are pre-rotated at ingest)
        cos_anc_exp = cos_anc.unsqueeze(2) # [N, 1, 1, D]
        sin_anc_exp = sin_anc.unsqueeze(2)
        VK_e_rot = VK_e * cos_anc_exp + rotate_half(VK_e) * sin_anc_exp
        
        q_proj_n = torch.einsum('hd,nhrd->nhr', q, VK_e_rot) * scale
        delta_s = torch.einsum('nhr,nsr->hns', q_proj_n, U_a)
        delta_s = delta_s * pool.scales[idx].float().view(1, N, 1)
        delta_s = delta_s + s_anc.unsqueeze(-1)
    else:
        # No RoPE / approximate formulation fallback
        AncK_e = AncK_a.repeat_interleave(gpk, dim=1)
        AncV_e = AncV_a.repeat_interleave(gpk, dim=1)
        VK_e   = VK_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
        VV_e   = VV_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()

        s_anc = torch.einsum('hd,nhd->hn', q, AncK_e) * scale
        q_proj_n = torch.einsum('hd,nhrd->nhr', q, VK_e) * scale
        delta_s = torch.einsum('nhr,nsr->hns', q_proj_n, U_a)
        delta_s = delta_s * pool.scales[idx].float().view(1, N, 1)
        delta_s = delta_s + s_anc.unsqueeze(-1)

    # ── Post-SVD Sparse Residual Correction for Key ──
    res_pos_K = getattr(pool, "residual_K_positions", None)
    res_val_K = getattr(pool, "residual_K_values", None)
    if res_pos_K is not None and res_val_K is not None:
        res_pos_K_idx = res_pos_K[idx]
        res_val_K_idx = res_val_K[idx].float()
        if res_pos_K_idx.numel() > 0:
            res_val_K_e = res_val_K_idx.repeat_interleave(gpk, dim=2)  # [N, MAX_RESIDUAL_TOKENS, H_q, D]
            has_rope = (anchor_indices is not None and cos is not None and sin is not None)
            if has_rope:
                cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                cos_anc = cos_flat[anchor_indices_clamped].to(device=q.device, dtype=q.dtype).unsqueeze(1)
                sin_anc = sin_flat[anchor_indices_clamped].to(device=q.device, dtype=q.dtype).unsqueeze(1)
                cos_anc_exp = cos_anc.unsqueeze(1) # [N, 1, 1, D]
                sin_anc_exp = sin_anc.unsqueeze(1)
                res_val_K_rot = res_val_K_e * cos_anc_exp + rotate_half(res_val_K_e) * sin_anc_exp
                corr_K = torch.sum(q.unsqueeze(0).unsqueeze(1) * res_val_K_rot, dim=-1) * scale
            else:
                corr_K = torch.sum(q.unsqueeze(0).unsqueeze(1) * res_val_K_e, dim=-1) * scale

            corr_K_perm = corr_K.permute(2, 0, 1)
            mask_K = (res_pos_K_idx >= 0).unsqueeze(0).expand(H_q, -1, -1)
            corr_K_perm = corr_K_perm.masked_fill(~mask_K, 0.0)

            res_pos_K_clamped_expanded = res_pos_K_idx.clamp(min=0).long().unsqueeze(0).expand(H_q, -1, -1)
            delta_s.scatter_add_(dim=2, index=res_pos_K_clamped_expanded, src=corr_K_perm)

    # ── Solution 3: Fact Anchor overrides for Key ──
    fact_pos = getattr(pool, "fact_anchor_positions", None)
    fact_anc_K_pool = getattr(pool, "fact_anchors_K", None)
    if fact_pos is not None and fact_anc_K_pool is not None and N > 0:
        fact_pos_idx = fact_pos[idx]  # [N, 3]
        fact_anc_K_idx = fact_anc_K_pool[idx].float()  # [N, 3, num_kv_heads, D]
        mask = (fact_pos_idx >= 0)  # [N, 3]
        if mask.any():
            K_exact = fact_anc_K_idx.repeat_interleave(gpk, dim=2)  # [N, 3, H_q, D]
            has_rope = (anchor_indices is not None and cos is not None and sin is not None)
            if has_rope:
                cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                cos_anc = cos_flat[anchor_indices_clamped].to(device=q.device, dtype=q.dtype).unsqueeze(1)
                sin_anc = sin_flat[anchor_indices_clamped].to(device=q.device, dtype=q.dtype).unsqueeze(1)
                cos_anc_exp = cos_anc.unsqueeze(1) # [N, 1, 1, D]
                sin_anc_exp = sin_anc.unsqueeze(1)
                K_exact = K_exact * cos_anc_exp + rotate_half(K_exact) * sin_anc_exp
            score_exact = torch.sum(q.view(1, 1, H_q, D) * K_exact, dim=-1) * scale
            score_exact = score_exact.permute(2, 0, 1)  # [H_q, N, 3]
            fact_pos_idx_clamped = fact_pos_idx.clamp(min=0).long()
            fact_pos_idx_clamped_expanded = fact_pos_idx_clamped.unsqueeze(0).expand(H_q, -1, -1)  # [H_q, N, 3]
            mask_expanded = mask.unsqueeze(0).expand(H_q, -1, -1)  # [H_q, N, 3]
            delta_s_updates = torch.zeros_like(delta_s)
            delta_s_updates.scatter_(dim=2, index=fact_pos_idx_clamped_expanded, src=score_exact)
            update_mask = torch.zeros_like(delta_s, dtype=torch.bool)
            update_mask.scatter_(dim=2, index=fact_pos_idx_clamped_expanded, src=mask_expanded)
            delta_s = torch.where(update_mask, delta_s_updates, delta_s)


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

    # Scale the base/anchor value by the total attention weight of all tokens in the block
    w_block_sum = w_anc + w_d.sum(dim=-1)

    O = torch.zeros((H_q, D), device=Q.device, dtype=torch.float32)
    O = O + torch.einsum('hn,nhd->hd', w_block_sum, AncV_e)

    w_proj = torch.einsum('hns,nsr->nhr', w_d, U_a)
    w_proj = w_proj * pool.scales[idx].float().view(N, 1, 1)
    O = O + torch.einsum('nhr,nhrd->hd', w_proj, VV_e)

    # ── Post-SVD Sparse Residual Correction for Value ──
    res_pos_V = getattr(pool, "residual_V_positions", None)
    res_val_V = getattr(pool, "residual_V_values", None)
    if res_pos_V is not None and res_val_V is not None:
        res_pos_V_idx = res_pos_V[idx]
        res_val_V_idx = res_val_V[idx].float()
        if res_pos_V_idx.numel() > 0:
            res_val_V_e = res_val_V_idx.repeat_interleave(gpk, dim=2)  # [N, MAX_RESIDUAL_TOKENS, H_q, D]
            w_d_perm = w_d.permute(1, 0, 2)
            res_pos_V_clamped = res_pos_V_idx.clamp(min=0).long()
            res_pos_V_expanded = res_pos_V_clamped.unsqueeze(1).expand(-1, H_q, -1)
            w_res_V = torch.gather(w_d_perm, dim=2, index=res_pos_V_expanded)
            
            mask_V = (res_pos_V_idx >= 0).unsqueeze(1).expand(-1, H_q, -1)
            w_res_V = w_res_V.masked_fill(~mask_V, 0.0)

            res_val_V_e_perm = res_val_V_e.permute(0, 2, 1, 3)
            O_res = torch.sum(w_res_V.unsqueeze(-1) * res_val_V_e_perm, dim=(0, 2))
            O = O + O_res

    # ── Solution 3: Fact Anchor overrides for Value ──
    fact_pos = getattr(pool, "fact_anchor_positions", None)
    fact_anc_V_pool = getattr(pool, "fact_anchors_V", None)
    if fact_pos is not None and fact_anc_V_pool is not None and N > 0:
        fact_pos_idx = fact_pos[idx]  # [N, 3]
        fact_anc_V_idx = fact_anc_V_pool[idx].float()  # [N, 3, num_kv_heads, D]
        mask = (fact_pos_idx >= 0)  # [N, 3]
        if mask.any():
            V_exact = fact_anc_V_idx.repeat_interleave(gpk, dim=2)  # [N, 3, H_q, D]
            fact_pos_idx_clamped = fact_pos_idx.clamp(min=0).long()
            R = U_a.shape[2]
            u_val = torch.gather(U_a, dim=1, index=fact_pos_idx_clamped.unsqueeze(-1).expand(-1, -1, R))
            v_svd_sum = torch.sum(u_val.unsqueeze(1).unsqueeze(-1) * VV_e.unsqueeze(2), dim=3)  # [N, H_q, 3, D]
            scales_idx = pool.scales[idx].float()
            v_svd = v_svd_sum * scales_idx.view(N, 1, 1, 1) + AncV_e.unsqueeze(2)  # [N, H_q, 3, D]
            v_svd = v_svd.permute(0, 2, 1, 3)  # [N, 3, H_q, D]
            v_diff = V_exact - v_svd  # [N, 3, H_q, D]
            w_d_perm = w_d.permute(1, 2, 0)  # [N, S_comp, H_q]
            w_pos = torch.gather(w_d_perm, dim=1, index=fact_pos_idx_clamped.unsqueeze(-1).expand(-1, -1, H_q))  # [N, 3, H_q]
            w_pos = w_pos.unsqueeze(-1)  # [N, 3, H_q, 1]
            update_term = w_pos * v_diff  # [N, 3, H_q, D]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [N, 3, 1, 1]
            update_term = update_term.masked_fill(~mask_expanded, 0.0)
            O = O + torch.sum(update_term, dim=(0, 1))


    if torch.isnan(O).any() or torch.isinf(O).any() or torch.isnan(lse).any():
        print("[fused_decode_mps DEBUG] NaN/Inf detected!")
        print(f"q finite: {torch.isfinite(q).all().item()} min: {q.min().item()} max: {q.max().item()}")
        print(f"AncK_e finite: {torch.isfinite(AncK_e).all().item()} min: {AncK_e.min().item()} max: {AncK_e.min().item()} max: {AncK_e.max().item()}")
        print(f"VK_e finite: {torch.isfinite(VK_e).all().item()} min: {VK_e.min().item()} max: {VK_e.max().item()}")
        print(f"U_a finite: {torch.isfinite(U_a).all().item()} min: {U_a.min().item()} max: {U_a.max().item()}")
        print(f"scales finite: {torch.isfinite(pool.scales[idx]).all().item()} min: {pool.scales[idx].min().item()} max: {pool.scales[idx].max().item()}")
        print(f"scale: {scale}")
        print(f"s_anc finite: {torch.isfinite(s_anc).all().item()}")
        print(f"q_proj_n finite: {torch.isfinite(q_proj_n).all().item()}")
        print(f"delta_s finite: {torch.isfinite(delta_s).all().item()}")
        print(f"scores finite: {torch.isfinite(scores).all().item()}")
        print(f"lse finite: {torch.isfinite(lse).all().item()}")
        print(f"w finite: {torch.isfinite(w).all().item()}")

    if lse.max().item() > 100.0:
        print(f"[fused_decode_mps DIAG] lse has large value! max={lse.max().item():.2f}")
        print(f"  q min/max: {q.min().item():.4f}/{q.max().item():.4f}")
        print(f"  AncK_e min/max: {AncK_e.min().item():.4f}/{AncK_e.max().item():.4f}")
        print(f"  VK_e min/max: {VK_e.min().item():.4f}/{VK_e.max().item():.4f}")
        print(f"  U_a min/max: {U_a.min().item():.4f}/{U_a.max().item():.4f}")
        print(f"  pool.scales[idx] min/max: {pool.scales[idx].min().item():.4f}/{pool.scales[idx].max().item():.4f}")
        print(f"  s_anc min/max: {s_anc.min().item():.4f}/{s_anc.max().item():.4f}")
        print(f"  delta_s min/max: {delta_s.min().item():.4f}/{delta_s.max().item():.4f}")
        print(f"  scores min/max: {scores.min().item():.4f}/{scores.max().item():.4f}")

    return O.to(Q.dtype), lse.to(torch.float32)


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
    # Issue 6 fix: The gathered-KV workspace cache was designed for MPS where
    # pool.gather() is expensive.  On CUDA the pool tensors are already contiguous
    # GPU memory and gather is cheap; caching stale tensors across block
    # evictions/reallocations causes silent accuracy bugs.  Disable the cache
    # on CUDA by default; enable explicitly with DIFFKV_DECODE_CACHE_ENABLED=1.
    config = getattr(pool, "config", None)
    _on_cuda = (str(getattr(pool, "device", "")) == "cuda" or
                (hasattr(pool, "device") and str(pool.device).startswith("cuda")))
    _default_cache_enabled = False if _on_cuda else True
    decode_cache_enabled = config.decode_cache_enabled if config is not None else _default_cache_enabled
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
        indices = block_indices.long()
        current_version = session_dict.get("routing_version", 0) if session_dict is not None else 0

        cached_gathered = None
        gathered_cache = None
        if use_workspace_cache and session_dict is not None and layer_idx is not None:
            gathered_cache = session_dict.setdefault("gathered_kv", {})
            cached_val = gathered_cache.get(layer_idx)
            if cached_val is not None and cached_val[0] == current_version:
                cached_gathered = cached_val[1]

        # Issue 5 fix: approximate_attn=True is the only supported formulation.
        # The Project-Then-Attend (PTA) approach rotates keys at the ANCHOR position
        # rather than at each token's exact position, which avoids the O(N*S) RoPE
        # embedding gather that per-token rotation would require.  The dead
        # approximate_attn=False branch (exact per-token RoPE) has been removed to
        # reduce confusion.  See the paper §3.2 for the theoretical justification.

        if cached_gathered is not None:
            U, V_K, V_V, anchors_K, anchors_V, scales, seq_lens_t = cached_gathered
        else:
            U = reconstruct_batch_U(pool, indices).to(q.dtype)
            
            V_K_raw = pool.V_K[indices]
            anchors_K_raw = pool.anchors_K[indices]
            
            if anchor_indices is not None and cos is not None and sin is not None:
                cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                cpu_anc_check = anchor_indices.cpu()
                if (cpu_anc_check >= cos_flat.shape[0]).any():
                    print(f"[DiffKV DEBUG] Out of bounds check: layer_idx={layer_idx} anchor_indices={cpu_anc_check.tolist()} cos_flat.shape={list(cos_flat.shape)}", flush=True)
                
                # Clamp anchor_indices to prevent GPU out of bounds
                anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                cos_anc = cos_flat[anchor_indices_clamped].to(device=V_K_raw.device, dtype=V_K_raw.dtype).unsqueeze(1).unsqueeze(2)
                sin_anc = sin_flat[anchor_indices_clamped].to(device=V_K_raw.device, dtype=V_K_raw.dtype).unsqueeze(1).unsqueeze(2)
                
                cos_anc_2d = cos_flat[anchor_indices_clamped].to(device=anchors_K_raw.device, dtype=anchors_K_raw.dtype).unsqueeze(1)
                sin_anc_2d = sin_flat[anchor_indices_clamped].to(device=anchors_K_raw.device, dtype=anchors_K_raw.dtype).unsqueeze(1)
                
                V_K_raw = V_K_raw * cos_anc + rotate_half(V_K_raw) * sin_anc
                anchors_K_raw = anchors_K_raw * cos_anc_2d + rotate_half(anchors_K_raw) * sin_anc_2d
                
            V_K = repeat_kv_at_dim(V_K_raw, num_key_value_groups, dim=2)
            V_V = repeat_kv_at_dim(pool.V_V[indices], num_key_value_groups, dim=2)
            anchors_K = repeat_kv_at_dim(anchors_K_raw, num_key_value_groups, dim=1)
            anchors_V = repeat_kv_at_dim(pool.anchors_V[indices], num_key_value_groups, dim=1)
            scales = pool.scales[indices].view(N, 1, 1)
            seq_lens_t = pool.seq_lens[indices]
            
            if use_workspace_cache and gathered_cache is not None:
                gathered_cache[layer_idx] = (current_version, (U, V_K, V_V, anchors_K, anchors_V, scales, seq_lens_t))
        
        block_capacity = U.shape[1]
        R = U.shape[2]

        if max_valid_len is None:
            max_valid_len = int(seq_lens_t.max().item())

        # Cast inputs/pool slices to float32 to prevent float16 overflow/inf issues on MPS
        q_sq_fp32 = q_sq.float()
        anchors_K_fp32 = anchors_K.float()
        V_K_fp32 = V_K.float()
        U_fp32 = U.float()

        has_rope = (anchor_indices is not None and cos is not None and sin is not None)

        # ── Project-Then-Attend formulation (anchor-position RoPE approximation) ──
        # Issue 5 note: per-token exact RoPE branch removed — see comment above.
        # exact anchor score: [H_q, N]
        scores_anchor = torch.einsum('hd,nhd->hn', q_sq_fp32, anchors_K_fp32) * inv_scale
        
        # Project query to V_K: [N, H_q, R]
        q_proj = torch.einsum('hd,nrhd->nhr', q_sq_fp32, V_K_fp32) * inv_scale
        
        # Inner product with U: [H_q, N, block_capacity]
        scores_block = torch.einsum('nhr,nsr->hns', q_proj, U_fp32) * scales.float().view(1, N, 1)
        scores_block = scores_block + scores_anchor.unsqueeze(-1)

        res_pos_K = getattr(pool, "residual_K_positions", None)
        res_val_K = getattr(pool, "residual_K_values", None)
        if res_pos_K is not None and res_val_K is not None and N > 0:
            res_pos_K_idx = res_pos_K[indices]
            res_val_K_idx = res_val_K[indices].float()
            if res_pos_K_idx.numel() > 0:
                res_val_K_e = res_val_K_idx.repeat_interleave(num_key_value_groups, dim=2)
                if has_rope:
                    cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                    sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                    anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                    cos_anc = cos_flat[anchor_indices_clamped].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(1)
                    sin_anc = sin_flat[anchor_indices_clamped].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(1)
                    cos_anc_exp = cos_anc.unsqueeze(1) # [N, 1, 1, D]
                    sin_anc_exp = sin_anc.unsqueeze(1)
                    res_val_K_rot = res_val_K_e * cos_anc_exp + rotate_half(res_val_K_e) * sin_anc_exp
                    corr_K = torch.sum(q_sq_fp32.unsqueeze(0).unsqueeze(1) * res_val_K_rot, dim=-1) * inv_scale
                else:
                    corr_K = torch.sum(q_sq_fp32.unsqueeze(0).unsqueeze(1) * res_val_K_e, dim=-1) * inv_scale

                corr_K_perm = corr_K.permute(2, 0, 1)
                mask_K = (res_pos_K_idx >= 0).unsqueeze(0).expand(H_q, -1, -1)
                corr_K_perm = corr_K_perm.masked_fill(~mask_K, 0.0)

                res_pos_K_clamped_expanded = res_pos_K_idx.clamp(min=0).long().unsqueeze(0).expand(H_q, -1, -1)
                scores_block.scatter_add_(dim=2, index=res_pos_K_clamped_expanded, src=corr_K_perm)

        # ── Solution 3: Fact Anchor overrides for Key ──
        fact_pos = getattr(pool, "fact_anchor_positions", None)
        fact_anc_K_pool = getattr(pool, "fact_anchors_K", None)
        if fact_pos is not None and fact_anc_K_pool is not None and N > 0:
            fact_pos_idx = fact_pos[indices]  # [N, 3]
            fact_anc_K_idx = fact_anc_K_pool[indices].float()  # [N, 3, num_kv_heads, D]
            mask = (fact_pos_idx >= 0)  # [N, 3]
            if mask.any():
                K_exact = fact_anc_K_idx.repeat_interleave(num_key_value_groups, dim=2)  # [N, 3, H_q, D]
                has_rope = (anchor_indices is not None and cos is not None and sin is not None)
                if has_rope:
                    cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                    sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                    anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                    cos_anc = cos_flat[anchor_indices_clamped].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(1)
                    sin_anc = sin_flat[anchor_indices_clamped].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(1)
                    cos_anc_exp = cos_anc.unsqueeze(1) # [N, 1, 1, D]
                    sin_anc_exp = sin_anc.unsqueeze(1)
                    K_exact = K_exact * cos_anc_exp + rotate_half(K_exact) * sin_anc_exp
                score_exact = torch.sum(q_sq_fp32.view(1, 1, H_q, D) * K_exact, dim=-1) * inv_scale  # [N, 3, H_q]
                score_exact = score_exact.permute(2, 0, 1)  # [H_q, N, 3]
                fact_pos_idx_clamped = fact_pos_idx.clamp(min=0).long()
                fact_pos_idx_clamped_expanded = fact_pos_idx_clamped.unsqueeze(0).expand(H_q, -1, -1)  # [H_q, N, 3]
                mask_expanded = mask.unsqueeze(0).expand(H_q, -1, -1)  # [H_q, N, 3]
                scores_block_updates = torch.zeros_like(scores_block)
                scores_block_updates.scatter_(dim=2, index=fact_pos_idx_clamped_expanded, src=score_exact)
                update_mask = torch.zeros_like(scores_block, dtype=torch.bool)
                update_mask.scatter_(dim=2, index=fact_pos_idx_clamped_expanded, src=mask_expanded)
                scores_block = torch.where(update_mask, scores_block_updates, scores_block)

        # Mask out-of-bounds tokens
        s_range = torch.arange(block_capacity, device=q.device).view(1, 1, -1)
        valid_mask = s_range < seq_lens_t.view(1, N, 1)
        scores_block = scores_block.masked_fill(~valid_mask, float('-inf'))
        
        scores_compressed = scores_block.reshape(H_q, N * block_capacity)
    else:
        scores_anchor = torch.empty((H_q, 0), device=q.device, dtype=torch.float32)
        scores_compressed = torch.empty((H_q, 0), device=q.device, dtype=torch.float32)

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
        scores_dense = torch.sum(q.float() * k_dense_rep.float(), dim=-1).squeeze(0) * inv_scale
    else:
        S_dense = 0
        scores_dense = torch.empty((H_q, 0), device=q.device, dtype=torch.float32)

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

    probs_all = torch.nn.functional.softmax(scores_all.float(), dim=-1).to(scores_all.dtype)
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

    # ── Post-SVD Sparse Residual Correction for Value ──
    res_pos_V = getattr(pool, "residual_V_positions", None)
    res_val_V = getattr(pool, "residual_V_values", None)
    if res_pos_V is not None and res_val_V is not None and N > 0:
        res_pos_V_idx = res_pos_V[indices]
        res_val_V_idx = res_val_V[indices].float()
        if res_pos_V_idx.numel() > 0:
            res_val_V_e = res_val_V_idx.repeat_interleave(num_key_value_groups, dim=2)  # [N, MAX_RESIDUAL_TOKENS, H_q, D]
            P_comp_reshaped = P_comp.view(H_q, N, block_capacity).permute(1, 0, 2)  # [N, H_q, block_capacity]
            res_pos_V_clamped = res_pos_V_idx.clamp(min=0).long()
            res_pos_V_expanded = res_pos_V_clamped.unsqueeze(1).expand(-1, H_q, -1)
            w_res_V = torch.gather(P_comp_reshaped, dim=2, index=res_pos_V_expanded)
            
            mask_V = (res_pos_V_idx >= 0).unsqueeze(1).expand(-1, H_q, -1)
            w_res_V = w_res_V.masked_fill(~mask_V, 0.0)

            res_val_V_e_perm = res_val_V_e.permute(0, 2, 1, 3)
            O_res = torch.sum(w_res_V.unsqueeze(-1) * res_val_V_e_perm, dim=(0, 2))
            O_final = O_final + O_res.to(O_final.dtype)

    # ── Solution 3: Fact Anchor overrides for Value ──
    fact_pos = getattr(pool, "fact_anchor_positions", None)
    fact_anc_V_pool = getattr(pool, "fact_anchors_V", None)
    if fact_pos is not None and fact_anc_V_pool is not None and N > 0:
        fact_pos_idx = fact_pos[indices]
        fact_anc_V_idx = fact_anc_V_pool[indices].float()
        mask = (fact_pos_idx >= 0)  # [N, 3]
        if mask.any():
            V_exact = fact_anc_V_idx.repeat_interleave(num_key_value_groups, dim=2)  # [N, 3, H_q, D]
            fact_pos_idx_clamped = fact_pos_idx.clamp(min=0).long()
            R = U.shape[2]
            u_val = torch.gather(U, dim=1, index=fact_pos_idx_clamped.unsqueeze(-1).expand(-1, -1, R))  # [N, 3, R]
            v_svd_sum = torch.sum(u_val.unsqueeze(1).unsqueeze(-1) * V_V.permute(0, 2, 1, 3).unsqueeze(2), dim=3)  # [N, H_q, 3, D]
            v_svd = v_svd_sum * scales.view(N, 1, 1, 1) + anchors_V.unsqueeze(2)  # [N, H_q, 3, D]
            v_svd = v_svd.permute(0, 2, 1, 3)  # [N, 3, H_q, D]
            v_diff = V_exact - v_svd  # [N, 3, H_q, D]
            fact_idx_expanded = fact_pos_idx_clamped.unsqueeze(1).expand(-1, H_q, -1)  # [N, H_q, 3]
            w_pos = torch.gather(P_comp_reshaped, dim=2, index=fact_idx_expanded)  # [N, H_q, 3]
            w_pos = w_pos.permute(0, 2, 1)  # [N, 3, H_q]
            w_pos = w_pos.unsqueeze(-1)  # [N, 3, H_q, 1]
            update_term = w_pos * v_diff  # [N, 3, H_q, D]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [N, 3, 1, 1]
            update_term = update_term.masked_fill(~mask_expanded, 0.0)
            O_final = O_final + torch.sum(update_term, dim=(0, 1)).to(O_final.dtype)

    # Issue 4 fix: Layer-0 NaN diagnostics gated behind DIFFKV_LAYER0_DEBUG=1.
    # Previously fired every decode step for every session, burning CPU time and
    # polluting logs.  Enable during bring-up / NaN debugging only.
    if layer_idx == 0 and os.environ.get("DIFFKV_LAYER0_DEBUG", "0") == "1":
        print(f"[DiffKV DEBUG] layer 0 check - q has nan: {torch.isnan(q).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - block_indices has nan: {torch.isnan(block_indices).any().item() if block_indices is not None else False}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - U has nan: {torch.isnan(U).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - V_K has nan: {torch.isnan(V_K).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - V_V has nan: {torch.isnan(V_V).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - anchors_K has nan: {torch.isnan(anchors_K).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - anchors_V has nan: {torch.isnan(anchors_V).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - scales has nan: {torch.isnan(scales).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - scores_anchor has nan: {torch.isnan(scores_anchor).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - scores_compressed has nan: {torch.isnan(scores_compressed).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - scores_dense has nan: {torch.isnan(scores_dense).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - scores_all has nan: {torch.isnan(scores_all).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - probs_all has nan: {torch.isnan(probs_all).any().item()}", flush=True)
        print(f"[DiffKV DEBUG] layer 0 check - O_final has nan: {torch.isnan(O_final).any().item()}", flush=True)

    return O_final.to(q.dtype).unsqueeze(0).unsqueeze(2)


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
            # Issue 1 fix: Pre-reconstruct stratified U (int4 semantic + fp16 factual)
            # into a full fp16 tensor before Triton dispatch.  The kernel's U_scale
            # tensor is all-ones in the proxy so u = u * 1.0 = exact fp16 values.
            # This gives CUDA accuracy parity with the MPS path (reconstruct_batch_U).
            pool_for_kernel, _used_strat = _build_stratified_U_for_triton(pool, block_indices)

            # F2 fix: gather + rotate ONLY the N routed rows (cached per routing
            # interval) instead of cloning the whole pool per token; the kernel
            # gets compact [N]-row tensors with block_indices remapped to
            # arange(N) — bit-identical inputs, O(pool)→O(N) per-token traffic.
            g = _gather_routed_blocks_for_kernel(
                pool_for_kernel, block_indices, anchor_indices, cos, sin)

            _fused_sparse_decode_kernel[grid](
                q_sq, g["idx"], g["anchors_K"], g["anchors_V"], g["V_K"], g["V_V"],
                g["U"], g["U_scale"], g["scales"], g["seq_lens"],
                g["res_k"], g["res_v"], g["res_pos"], g["res_pos_v"], g["res_n"],
                g["fact_pos"], g["fact_ak"], g["fact_av"],
                out_workspace, m_workspace, l_workspace,
                q_sq.stride(0), q_sq.stride(1),
                g["anchors_K"].stride(0), g["anchors_K"].stride(1), g["anchors_K"].stride(2),
                g["anchors_V"].stride(0), g["anchors_V"].stride(1), g["anchors_V"].stride(2),
                g["V_K"].stride(0), g["V_K"].stride(1), g["V_K"].stride(2), g["V_K"].stride(3),
                g["V_V"].stride(0), g["V_V"].stride(1), g["V_V"].stride(2), g["V_V"].stride(3),
                g["U"].stride(0), g["U"].stride(1), g["U"].stride(2),
                g["res_k"].stride(0), g["res_k"].stride(1), g["res_k"].stride(2), g["res_k"].stride(3),
                g["res_v"].stride(0), g["res_v"].stride(1), g["res_v"].stride(2), g["res_v"].stride(3),
                g["res_pos"].stride(0), g["res_pos_v"].stride(0),
                g["fact_pos"].stride(0),
                g["fact_ak"].stride(0), g["fact_ak"].stride(1), g["fact_ak"].stride(2), g["fact_ak"].stride(3),
                g["fact_av"].stride(0), g["fact_av"].stride(1), g["fact_av"].stride(2), g["fact_av"].stride(3),
                out_workspace.stride(0), out_workspace.stride(1),
                N, H_q, g["anchors_K"].shape[1], num_key_value_groups, D_pad,
                R_pad, S_pad, inv_scale, BLOCKS_PER_CHUNK, num_chunks,
                MAX_RESIDUAL=g["max_res_pad"], MAX_FACT=g["max_fact"],
                HAS_RESIDUAL=g["has_res"], HAS_FACT=g["has_fact"]
            )
            
            if num_chunks > 1:
                # OPT-E: dispatches sequential vs. parallel tree reduction based on num_chunks
                _dispatch_reduction(
                    out_workspace, m_workspace, l_workspace, out, m_out, l_out,
                    num_chunks, D_pad, H_q,
                )
            
            if dense_blocks or (active_k is not None and active_k.shape[2] > 0):
                O_i = out * l_out.unsqueeze(-1)
                m_i = m_out
                l_i = l_out

                # Prefer active_k (pre-assembled workspace) over iterating dense_blocks
                # to avoid double-counting (active_k already contains all anchor+active data
                # from every dense block, assembled by assemble_dense_window_kv).
                if active_k is not None and active_k.shape[2] > 0:
                    k_kv = active_k.float()
                    v_kv = active_v.float()
                else:
                    dense_k_parts = []
                    dense_v_parts = []
                    for blk in (dense_blocks or []):
                        if blk.anchor_kv is not None:
                            dense_k_parts.append(blk.anchor_kv[:, 0].unsqueeze(2))
                            dense_v_parts.append(blk.anchor_kv[:, 1].unsqueeze(2))
                        if blk.active_k is not None and blk.active_k.shape[2] > 0:
                            dense_k_parts.append(blk.active_k)
                            dense_v_parts.append(blk.active_v)
                    if dense_k_parts:
                        k_kv = torch.cat(dense_k_parts, dim=2).float()
                        v_kv = torch.cat(dense_v_parts, dim=2).float()
                    else:
                        k_kv = None
                        v_kv = None

                if k_kv is not None and k_kv.shape[2] > 0:
                    # Apply RoPE rotation with correct absolute token positions.
                    # Without this, q_rot(pos_q) @ k_unrot gives wrong attention scores
                    # for dense (ACCUMULATING) blocks, breaking NIAH retrieval on CUDA.
                    if dense_blocks and cos is not None and sin is not None:
                        _dense_pos_list = []
                        for _blk in (dense_blocks or []):
                            _dense_pos_list.extend(_blk.token_indices)
                        if _dense_pos_list and len(_dense_pos_list) == k_kv.shape[2]:
                            _dp = torch.tensor(_dense_pos_list, dtype=torch.long, device=k_kv.device)
                            # cos/sin: [1, seq_len, head_dim]  (standard rotary_emb output)
                            _cos_d = cos[0, _dp.clamp(max=cos.shape[1] - 1)].unsqueeze(0).unsqueeze(1)  # [1,1,L,D]
                            _sin_d = sin[0, _dp.clamp(max=sin.shape[1] - 1)].unsqueeze(0).unsqueeze(1)
                            _hd = k_kv.shape[-1] // 2
                            _k_half = torch.cat([-k_kv[..., _hd:], k_kv[..., :_hd]], dim=-1)
                            k_kv = (k_kv * _cos_d.to(k_kv.dtype)
                                    + _k_half * _sin_d.to(k_kv.dtype))

                    H_q, D = q_sq.shape
                    H_kv = k_kv.shape[1]
                    n_rep = num_key_value_groups

                    q_reshaped = q_sq.float().view(H_kv, n_rep, D)
                    k_permuted = k_kv[0].permute(0, 2, 1)

                    s = torch.bmm(q_reshaped, k_permuted).view(H_q, -1) * inv_scale
                    
                    # ── Sparse LSE Bias (Ported from MLX) ────────────────────
                    bias_env = os.environ.get("DIFFKV_SPARSE_BIAS", "0.0").strip().lower()
                    if bias_env.startswith("auto"):
                        bias_parts = bias_env.split(",")
                        try:
                            bias_base = float(bias_parts[1]) if len(bias_parts) > 1 and bias_parts[1] else 2.0
                        except ValueError:
                            bias_base = 2.0
                        
                        lse_dense = torch.logsumexp(s, dim=-1)
                        lse_sparse = m_i + torch.log(torch.clamp(l_i, min=1e-9))
                        diff = lse_dense - lse_sparse
                        diff_clamped = torch.clamp(diff - 4.0, min=0.0)
                        adaptive_bias = torch.clamp(bias_base - 0.5 * diff_clamped, min=0.0)
                        
                        factor = torch.exp(adaptive_bias)
                        l_i = l_i * factor
                        O_i = O_i * factor.unsqueeze(-1)
                    else:
                        try:
                            bias_val = float(bias_env)
                        except ValueError:
                            bias_val = 0.0
                        if bias_val != 0.0:
                            factor = math.exp(bias_val)
                            l_i = l_i * factor
                            O_i = O_i * factor
                    
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

            # Observability (F3): confirm ONCE that the Triton kernel path is live.
            # On a GPU box this is the only positive signal that you're measuring
            # Triton and not the silent PyTorch fallback below.
            if not getattr(native_triton_sparse_attn_decode, "_triton_active_logged", False):
                print("[DiffKV] Triton fused-decode path ACTIVE (CUDA).")
                native_triton_sparse_attn_decode._triton_active_logged = True
            return out.unsqueeze(0).unsqueeze(2).to(q.dtype)
        except Exception as e:
            # Strict mode (F3): re-raise instead of masking a broken kernel behind
            # the slow PyTorch fallback. Use during GPU bring-up / validation so a
            # compile or numerics failure is loud. Default (unset) preserves the
            # historical silent-fallback behavior exactly.
            if os.environ.get("DIFFKV_TRITON_STRICT") == "1":
                raise
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


# ── 3b. Fused Combined Kernel: Compressed Blocks + Dense Window ───────────────
#
# Extends _fused_sparse_decode_kernel by iterating over dense window tokens in
# the SAME online softmax accumulator, replacing the 3-step pattern:
#   sparse Triton → dense SDPA → Python LSE-merge
# with a single Triton dispatch.  Grid and chunk logic are identical to the
# sparse-only kernel; dense tokens are processed after all sparse blocks in
# each chunk's accumulator.
#
# Key design decisions:
#   • Dense K/V are expected pre-RoPE-rotated by the caller (same as the
#     existing Python LSE-merge path in native_triton_sparse_attn_decode).
#   • L_dense is a tl.constexpr so the compiler elides the dense loop entirely
#     when there are no dense tokens (L_dense == 0), keeping the sparse-only
#     performance identical.
#   • Dense tokens are chunked identically to blocks: each Triton program
#     processes BLOCKS_PER_CHUNK sparse blocks followed by up to
#     DENSE_PER_CHUNK dense tokens from the matching dense slice.

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'num_warps': 4, 'num_stages': 2}),
            triton.Config({'num_warps': 4, 'num_stages': 4}),
            triton.Config({'num_warps': 8, 'num_stages': 2}),
            triton.Config({'num_warps': 8, 'num_stages': 4}),
        ],
        key=['N', 'L_dense']
    )
    @triton.jit
    def _fused_decode_combined_kernel(
        # ── Sparse compressed-block inputs (identical to _fused_sparse_decode_kernel) ──
        q_ptr, block_indices_ptr, pool_ak_ptr, pool_av_ptr, pool_vk_ptr, pool_vv_ptr,
        pool_u_ptr, pool_u_scale_ptr, pool_scales_ptr, pool_seq_lens_ptr,
        pool_res_k_ptr, pool_res_v_ptr, pool_res_pos_ptr, pool_res_pos_v_ptr, pool_res_n_ptr,
        pool_fact_pos_ptr, pool_fact_ak_ptr, pool_fact_av_ptr,
        # ── Dense window inputs (new) ──
        dense_k_ptr,        # [H_kv, L_dense, D]  pre-RoPE-rotated
        dense_v_ptr,        # [H_kv, L_dense, D]
        L_dense,            # int  (total dense tokens, NOT constexpr — passed as a scalar)
        # ── Strides for dense tensors ──
        stride_dk_h, stride_dk_l, stride_dk_d,
        stride_dv_h, stride_dv_l, stride_dv_d,
        # ── Output buffers ──
        out_ptr, m_ptr, l_ptr,
        # ── Strides (sparse, identical ordering to _fused_sparse_decode_kernel) ──
        stride_q_h, stride_q_d,
        stride_ak_n, stride_ak_h, stride_ak_d,
        stride_av_n, stride_av_h, stride_av_d,
        stride_vk_n, stride_vk_r, stride_vk_h, stride_vk_d,
        stride_vv_n, stride_vv_r, stride_vv_h, stride_vv_d,
        stride_u_n, stride_u_s, stride_u_r,
        stride_res_k_n, stride_res_k_s, stride_res_k_h, stride_res_k_d,
        stride_res_v_n, stride_res_v_s, stride_res_v_h, stride_res_v_d,
        stride_res_pos_n, stride_res_pos_v_n,
        stride_fact_pos_n,
        stride_fact_ak_n, stride_fact_ak_f, stride_fact_ak_h, stride_fact_ak_d,
        stride_fact_av_n, stride_fact_av_f, stride_fact_av_h, stride_fact_av_d,
        stride_out_h, stride_out_d,
        # ── Constexpr shape/config ──
        N: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr, KV_GRP: tl.constexpr, D: tl.constexpr,
        R: tl.constexpr, S_MAX: tl.constexpr, INV_SCALE: tl.constexpr,
        BLOCKS_PER_CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr,
        MAX_RESIDUAL: tl.constexpr, MAX_FACT: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr, HAS_FACT: tl.constexpr,
        DENSE_PER_CHUNK: tl.constexpr,   # dense tokens each chunk processes (0 disables the loop)
        BLOCK_SIZE_T: tl.constexpr = 64,  # Parallelize dense window loads in blocks of 64
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

        # ── Sparse compressed-block loop (identical to _fused_sparse_decode_kernel) ──
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
            u_scale = tl.load(pool_u_scale_ptr + pool_idx)
            u = u * u_scale

            s_anchor = tl.sum(q * ak) * INV_SCALE
            q_proj = tl.sum(q[None, :] * vk, axis=1) * INV_SCALE
            delta_scores = tl.sum(u * q_proj[None, :], axis=1) * scale
            s = s_anchor + delta_scores

            if HAS_RESIDUAL:
                for ri in range(MAX_RESIDUAL):
                    r_pos_k = tl.load(pool_res_pos_ptr + pool_idx * stride_res_pos_n + ri)
                    if r_pos_k >= 0:
                        rk = tl.load(pool_res_k_ptr + pool_idx * stride_res_k_n +
                                     ri * stride_res_k_s + h_kv * stride_res_k_h +
                                     offs_d * stride_res_k_d).to(tl.float32)
                        r_corr = tl.sum(q * rk) * INV_SCALE
                        s = tl.where(offs_s == r_pos_k, s + r_corr, s)

            if HAS_FACT:
                for fi in range(MAX_FACT):
                    fact_pos = tl.load(pool_fact_pos_ptr + pool_idx * stride_fact_pos_n + fi)
                    if fact_pos >= 0:
                        fact_k_ptrs = pool_fact_ak_ptr + pool_idx * stride_fact_ak_n + fi * stride_fact_ak_f + h_kv * stride_fact_ak_h + offs_d * stride_fact_ak_d
                        fact_k = tl.load(fact_k_ptrs).to(tl.float32)
                        fact_score = tl.sum(q * fact_k) * INV_SCALE
                        replace_mask = offs_s == fact_pos
                        s = tl.where(replace_mask, fact_score, s)

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

            O_fact_corr = tl.zeros([D], dtype=tl.float32)
            if HAS_FACT:
                for fi in range(MAX_FACT):
                    fact_pos = tl.load(pool_fact_pos_ptr + pool_idx * stride_fact_pos_n + fi)
                    if fact_pos >= 0:
                        replace_mask = offs_s == fact_pos
                        p_fact = tl.sum(tl.where(replace_mask, p_delta, 0.0), axis=0)
                        fact_v_ptrs = pool_fact_av_ptr + pool_idx * stride_fact_av_n + fi * stride_fact_av_f + h_kv * stride_fact_av_h + offs_d * stride_fact_av_d
                        fact_v = tl.load(fact_v_ptrs).to(tl.float32)
                        u_val_ptrs = pool_u_ptr + pool_idx * stride_u_n + fact_pos * stride_u_s + offs_r * stride_u_r
                        u_val = tl.load(u_val_ptrs).to(tl.float32) * u_scale
                        v_recon = tl.sum(u_val[:, None] * vv, axis=0) * scale + av
                        O_fact_corr += p_fact * (fact_v - v_recon)

            O_res_corr = tl.zeros([D], dtype=tl.float32)
            if HAS_RESIDUAL:
                for ri in range(MAX_RESIDUAL):
                    r_pos_v = tl.load(pool_res_pos_v_ptr + pool_idx * stride_res_pos_v_n + ri)
                    if r_pos_v >= 0:
                        p_at = tl.sum(tl.where(offs_s == r_pos_v, p_delta, 0.0), axis=0)
                        rv = tl.load(pool_res_v_ptr + pool_idx * stride_res_v_n +
                                     ri * stride_res_v_s + h_kv * stride_res_v_h +
                                     offs_d * stride_res_v_d).to(tl.float32)
                        O_res_corr += p_at * rv

            O_i = O_i * alpha + (p_anchor + p_delta_sum) * av + o_delta + O_fact_corr + O_res_corr
            m_i = m_new

        # ── Dense window token loop (NEW — fused into the same online softmax) ──
        # Each chunk processes a slice of dense tokens [dense_start, dense_end).
        # dense_K/dense_V are [H_kv, L_dense, D] pre-RoPE-rotated by the caller.
        # GQA: use h_kv for loading, same as the sparse branch above.
        if DENSE_PER_CHUNK > 0:
            dense_start = chunk_id * DENSE_PER_CHUNK
            dense_end = dense_start + DENSE_PER_CHUNK
            if dense_end > L_dense:
                dense_end = L_dense

            for t_start in range(dense_start, dense_end, BLOCK_SIZE_T):
                offs_t = t_start + tl.arange(0, BLOCK_SIZE_T)
                mask_t = offs_t < dense_end

                dk_ptrs = dense_k_ptr + h_kv * stride_dk_h + offs_t[:, None] * stride_dk_l + offs_d[None, :] * stride_dk_d
                dk = tl.load(dk_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)

                score = tl.sum(q[None, :] * dk, axis=1) * INV_SCALE
                score = tl.where(mask_t, score, -float("inf"))

                mb = tl.max(score, axis=0)
                m_new = tl.maximum(m_i, mb)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(score - m_new)
                p = tl.where(mask_t, p, 0.0)

                l_i = l_i * alpha + tl.sum(p, axis=0)

                dv_ptrs = dense_v_ptr + h_kv * stride_dv_h + offs_t[:, None] * stride_dv_l + offs_d[None, :] * stride_dv_d
                dv = tl.load(dv_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)

                O_i = O_i * alpha + tl.sum(p[:, None] * dv, axis=0)
                m_i = m_new

        # ── Write partial outputs (identical epilogue to _fused_sparse_decode_kernel) ──
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


def native_triton_sparse_attn_decode_combined(
    q:                    torch.Tensor,       # [1, H_q, 1, D]
    block_indices:        torch.Tensor,       # [N]  active compressed-block indices
    pool:                 object,             # NativeBlockPool
    dense_k:              Optional[torch.Tensor],  # [1, H_kv, L_dense, D] pre-RoPE-rotated; None if no dense
    dense_v:              Optional[torch.Tensor],  # [1, H_kv, L_dense, D]
    num_key_value_groups: int,
    R:                    int = 16,
    S_MAX:                int = 64,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Single-dispatch fused attention over both compressed blocks and dense window tokens.

    Replaces the 3-step pattern used in native_triton_sparse_attn_decode:
        sparse Triton → dense F.sdpa → Python LSE-merge
    with one Triton kernel that processes both token classes in the same online softmax.

    Falls back to native_triton_sparse_attn_decode (which does its own dense merge)
    on any error or if HAS_TRITON is False.

    Returns: [1, H_q, 1, D] in q.dtype.
    """
    if not HAS_TRITON:
        # MPS / CPU: fall back to the existing separate-path wrapper
        # (dense is handled by its own Python LSE merge inside that function)
        return native_triton_sparse_attn_decode(
            q, block_indices, pool, [], dense_k, dense_v,
            num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin,
        )

    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1

    N = block_indices.shape[0] if block_indices is not None else 0
    has_dense = dense_k is not None and dense_k.shape[2] > 0
    L_dense = dense_k.shape[2] if has_dense else 0

    # If nothing to attend to, return zeros
    if N == 0 and L_dense == 0:
        return torch.zeros((1, H_q, 1, D), device=q.device, dtype=q.dtype)

    # If there are no compressed blocks, just run dense SDPA (fast path)
    if N == 0 and has_dense:
        H_kv = dense_k.shape[1]
        n_rep = H_q // H_kv
        q_sq = q[0, :, 0, :].float()
        dk = dense_k[0].float()  # [H_kv, L_dense, D]
        dv = dense_v[0].float()
        q_r = q_sq.view(H_kv, n_rep, D)
        s = torch.bmm(q_r, dk.permute(0, 2, 1)).view(H_q, L_dense) / math.sqrt(D)
        w = torch.softmax(s, dim=-1)
        w_r = w.view(H_kv, n_rep, L_dense)
        out = torch.bmm(w_r, dv).view(H_q, D)
        return out.unsqueeze(0).unsqueeze(2).to(q.dtype)

    try:
        inv_scale = 1.0 / math.sqrt(D)
        q_sq = q[0, :, 0, :]  # [H_q, D]

        D_pad   = triton.next_power_of_2(D)
        R_pad   = triton.next_power_of_2(R)
        S_pad   = triton.next_power_of_2(S_MAX)

        BLOCKS_PER_CHUNK = 16
        num_chunks_sparse = max(1, (N + BLOCKS_PER_CHUNK - 1) // BLOCKS_PER_CHUNK)

        # Distribute dense tokens across the same chunk grid so each program sees
        # a balanced slice.  When L_dense == 0, DENSE_PER_CHUNK = 0 → loop elided.
        if L_dense > 0:
            DENSE_PER_CHUNK = max(1, (L_dense + num_chunks_sparse - 1) // num_chunks_sparse)
            num_chunks = max(num_chunks_sparse,
                             (L_dense + DENSE_PER_CHUNK - 1) // DENSE_PER_CHUNK)
        else:
            DENSE_PER_CHUNK = 0
            num_chunks = num_chunks_sparse

        grid = (H_q, num_chunks)

        # Allocate output/workspace buffers
        if num_chunks > 1:
            cache_key = (H_q, num_chunks, D_pad, q.device)
            if not hasattr(native_triton_sparse_attn_decode_combined, "_ws_cache"):
                native_triton_sparse_attn_decode_combined._ws_cache = {}
            ws = native_triton_sparse_attn_decode_combined._ws_cache.get(cache_key)
            if ws is None:
                ow = torch.empty((H_q, num_chunks, D_pad), device=q.device, dtype=torch.float32)
                mw = torch.empty((H_q, num_chunks),        device=q.device, dtype=torch.float32)
                lw = torch.empty((H_q, num_chunks),        device=q.device, dtype=torch.float32)
                ws = (ow, mw, lw)
                native_triton_sparse_attn_decode_combined._ws_cache[cache_key] = ws
            out_workspace, m_workspace, l_workspace = ws
            out   = torch.empty((H_q, D), device=q.device, dtype=torch.float32)
            m_out = torch.empty((H_q,),   device=q.device, dtype=torch.float32)
            l_out = torch.empty((H_q,),   device=q.device, dtype=torch.float32)
        else:
            out   = torch.empty((H_q, D), device=q.device, dtype=torch.float32)
            m_out = torch.empty((H_q,),   device=q.device, dtype=torch.float32)
            l_out = torch.empty((H_q,),   device=q.device, dtype=torch.float32)
            out_workspace, m_workspace, l_workspace = out, m_out, l_out

        # ── Gather + rotate the N routed rows (identical semantics to
        # native_triton_sparse_attn_decode; see F2 helper). Issue 1 fix included:
        # stratified U is pre-reconstructed before dispatch for CUDA/MPS parity.
        pool_for_kernel, _used_strat = _build_stratified_U_for_triton(pool, block_indices)
        g = _gather_routed_blocks_for_kernel(
            pool_for_kernel, block_indices, anchor_indices, cos, sin)

        # ── Dense window tensors ──
        # Caller provides pre-RoPE-rotated dense_k/dense_v as [1, H_kv, L_dense, D].
        # We need [H_kv, L_dense, D] contiguous for the kernel.
        if has_dense:
            dk_t = dense_k[0].contiguous().to(torch.float32)  # [H_kv, L_dense, D]
            dv_t = dense_v[0].contiguous().to(torch.float32)
        else:
            dk_t = torch.empty((1, 0, D_pad), device=q.device, dtype=torch.float32)
            dv_t = torch.empty((1, 0, D_pad), device=q.device, dtype=torch.float32)

        # ── Kernel launch ──
        _fused_decode_combined_kernel[grid](
            q_sq, g["idx"], g["anchors_K"], g["anchors_V"], g["V_K"], g["V_V"],
            g["U"], g["U_scale"], g["scales"], g["seq_lens"],
            g["res_k"], g["res_v"], g["res_pos"], g["res_pos_v"], g["res_n"],
            g["fact_pos"], g["fact_ak"], g["fact_av"],
            dk_t, dv_t, L_dense,
            dk_t.stride(0), dk_t.stride(1), dk_t.stride(2),
            dv_t.stride(0), dv_t.stride(1), dv_t.stride(2),
            out_workspace, m_workspace, l_workspace,
            q_sq.stride(0), q_sq.stride(1),
            g["anchors_K"].stride(0), g["anchors_K"].stride(1), g["anchors_K"].stride(2),
            g["anchors_V"].stride(0), g["anchors_V"].stride(1), g["anchors_V"].stride(2),
            g["V_K"].stride(0), g["V_K"].stride(1), g["V_K"].stride(2), g["V_K"].stride(3),
            g["V_V"].stride(0), g["V_V"].stride(1), g["V_V"].stride(2), g["V_V"].stride(3),
            g["U"].stride(0), g["U"].stride(1), g["U"].stride(2),
            g["res_k"].stride(0), g["res_k"].stride(1), g["res_k"].stride(2), g["res_k"].stride(3),
            g["res_v"].stride(0), g["res_v"].stride(1), g["res_v"].stride(2), g["res_v"].stride(3),
            g["res_pos"].stride(0), g["res_pos_v"].stride(0),
            g["fact_pos"].stride(0),
            g["fact_ak"].stride(0), g["fact_ak"].stride(1), g["fact_ak"].stride(2), g["fact_ak"].stride(3),
            g["fact_av"].stride(0), g["fact_av"].stride(1), g["fact_av"].stride(2), g["fact_av"].stride(3),
            out_workspace.stride(0), out_workspace.stride(1),
            N, H_q, g["anchors_K"].shape[1], num_key_value_groups, D_pad,
            R_pad, S_pad, inv_scale, BLOCKS_PER_CHUNK, num_chunks,
            MAX_RESIDUAL=g["max_res_pad"], MAX_FACT=g["max_fact"],
            HAS_RESIDUAL=g["has_res"], HAS_FACT=g["has_fact"],
            DENSE_PER_CHUNK=DENSE_PER_CHUNK,
        )

        if num_chunks > 1:
            # OPT-E: dispatches sequential vs. parallel tree reduction based on num_chunks
            _dispatch_reduction(
                out_workspace, m_workspace, l_workspace, out, m_out, l_out,
                num_chunks, D_pad, H_q,
            )

        if not getattr(native_triton_sparse_attn_decode_combined, "_logged", False):
            print("[DiffKV] Triton fused-decode COMBINED path ACTIVE (CUDA). "
                  f"N_sparse={N}, L_dense={L_dense}")
            native_triton_sparse_attn_decode_combined._logged = True

        return out.unsqueeze(0).unsqueeze(2).to(q.dtype)

    except Exception as e:
        if os.environ.get("DIFFKV_TRITON_STRICT") == "1":
            raise
        if not hasattr(native_triton_sparse_attn_decode_combined, "_fallback_warned"):
            print(f"[DiffKV] WARNING: combined Triton kernel failed ({e}). "
                  "Falling back to native_triton_sparse_attn_decode.")
            native_triton_sparse_attn_decode_combined._fallback_warned = True
        return native_triton_sparse_attn_decode(
            q, block_indices, pool, [], dense_k, dense_v,
            num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin,
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
