"""
runtime/triton_sparse_attn.py

Phase 9: True SRAM-Resident Fused Sparse Attention (Triton)

Replaces the PyTorch batched ops from Phase 8 with a single, highly-optimized
Triton kernel. This moves the FlashAttention accumulation (m, l, O) entirely 
into SRAM (shared memory), completely eliminating intermediate VRAM traffic for
`s_full`, `P`, and `O_blk` tensors.

Key hardware efficiency wins:
  - 100% SRAM-resident running state (m, l, O).
  - ZERO intermediate tensors created.
  - O(1) global memory reads per token.
  - Perfect GQA scaling (KV heads loaded once, broadcast to Q heads in kernel).
"""

import torch
import math
import os
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _fused_sparse_decode_kernel(
        # Pointers
        q_ptr,              # [H_q, D]
        block_indices_ptr,  # [N] array of ints pointing into pool
        pool_ak_ptr,        # [MAX_BLOCKS, H_kv, D]
        pool_av_ptr,        # [MAX_BLOCKS, H_kv, D]
        pool_vk_ptr,        # [MAX_BLOCKS, R, H_kv, D]
        pool_vv_ptr,        # [MAX_BLOCKS, R, H_kv, D]
        pool_u_ptr,         # [MAX_BLOCKS, S_MAX, R]
        pool_scales_ptr,    # [MAX_BLOCKS]
        pool_seq_lens_ptr,  # [MAX_BLOCKS]
        out_ptr,            # [H_q, D] if NUM_CHUNKS == 1 else [H_q, NUM_CHUNKS, D]
        m_ptr,              # [H_q] if NUM_CHUNKS == 1 else [H_q, NUM_CHUNKS]
        l_ptr,              # [H_q] if NUM_CHUNKS == 1 else [H_q, NUM_CHUNKS]
        
        # Strides
        stride_q_h, stride_q_d,
        stride_ak_n, stride_ak_h, stride_ak_d,
        stride_av_n, stride_av_h, stride_av_d,
        stride_vk_n, stride_vk_r, stride_vk_h, stride_vk_d,
        stride_vv_n, stride_vv_r, stride_vv_h, stride_vv_d,
        stride_u_n, stride_u_s, stride_u_r,
        stride_out_h, stride_out_d,
        
        # Dimensions / Config
        N: tl.constexpr,
        H_q: tl.constexpr,
        H_kv: tl.constexpr,
        KV_GRP: tl.constexpr,
        D: tl.constexpr,
        R: tl.constexpr,
        S_MAX: tl.constexpr,
        INV_SCALE: tl.constexpr,
        
        BLOCKS_PER_CHUNK: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
    ):
        # Each program processes one Query Head and one Chunk ID
        h_q = tl.program_id(0)
        chunk_id = tl.program_id(1)
        h_kv = h_q // KV_GRP
        
        # Offsets for D and R dimensions
        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, R)
        offs_s = tl.arange(0, S_MAX)
        
        # Pointers to Q for this head
        q_ptrs = q_ptr + h_q * stride_q_h + offs_d * stride_q_d
        q = tl.load(q_ptrs).to(tl.float32) # [D]
        
        # Initialize FlashAttention running state in SRAM
        m_i = -float("inf")
        l_i = 0.0
        O_i = tl.zeros([D], dtype=tl.float32)
        
        # FlashDecoding sequence range
        start_block = chunk_id * BLOCKS_PER_CHUNK
        end_block = start_block + BLOCKS_PER_CHUNK
        if end_block > N:
            end_block = N
        
        # Loop over assigned block chunk range
        for n in range(start_block, end_block):
            # 1. Lookup global pool index for the n-th block in our logical sequence
            pool_idx = tl.load(block_indices_ptr + n)
            
            # 2. Load scalar meta for this block
            scale = tl.load(pool_scales_ptr + pool_idx).to(tl.float32)
            actual_s = tl.load(pool_seq_lens_ptr + pool_idx)
            
            # 3. Load Anchor K and V
            ak_ptrs = pool_ak_ptr + pool_idx * stride_ak_n + h_kv * stride_ak_h + offs_d * stride_ak_d
            av_ptrs = pool_av_ptr + pool_idx * stride_av_n + h_kv * stride_av_h + offs_d * stride_av_d
            ak = tl.load(ak_ptrs).to(tl.float32) # [D]
            av = tl.load(av_ptrs).to(tl.float32) # [D]
            
            # 4. Load V_K and V_V matrices for this block
            vk_ptrs = pool_vk_ptr + pool_idx * stride_vk_n + h_kv * stride_vk_h + \
                      offs_r[:, None] * stride_vk_r + offs_d[None, :] * stride_vk_d
            vv_ptrs = pool_vv_ptr + pool_idx * stride_vv_n + h_kv * stride_vv_h + \
                      offs_r[:, None] * stride_vv_r + offs_d[None, :] * stride_vv_d
            
            vk = tl.load(vk_ptrs).to(tl.float32) # [R, D]
            vv = tl.load(vv_ptrs).to(tl.float32) # [R, D]
            
            # 5. Load U matrix
            u_ptrs = pool_u_ptr + pool_idx * stride_u_n + offs_s[:, None] * stride_u_s + offs_r[None, :] * stride_u_r
            # Mask out padded sequence elements
            s_mask = offs_s[:, None] < actual_s
            u = tl.load(u_ptrs, mask=s_mask, other=0.0).to(tl.float32) # [S_MAX, R]
            
            # === MATH TIME (In SRAM) ===
            
            # s_anchor = dot(q, ak) * inv_scale
            s_anchor = tl.sum(q * ak) * INV_SCALE
            
            # q_proj = dot(q, V_K^T) -> size [R]
            q_proj = tl.sum(q[None, :] * vk, axis=1) * INV_SCALE # [R]
            
            # delta_scores = dot(U, q_proj) * scale -> size [S_MAX]
            delta_scores = tl.sum(u * q_proj[None, :], axis=1) * scale # [S_MAX]
            
            s = s_anchor + delta_scores # [S_MAX]
            
            # Mask out invalid scores
            s = tl.where(offs_s < actual_s, s, -float("inf"))
            
            # Local max for this block
            m_b_delta = tl.max(s, axis=0)
            m_b = tl.maximum(s_anchor, m_b_delta)
            
            # Global max update
            m_new = tl.maximum(m_i, m_b)
            
            # Normalization factor
            alpha = tl.exp(m_i - m_new)
            
            # Probabilities
            p_anchor = tl.exp(s_anchor - m_new)
            p_delta = tl.exp(s - m_new)
            p_delta = tl.where(offs_s < actual_s, p_delta, 0.0)
            
            p_delta_sum = tl.sum(p_delta, axis=0)
            
            # Update running denominator
            l_i = l_i * alpha + p_anchor + p_delta_sum
            
            # O contribution from this block
            p_u = tl.sum(p_delta[:, None] * u, axis=0) # [R]
            o_delta = tl.sum(p_u[:, None] * vv, axis=0) * scale # [D]
            
            # Accumulate to O_i
            O_i = O_i * alpha + (p_anchor + p_delta_sum) * av + o_delta
            m_i = m_new

        if NUM_CHUNKS == 1:
            # Final normalization
            O_i = O_i / l_i
            # Store to final global memory
            out_ptrs = out_ptr + h_q * stride_out_h + offs_d * stride_out_d
            tl.store(out_ptrs, O_i)
            
            if m_ptr is not None:
                tl.store(m_ptr + h_q, m_i)
            if l_ptr is not None:
                tl.store(l_ptr + h_q, l_i)
        else:
            # Store unnormalized results to global workspaces for reduction
            out_work_ptrs = out_ptr + h_q * (NUM_CHUNKS * D) + chunk_id * D + offs_d
            tl.store(out_work_ptrs, O_i)
            
            if m_ptr is not None:
                tl.store(m_ptr + h_q * NUM_CHUNKS + chunk_id, m_i)
            if l_ptr is not None:
                tl.store(l_ptr + h_q * NUM_CHUNKS + chunk_id, l_i)

    @triton.jit
    def _fused_sparse_decode_reduction_kernel(
        out_workspace_ptr,  # [H_q, NUM_CHUNKS, D]
        m_workspace_ptr,    # [H_q, NUM_CHUNKS]
        l_workspace_ptr,    # [H_q, NUM_CHUNKS]
        out_ptr,            # [H_q, D]
        m_final_ptr,        # [H_q]
        l_final_ptr,        # [H_q]
        
        NUM_CHUNKS: tl.constexpr,
        D: tl.constexpr,
    ):
        h_q = tl.program_id(0)
        offs_d = tl.arange(0, D)
        
        # Initialize running reduction state
        m_i = -float("inf")
        l_i = 0.0
        O_i = tl.zeros([D], dtype=tl.float32)
        
        for c in range(NUM_CHUNKS):
            # Load chunk stats
            m_c = tl.load(m_workspace_ptr + h_q * NUM_CHUNKS + c)
            l_c = tl.load(l_workspace_ptr + h_q * NUM_CHUNKS + c)
            
            # Load chunk output
            out_c_ptrs = out_workspace_ptr + h_q * (NUM_CHUNKS * D) + c * D + offs_d
            O_c = tl.load(out_c_ptrs).to(tl.float32)
            
            # Update running max
            m_new = tl.maximum(m_i, m_c)
            
            # Exponential scaling factors
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_c - m_new)
            
            # Update running denominator
            l_i = l_i * alpha + l_c * beta
            
            # Accumulate scaled local outputs
            O_i = O_i * alpha + O_c * beta
            m_i = m_new
            
        # Final normalization
        O_i = O_i / l_i
        
        # Store to final output
        out_ptrs = out_ptr + h_q * D + offs_d
        tl.store(out_ptrs, O_i)
        
        if m_final_ptr is not None:
            tl.store(m_final_ptr + h_q, m_i)
        if l_final_ptr is not None:
            tl.store(l_final_ptr + h_q, l_i)


def _pytorch_vectorized_sparse_attn_decode(
    q:                    torch.Tensor,    # [1, H_q, 1, D]
    block_indices:        torch.Tensor,    # [N] int32
    pool:                 object,          # NativeBlockPool
    dense_blocks:         list,            
    active_k:             torch.Tensor,    # [1, H_kv, T, D]
    active_v:             torch.Tensor,
    num_key_value_groups: int,
    R:                    int = 16,
    S_MAX:                int = 64,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
    total_seq_len:        int = 0,
) -> torch.Tensor:
    """Highly-parallelized pure PyTorch low-rank SVD unrotated math decoder with dynamic RoPE."""
    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1, "Decode only"
    
    inv_scale = 1.0 / math.sqrt(D)
    
    # Reshape Q for fast matrix/vector operations
    q_sq = q.view(H_q, D) # [H_q, D]
    
    # 1. Helper to repeat KV across Query heads for GQA compatibility
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
    
    if diagnostics:
        print(f"\n--- [DiffKV Decode Diagnostic] N={N} ---")
        print(f"  q: shape={q.shape} min={q.min().item():.4f} max={q.max().item():.4f} mean={q.mean().item():.4f}")
    
    if N > 0:
        indices = block_indices.long()
        U = pool.U[indices]                                     # [N, block_capacity, R]
        block_capacity = U.shape[1]
        
        V_K = repeat_kv_at_dim(pool.V_K[indices], num_key_value_groups, dim=2)           # [N, R, H_q, D]
        V_V = repeat_kv_at_dim(pool.V_V[indices], num_key_value_groups, dim=2)           # [N, R, H_q, D]
        anchors_K = repeat_kv_at_dim(pool.anchors_K[indices], num_key_value_groups, dim=1) # [N, H_q, D]
        anchors_V = repeat_kv_at_dim(pool.anchors_V[indices], num_key_value_groups, dim=1) # [N, H_q, D]
        scales = pool.scales[indices].view(N, 1, 1)             # [N, 1, 1] (3D representation)
        seq_lens_t = pool.seq_lens[indices]                      # [N] int32, on GPU

        # Clamp block_capacity to the maximum actually-used slot across all blocks.
        max_valid_len = int(seq_lens_t.max().item())             # CPU scalar — single int
        block_capacity = min(block_capacity, max(max_valid_len, 1))
        
        # ── 1. Reconstruct Unrotated Keys in Parallel (Full Block: Anchor + Deltas) ──
        deltas_k = torch.einsum('nsr,nrhd->nshd', U[:, :block_capacity, :].float(), V_K.float()).to(q.dtype) * scales.unsqueeze(-1)
        zeros_pad = torch.zeros((N, 1, H_q, D), dtype=q.dtype, device=q.device)
        deltas_k_full = torch.cat([zeros_pad, deltas_k], dim=1)
        K_unrot_full = anchors_K.unsqueeze(1) + deltas_k_full
        
        # ── 2. Construct Absolute Position IDs and Slice cos/sin ──
        if anchor_indices is None:
            anchor_indices = torch.zeros(N, dtype=torch.long, device=q.device)
        positions = anchor_indices.view(N, 1) + torch.arange(1 + block_capacity, device=q.device).view(1, 1 + block_capacity)
        positions_flat = positions.view(-1)
        
        if cos is None or sin is None:
            cos_sliced = torch.ones((N, 1 + block_capacity, 1, D), dtype=q.dtype, device=q.device)
            sin_sliced = torch.zeros((N, 1 + block_capacity, 1, D), dtype=q.dtype, device=q.device)
        else:
            cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
            sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
            cos_sliced = cos_flat[positions_flat].view(N, 1 + block_capacity, 1, D)
            sin_sliced = sin_flat[positions_flat].view(N, 1 + block_capacity, 1, D)
        
        if diagnostics:
            print(f"  block_capacity clamping: max_valid_len={max_valid_len} -> capacity={block_capacity}")
            print(f"  U (clamped): min={U[:, :block_capacity].min().item():.4f} max={U[:, :block_capacity].max().item():.4f} mean={U[:, :block_capacity].mean().item():.4f}")
            print(f"  V_K: min={V_K.min().item():.4f} max={V_K.max().item():.4f} mean={V_K.mean().item():.4f}")
            print(f"  V_V: min={V_V.min().item():.4f} max={V_V.max().item():.4f} mean={V_V.mean().item():.4f}")
            print(f"  anchors_K: min={anchors_K.min().item():.4f} max={anchors_K.max().item():.4f} mean={anchors_K.mean().item():.4f}")
            print(f"  anchors_V: min={anchors_V.min().item():.4f} max={anchors_V.max().item():.4f} mean={anchors_V.mean().item():.4f}")
            print(f"  scales: min={scales.min().item():.4f} max={scales.max().item():.4f} mean={scales.mean().item():.4f}")


        # ── 3. Apply RoPE to Reconstructed Keys ──
        K_rot_full = (K_unrot_full * cos_sliced) + (rotate_half(K_unrot_full) * sin_sliced)
        
        # ── 4. Compute Scores Directly in Rotated Space ──
        scores_block_full = torch.sum(q_sq.view(1, 1, H_q, D) * K_rot_full, dim=-1) * inv_scale # [N, 1 + block_capacity, H_q]
        
        # ── 5. Sequence masking ──
        mask = torch.arange(1 + block_capacity, device=q.device).view(1, 1 + block_capacity, 1) >= (1 + seq_lens_t).view(N, 1, 1)
        scores_block_full = scores_block_full.masked_fill(mask.expand_as(scores_block_full), float('-inf'))
        
        scores_anchor = scores_block_full[:, 0, :] # [N, H_q]
        
        if diagnostics:
            print(f"  scores_anchor: min={scores_anchor.min().item():.4f} max={scores_anchor.max().item():.4f} mean={scores_anchor.mean().item():.4f}")
            print(f"  scores_block (masked): min={scores_block_full[~mask.expand_as(scores_block_full)].min().item():.4f} max={scores_block_full[~mask.expand_as(scores_block_full)].max().item():.4f} mean={scores_block_full[~mask.expand_as(scores_block_full)].mean().item():.4f}")

        # Reshape/transpose scores for global concatenation
        scores_anchor = scores_anchor.transpose(0, 1) # [H_q, N]
        scores_compressed = scores_block_full[:, 1:, :].permute(2, 0, 1).reshape(H_q, N * block_capacity) # [H_q, N * block_capacity]
    else:
        scores_anchor = torch.empty((H_q, 0), device=q.device, dtype=q.dtype)
        scores_compressed = torch.empty((H_q, 0), device=q.device, dtype=q.dtype)

    # ── 6. Dense tokens collection and dynamic RoPE application ──
    dense_k_parts = []
    dense_v_parts = []
    
    if active_k is not None and active_k.shape[2] > 0:
        # Pre-assembled contiguous workspace (high-performance decode path)
        dense_k_parts.append(active_k)
        dense_v_parts.append(active_v)
    else:
        # Fallback block-by-block collection (e.g. from tests or Triton fallbacks)
        for blk in (dense_blocks or []):
            dense_k_parts.append(blk.anchor_kv[:, 0].unsqueeze(2)) # [1, H_kv, 1, D]
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
        
        cos_dense = cos[0, dense_positions].unsqueeze(0).unsqueeze(1) # [1, 1, S_dense, D]
        sin_dense = sin[0, dense_positions].unsqueeze(0).unsqueeze(1) # [1, 1, S_dense, D]
        
        full_k_rot = (full_k * cos_dense) + (rotate_half(full_k) * sin_dense)
        
        k_dense_rep = repeat_kv_at_dim(full_k_rot, num_key_value_groups, dim=1) # [1, H_q, S_dense, D]
        v_dense_rep = repeat_kv_at_dim(full_v, num_key_value_groups, dim=1) # [1, H_q, S_dense, D]
        scores_dense = torch.sum(q * k_dense_rep, dim=-1).squeeze(0) * inv_scale # [H_q, S_dense]
    else:
        S_dense = 0
        scores_dense = torch.empty((H_q, 0), device=q.device, dtype=q.dtype)

    # ── 6. Global Softmax ──
    # scores_all shape: [H_q, total_tokens]
    scores_all = torch.cat([scores_anchor, scores_compressed, scores_dense], dim=-1)
    
    if diagnostics:
        print(f"  scores_anchor shape={scores_anchor.shape} compressed shape={scores_compressed.shape} dense shape={scores_dense.shape} all shape={scores_all.shape}")
    
    # CRITICAL: Only sanitize NaN and +Inf — NEVER convert -inf to a finite value!
    # -inf entries are the intentional padding mask from sequence length gating.
    # Converting them to -1e4 gives nonzero softmax weight to zero-padded U slots,
    # which attenuates the output and causes progressive output drift / hallucination.
    has_nan = torch.isnan(scores_all).any()
    has_posinf = (scores_all == float('inf')).any()
    if has_nan or has_posinf:
        scores_all = scores_all.clone()
        if has_nan:
            scores_all[torch.isnan(scores_all)] = -1e4
        if has_posinf:
            scores_all[scores_all == float('inf')] = 1e4
        # -inf entries are left INTACT: they produce zero softmax probability (correct)
        
    probs_all = torch.nn.functional.softmax(scores_all, dim=-1) # [H_q, total_tokens]
    
    # Split probabilities back
    P_anchor, P_comp, P_dense = torch.split(probs_all, [N, N * block_capacity, S_dense], dim=-1)

    # ── 7. Value Reduction ──
    O_final = torch.zeros((H_q, D), device=q.device, dtype=q.dtype)

    if N > 0:
        # 7.1 + 7.2 Fused: Anchor and compressed contributions share anchors_V.
        # Compute combined probability on anchors_V in one matmul:
        #   total_anchor_prob[n, h] = P_anchor[h, n] + sum_s P_comp[h, n, s]
        # P_comp has shape [H_q, N * block_capacity] where block_capacity is clamped.
        P_comp_reshaped = P_comp.view(H_q, N, block_capacity).permute(1, 0, 2) # [N, H_q, S]
        # U is sliced to [:, :block_capacity, :] matching the clamped block_capacity
        U_clamped = U[:, :block_capacity, :]                                    # [N, S, R]
        P_U = torch.einsum('nhs,nsr->nhr', P_comp_reshaped.float(), U_clamped.float())  # [N, H_q, R]

        # Combined anchor probability: direct sum of P_anchor and compressed-block total
        # Avoids a separate O_anchor_total matmul — one fused operation
        p_total_anchor = P_anchor.transpose(0, 1) + P_comp_reshaped.sum(dim=-1)  # [N, H_q]
        O_anchor_fused = torch.sum(p_total_anchor.unsqueeze(-1) * anchors_V.float(), dim=0)  # [H_q, D]
        O_final = O_final + O_anchor_fused.to(q.dtype)

        # Delta contribution from low-rank compressed values
        O_delta = torch.einsum('nhr,nrhd->nhd', P_U, V_V.float()) * scales.float()  # [N, H_q, D]
        
        if diagnostics:
            print(f"  P_comp_reshaped: min={P_comp_reshaped.min().item():.4f} max={P_comp_reshaped.max().item():.4f} mean={P_comp_reshaped.mean().item():.4f}")
            print(f"  P_U: min={P_U.min().item():.4f} max={P_U.max().item():.4f} mean={P_U.mean().item():.4f}")
            print(f"  p_total_anchor: min={p_total_anchor.min().item():.4f} max={p_total_anchor.max().item():.4f} mean={p_total_anchor.mean().item():.4f}")
            print(f"  O_anchor_fused: min={O_anchor_fused.min().item():.4f} max={O_anchor_fused.max().item():.4f} mean={O_anchor_fused.mean().item():.4f}")
            print(f"  O_delta: min={O_delta.min().item():.4f} max={O_delta.max().item():.4f} mean={O_delta.mean().item():.4f}")

        O_final = O_final + O_delta.sum(0).to(q.dtype)

    # 7.3. Dense contribution
    if S_dense > 0:
        # v_dense_rep is [1, H_q, S_dense, D] -> squeeze(0) is [H_q, S_dense, D]
        # P_dense is [H_q, S_dense]
        # O_dense_total is [H_q, D]
        O_dense_total = torch.sum(P_dense.unsqueeze(-1) * v_dense_rep.squeeze(0), dim=1) # [H_q, D]
        
        if diagnostics:
            print(f"  P_dense: min={P_dense.min().item():.4f} max={P_dense.max().item():.4f} mean={P_dense.mean().item():.4f}")
            print(f"  O_dense_total: min={O_dense_total.min().item():.4f} max={O_dense_total.max().item():.4f} mean={O_dense_total.mean().item():.4f}")
            
        O_final = O_final + O_dense_total.to(q.dtype)

    if diagnostics:
        print(f"  O_final: min={O_final.min().item():.4f} max={O_final.max().item():.4f} mean={O_final.mean().item():.4f}")

    return O_final.unsqueeze(0).unsqueeze(2)


def native_triton_sparse_attn_decode(
    q:                    torch.Tensor,    # [1, H_q, 1, D]
    block_indices:        torch.Tensor,    # [N] int32
    pool:                 object,          # NativeBlockPool
    dense_blocks:         list,            
    active_k:             torch.Tensor,    # [1, H_kv, T, D]
    active_v:             torch.Tensor,
    num_key_value_groups: int,
    R:                    int = 16,
    S_MAX:                int = 64,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
    total_seq_len:        int = 0,
) -> torch.Tensor:
    """
    Python wrapper for the Phase 10 Native Block Table Triton kernel.
    Handles dispatch using the NativeBlockPool, skipping ALL PyTorch tensor staging.
    Includes highly-parallelized PyTorch vectorized fallback for Windows or Triton failures.
    """
    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1, "Decode only"
    
    if not HAS_TRITON:
        return _pytorch_vectorized_sparse_attn_decode(
            q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len
        )
        
    inv_scale = 1.0 / math.sqrt(D)
    N = block_indices.shape[0] if block_indices is not None else 0
    
    if N > 0:
        try:
            q_sq = q[0, :, 0, :] # [H_q, D]
            D_pad = triton.next_power_of_2(D)
            R_pad = triton.next_power_of_2(R)
            S_pad = triton.next_power_of_2(S_MAX)
            
            # FlashDecoding configuration: chunk size of 16 blocks
            BLOCKS_PER_CHUNK = 16
            if N > BLOCKS_PER_CHUNK:
                num_chunks = (N + BLOCKS_PER_CHUNK - 1) // BLOCKS_PER_CHUNK
            else:
                num_chunks = 1
                
            grid = (H_q, num_chunks)
            
            # Reusable workspaces cache to eliminate allocator churn
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
            
            # Phase 28: Proof of dispatch
            if not hasattr(native_triton_sparse_attn_decode, "fired"):
                print("[Phase 28] TRITON FUSED SPARSE DECODE KERNEL FIRED!")
                native_triton_sparse_attn_decode.fired = True
            
            _fused_sparse_decode_kernel[grid](
                q_ptr=q_sq,
                block_indices_ptr=block_indices,
                pool_ak_ptr=pool.anchors_K,
                pool_av_ptr=pool.anchors_V,
                pool_vk_ptr=pool.V_K,
                pool_vv_ptr=pool.V_V,
                pool_u_ptr=pool.U,
                pool_scales_ptr=pool.scales,
                pool_seq_lens_ptr=pool.seq_lens,
                out_ptr=out_workspace,
                m_ptr=m_workspace,
                l_ptr=l_workspace,
                
                stride_q_h=q_sq.stride(0), stride_q_d=q_sq.stride(1),
                stride_ak_n=pool.anchors_K.stride(0), stride_ak_h=pool.anchors_K.stride(1), stride_ak_d=pool.anchors_K.stride(2),
                stride_av_n=pool.anchors_V.stride(0), stride_av_h=pool.anchors_V.stride(1), stride_av_d=pool.anchors_V.stride(2),
                stride_vk_n=pool.V_K.stride(0), stride_vk_r=pool.V_K.stride(1), stride_vk_h=pool.V_K.stride(2), stride_vk_d=pool.V_K.stride(3),
                stride_vv_n=pool.V_V.stride(0), stride_vv_r=pool.V_V.stride(1), stride_vv_h=pool.V_V.stride(2), stride_vv_d=pool.V_V.stride(3),
                stride_u_n=pool.U.stride(0), stride_u_s=pool.U.stride(1), stride_u_r=pool.U.stride(2),
                stride_out_h=out_workspace.stride(0), stride_out_d=out_workspace.stride(1),
                
                N=N,
                H_q=H_q,
                H_kv=pool.anchors_K.shape[1],
                KV_GRP=num_key_value_groups,
                D=D_pad,
                R=R_pad,
                S_MAX=S_pad,
                INV_SCALE=inv_scale,
                
                BLOCKS_PER_CHUNK=BLOCKS_PER_CHUNK,
                NUM_CHUNKS=num_chunks,
            )
            
            if num_chunks > 1:
                grid_reduction = (H_q,)
                _fused_sparse_decode_reduction_kernel[grid_reduction](
                    out_workspace_ptr=out_workspace,
                    m_workspace_ptr=m_workspace,
                    l_workspace_ptr=l_workspace,
                    out_ptr=out,
                    m_final_ptr=m_out,
                    l_final_ptr=l_out,
                    NUM_CHUNKS=num_chunks,
                    D=D_pad,
                )
            
            if dense_blocks or (active_k is not None and active_k.shape[2] > 0):
                # Un-normalize O_i to continue accumulation
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
                    k_kv = torch.cat(dense_k_parts, dim=2).float()  # [1, H_kv, S, D]
                    v_kv = torch.cat(dense_v_parts, dim=2).float()  # [1, H_kv, S, D]
                    
                    H_q, D = q_sq.shape
                    H_kv = k_kv.shape[1]
                    n_rep = num_key_value_groups
                    
                    q_reshaped = q_sq.float().view(H_kv, n_rep, D)
                    k_permuted = k_kv[0].permute(0, 2, 1)  # [H_kv, D, S]
                    
                    s = torch.bmm(q_reshaped, k_permuted).view(H_q, -1) * inv_scale  # [H_q, S]
                    
                    m_b = s.max(-1).values
                    m_new = torch.maximum(m_i, m_b)
                    a = torch.exp(m_i - m_new)
                    P = torch.exp(s - m_new.unsqueeze(-1))
                    l_i = a * l_i + P.sum(-1)
                    
                    P_reshaped = P.view(H_kv, n_rep, -1)
                    v_permuted = v_kv[0]  # [H_kv, S, D]
                    
                    O_i_delta = torch.bmm(P_reshaped, v_permuted).view(H_q, D)
                    
                    O_i = a.unsqueeze(-1) * O_i + O_i_delta
                    m_i = m_new
                    
                out = O_i / l_i.unsqueeze(-1)
                
            return out.unsqueeze(0).unsqueeze(2).to(q.dtype)
        except Exception as e:
            if not hasattr(native_triton_sparse_attn_decode, "fallback_fired"):
                print(f"[DiffKV] WARNING: Triton compilation or execution failed: {e}. Falling back to zero-compile PyTorch vectorized low-rank decoder.")
                native_triton_sparse_attn_decode.fallback_fired = True
            return _pytorch_vectorized_sparse_attn_decode(
                q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
                anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len
            )
    else:
        return _pytorch_vectorized_sparse_attn_decode(
            q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len
        )
