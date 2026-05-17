import os
import sys
import torch
import triton
import triton.language as tl
from pathlib import Path

# FlashSparse attention kernel utilizing tiled Q/K/V loads and shared register accumulation
@triton.jit
def _flash_sparse_attention_kernel(
    Q_ptr, K_ptr, V_ptr, sparse_indices_ptr, O_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_idx_b, stride_idx_q, stride_idx_n,
    stride_ob, stride_oh, stride_os, stride_od,
    B, H, S_q, S_k, D, num_sparse_blocks, block_size,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q_tile = tl.program_id(2) # Query token tile program

    # Offsets for query tile
    q_start = pid_q_tile * BLOCK_Q
    if q_start >= S_q:
        return

    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, BLOCK_D)

    # Base pointers
    q_ptr = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    o_ptr = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    idx_base = sparse_indices_ptr + pid_b * stride_idx_b

    # Load Query tile [BLOCK_Q, BLOCK_D]
    q_off = q_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd
    q_mask = (q_offsets[:, None] < S_q) & (d_offsets[None, :] < D)
    q = tl.load(q_ptr + q_off, mask=q_mask, other=0.0)

    # Softmax state
    max_scores = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    denominators = tl.zeros([BLOCK_Q], dtype=tl.float32)
    accum = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # Loop over the sparse block indices
    for sb in range(0, num_sparse_blocks):
        # Retrieve sparse block index map (load for the first query in this tile)
        idx_ptr = idx_base + tl.minimum(q_start, S_q - 1) * stride_idx_q + sb * stride_idx_n
        block_idx = tl.load(idx_ptr)
        
        if block_idx >= 0 and block_idx * block_size < S_k:
            # Tiled key streaming in SRAM
            for kn in range(0, block_size, BLOCK_N):
                k_offsets = tl.arange(0, BLOCK_N)
                k_block_ptr = K_ptr + pid_b * stride_kb + pid_h * stride_kh + (block_idx * block_size + kn) * stride_ks
                
                # Load keys [BLOCK_N, BLOCK_D]
                k_off = k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
                # Enforce physical S_k sequence boundary to prevent GPU indexing faults
                k_mask = ((block_idx * block_size + kn + k_offsets[:, None]) < S_k) & (d_offsets[None, :] < D)
                k = tl.load(k_block_ptr + k_off, mask=k_mask, other=0.0)

                # Compute Score [BLOCK_Q, BLOCK_N] via Native Tensor-Core Matrix Multiplication
                scores = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16))) * scale

                # Online fused normalization update
                row_max = tl.max(scores, axis=1)
                new_max = tl.maximum(max_scores, row_max)
                alpha = tl.exp(max_scores - new_max)
                
                # Softmax exponentials
                beta = tl.exp(scores - new_max[:, None])
                
                # Denominator update
                max_scores = new_max
                denominators = denominators * alpha + tl.sum(beta, axis=1)
                
                # Load values [BLOCK_N, BLOCK_D]
                v_block_ptr = V_ptr + pid_b * stride_vb + pid_h * stride_vh + (block_idx * block_size + kn) * stride_vs
                v_off = k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
                v = tl.load(v_block_ptr + v_off, mask=k_mask, other=0.0)
                
                # Accumulate Output tile
                accum = accum * alpha[:, None] + tl.dot(beta.to(tl.float16), v.to(tl.float16))

    # Write output projection
    accum = accum / (denominators[:, None] + 1e-6)
    o_off = q_offsets[:, None] * stride_os + d_offsets[None, :] * stride_od
    tl.store(o_ptr + o_off, accum, mask=q_mask)


class FlashSparseAttentionEngine:
    """
    SGC Stage 3C.3: FlashSparseAttention Engine.
    Adapts FlashAttention tiled, register-resident execution logic to Differential KV sparse coordinate maps.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)

    def execute(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sparse_indices: torch.Tensor, block_size: int, scale: float) -> torch.Tensor:
        """
        Executes register-resident tiled FlashSparse attention.
        """
        B, H, S_q, D = Q.shape
        _, _, S_k, _ = K.shape
        num_sparse_blocks = sparse_indices.shape[-1]

        O = torch.zeros_like(Q)

        BLOCK_Q = 16  # Tensor-Core compliant power-of-2 bounds
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_N = 16  # Tensor-Core compliant power-of-2 bounds

        grid = (B, H, (S_q + BLOCK_Q - 1) // BLOCK_Q)

        _flash_sparse_attention_kernel[grid](
            Q, K, V, sparse_indices, O,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            sparse_indices.stride(0), sparse_indices.stride(1), sparse_indices.stride(2),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B, H, S_q, S_k, D, num_sparse_blocks, block_size,
            scale,
            BLOCK_Q=BLOCK_Q,
            BLOCK_D=BLOCK_D,
            BLOCK_N=BLOCK_N
        )
        return O
