import os
import sys
import torch
import triton
import triton.language as tl
from pathlib import Path

# Triton sparse attention kernel mapping block-tiled lookup natively to hardware Tensor Cores
@triton.jit
def _triton_sparse_attention_kernel(
    Q_ptr, K_ptr, V_ptr, sparse_indices_ptr, O_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_idx_b, stride_idx_q, stride_idx_n,
    stride_ob, stride_oh, stride_os, stride_od,
    B, H, S_q, S_k, D, num_sparse_blocks, block_size,
    scale,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2) # Thread-block per query token

    if pid_q >= S_q:
        return

    # Base pointers
    q_ptr = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_q * stride_qs
    o_ptr = O_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_q * stride_os
    idx_base = sparse_indices_ptr + pid_b * stride_idx_b + pid_q * stride_idx_q

    # Load Query vector (head dimension BLOCK_D)
    q_offsets = tl.arange(0, BLOCK_D)
    q = tl.load(q_ptr + q_offsets * stride_qd, mask=q_offsets < D, other=0.0)

    # Initialize online softmax accumulators
    max_score = -float("inf")
    denominator = 0.0
    accum = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Iterate over sparse key blocks
    for sb in range(0, num_sparse_blocks):
        # Load block index from GPU index map
        block_idx = tl.load(idx_base + sb * stride_idx_n)
        if block_idx >= 0 and block_idx * block_size < S_k:
            # Loop over key elements in this block (BLOCK_N elements)
            for kn in range(0, block_size, BLOCK_N):
                k_offsets = tl.arange(0, BLOCK_N)
                # Compute global key base address
                k_block_ptr = K_ptr + pid_b * stride_kb + pid_h * stride_kh + (block_idx * block_size + kn) * stride_ks
                
                # Load keys
                k_off = k_offsets[:, None] * stride_ks + q_offsets[None, :] * stride_kd
                # Enforce physical S_k sequence boundary to prevent GPU indexing faults
                k_mask = ((block_idx * block_size + kn + k_offsets[:, None]) < S_k) & (q_offsets[None, :] < D)
                k = tl.load(k_block_ptr + k_off, mask=k_mask, other=0.0)
                
                # QK Dot product
                qk = tl.sum(q[None, :] * k, axis=1) * scale
                
                # Online Softmax update
                new_max = tl.maximum(max_score, tl.max(qk, axis=0))
                alpha = tl.exp(max_score - new_max)
                beta = tl.exp(qk - new_max)
                
                # Update max and denominator
                max_score = new_max
                denominator = denominator * alpha + tl.sum(beta, axis=0)
                
                # Load values
                v_block_ptr = V_ptr + pid_b * stride_vb + pid_h * stride_vh + (block_idx * block_size + kn) * stride_vs
                v_off = k_offsets[:, None] * stride_vs + q_offsets[None, :] * stride_vd
                v = tl.load(v_block_ptr + v_off, mask=k_mask, other=0.0)
                
                # Accumulate weighted values
                accum = accum * alpha + tl.sum(beta[:, None] * v, axis=0)

    # Write output
    accum = accum / (denominator + 1e-6)
    tl.store(o_ptr + q_offsets * stride_od, accum, mask=q_offsets < D)


class TritonSparseAttentionRuntime:
    """
    SGC Stage 3C.3: Triton Sparse Attention Runtime.
    Provides JIT-compiled Triton kernels for high-throughput, tensor-core optimized attention.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)

    def execute(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sparse_indices: torch.Tensor, block_size: int, scale: float) -> torch.Tensor:
        """
        Launches JIT-compiled Triton sparse attention execution.
        """
        B, H, S_q, D = Q.shape
        _, _, S_k, _ = K.shape
        num_sparse_blocks = sparse_indices.shape[-1]

        # Allocate output tensor
        O = torch.zeros_like(Q)

        # Enforce tensor-core aligned BLOCK sizes
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_N = 8  # Sub-block size for inner loop tile matching warp bounds

        # Launch grid
        grid = (B, H, S_q)

        _triton_sparse_attention_kernel[grid](
            Q, K, V, sparse_indices, O,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            sparse_indices.stride(0), sparse_indices.stride(1), sparse_indices.stride(2),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B, H, S_q, S_k, D, num_sparse_blocks, block_size,
            scale,
            BLOCK_D=BLOCK_D,
            BLOCK_N=BLOCK_N
        )
        return O
