"""
hardware_materialization/triton_sparse_attention_materializer.py

Materializes Triton sparse attention kernels from emulated paths.
"""

import torch
import logging
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

logger = logging.getLogger("TritonMaterializer")

@triton.jit
def _sparse_attn_fused_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    head_dim, seq_len,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Simple fused sparse attention kernel (decode-style, q_len=1).
    Computes: Out = Softmax(Q @ K.T / sqrt(d)) @ V
    """
    # Program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Offsets for head dimension
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < head_dim

    # Load Q [head_dim]
    q_ptrs = Q_ptr + batch_id * stride_qb + head_id * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=mask_d, other=0.0)

    # Online softmax accumulators
    max_score = -float('inf')
    sum_exp = 0.0
    acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)

    # Pre-calculated scale
    scale = 1.0 / tl.sqrt(head_dim.to(tl.float32))

    # Iterate over sparse sequence dimension
    for s in range(seq_len):
        # Load K [head_dim]
        k_ptrs = K_ptr + batch_id * stride_kb + head_id * stride_kh + s * stride_ks + offs_d * stride_kd
        k = tl.load(k_ptrs, mask=mask_d, other=0.0)

        # Compute score
        score = tl.sum(q * k) * scale
        
        # Online softmax update
        new_max = tl.maximum(max_score, score)
        alpha = tl.exp(max_score - new_max)
        beta = tl.exp(score - new_max)
        
        sum_exp = sum_exp * alpha + beta
        max_score = new_max
        
        # Load V [head_dim]
        v_ptrs = V_ptr + batch_id * stride_vb + head_id * stride_vh + s * stride_vs + offs_d * stride_vd
        v = tl.load(v_ptrs, mask=mask_d, other=0.0)
        
        # Accumulate weighted V
        acc = acc * alpha + v * beta

    # Normalize by sum_exp
    acc = acc / sum_exp

    # Store result
    out_ptrs = Out_ptr + batch_id * stride_ob + head_id * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc, mask=mask_d)

class TritonSparseAttentionMaterializer:
    """
    Materializes Triton sparse attention operations with real GPU execution.
    """
    def __init__(self):
        self.enabled = HAS_TRITON and torch.cuda.is_available()
        if not self.enabled:
            logger.warning("Triton or CUDA not available. Materializer will use fallback.")

    def execute(self, q, k, v, mask=None):
        """
        Executes the materialized Triton kernel.
        q: [bsz, n_heads, 1, head_dim]
        k: [bsz, n_heads, sparse_len, head_dim]
        v: [bsz, n_heads, sparse_len, head_dim]
        """
        if not self.enabled or not q.is_cuda:
            return self._fallback(q, k, v, mask)

        bsz, n_heads, q_len, head_dim = q.shape
        sparse_len = k.shape[2]
        
        if q_len != 1:
            # Current kernel only supports decode-style (q_len=1)
            return self._fallback(q, k, v, mask)

        out = torch.empty_like(q)
        
        # Grid: (bsz, n_heads)
        grid = (bsz, n_heads)
        
        # BLOCK_SIZE_D must be a power of 2 >= head_dim
        block_size_d = triton.next_power_of_2(head_dim)
        
        _sparse_attn_fused_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(3),
            head_dim, sparse_len,
            BLOCK_SIZE_D=block_size_d
        )
        
        return out

    def _fallback(self, q, k, v, mask=None):
        """Standard PyTorch fallback."""
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)
