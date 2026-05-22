"""
runtime/triton_diffkv.py

Triton-optimized fused reconstruction kernels for Differential KV.
Provides maximum memory bandwidth efficiency for ΔKV = U @ V.T + S + anchor.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def lowrank_recon_kernel(
    U_ptr, V_ptr, anchor_ptr, out_ptr,
    stride_un, stride_uk,
    stride_vk, stride_vd,
    stride_ad,
    stride_on, stride_od,
    n_tokens, rank, feat_dim,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel: out[n, d] = anchor[d] + sum_k(U[n, k] * V[k, d])
    """
    # Program IDs
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Offsets
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    
    # Boundary masks
    mask_n = offs_n < n_tokens
    mask_d = offs_d < feat_dim

    # Load anchor
    anchor = tl.load(anchor_ptr + offs_d, mask=mask_d, other=0.0)
    
    # Initialize accumulator with anchor
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=tl.float32)
    
    # Loop over rank
    for k_start in range(0, rank, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < rank
        
        # Load U [BLOCK_SIZE_N, BLOCK_SIZE_K]
        u = tl.load(U_ptr + offs_n[:, None] * stride_un + offs_k[None, :] * stride_uk, 
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Load V [BLOCK_SIZE_K, BLOCK_SIZE_D]
        v = tl.load(V_ptr + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd, 
                    mask=mask_k[:, None] & mask_d[None, :], other=0.0)
        
        acc += tl.dot(u, v)

    # Add anchor (broadcasted)
    acc += anchor[None, :]

    # Store result
    out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=mask_n[:, None] & mask_d[None, :])

def triton_fused_reconstruct(
    U: torch.Tensor,
    V: torch.Tensor,
    anchor: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Python wrapper for Triton fused low-rank reconstruction.
    """
    n_tokens, rank = U.shape
    _, feat_dim = V.shape
    
    if out is None:
        out = torch.empty((n_tokens, feat_dim), device=U.device, dtype=U.dtype)

    # Grid size
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = 64
    BLOCK_SIZE_K = 16
    
    grid = (triton.cdiv(n_tokens, BLOCK_SIZE_N), triton.cdiv(feat_dim, BLOCK_SIZE_D))
    
    lowrank_recon_kernel[grid](
        U, V, anchor, out,
        U.stride(0), U.stride(1),
        V.stride(0), V.stride(1),
        anchor.stride(0),
        out.stride(0), out.stride(1),
        n_tokens, rank, feat_dim,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    if scale != 1.0:
        out *= scale
        
    return out

class TritonDiffKV:
    """
    Manager for Triton-optimized KV operations.
    """
    @staticmethod
    def reconstruct_lowrank(U, V, anchor, scale=1.0):
        # Fallback to PyTorch if Triton fails or is not available
        try:
            return triton_fused_reconstruct(U, V, anchor, scale=scale)
        except Exception as e:
            # print(f"Triton failed, falling back to PyTorch: {e}")
            return (torch.matmul(U.float(), V.float()) * scale + anchor.float()).to(U.dtype)

    @staticmethod
    def reconstruct_lowrank_sparse(U, V, anchor, sparse_indices, sparse_values, scale=1.0):
        # Fused kernel for sparse is more complex, we'll start with low-rank + sparse scatter
        out = TritonDiffKV.reconstruct_lowrank(U, V, anchor, scale)
        if sparse_indices is not None and sparse_indices.numel() > 0:
            out.view(-1).index_add_(0, sparse_indices.long(), sparse_values.to(out.dtype))
        return out
