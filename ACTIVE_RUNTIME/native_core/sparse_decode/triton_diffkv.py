"""
runtime/triton_diffkv.py

Triton-optimized fused reconstruction kernels for Differential KV.
Provides maximum memory bandwidth efficiency for DeltaKV = U @ V.T + anchor.
Falls back to pure-PyTorch on any system where Triton is unavailable.

Mac/MPS: Triton is CUDA-only; the PyTorch fallback is always used on Apple Silicon.
NVTX tracing is replaced with mac_utils no-op shims on non-CUDA platforms.
"""

import torch
from typing import Optional

try:
    from native_core.mac_utils import nvtx_push as _nvtx_push, nvtx_pop as _nvtx_pop, has_cuda as _has_cuda
except ImportError:
    def _nvtx_push(label, device=None): pass
    def _nvtx_pop(device=None): pass
    def _has_cuda(): return torch.cuda.is_available()

try:
    import triton
    import triton.language as tl

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
        """
        Fused kernel: out[n, d] = anchor[d] + sum_k(U[n, k] * V[k, d]) * scale
        """
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

    _HAS_TRITON = True

except (ImportError, Exception):
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Python wrapper — always importable regardless of Triton availability
# ---------------------------------------------------------------------------

def triton_fused_reconstruct(
    U: torch.Tensor,
    V: torch.Tensor,
    anchor: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Fused low-rank reconstruction: out = U @ V * scale + anchor.
    Uses Triton kernel when available; falls back to PyTorch otherwise.
    """
    n_tokens, rank = U.shape
    _, feat_dim = V.shape

    # --- PyTorch fallback (no Triton) ---
    if not _HAS_TRITON:
        result = (torch.matmul(U.float(), V.float()) * scale + anchor.float()).to(U.dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # --- Triton path ---
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


# ---------------------------------------------------------------------------
# TritonDiffKV — buffer-pooled reconstruction manager
# ---------------------------------------------------------------------------

class TritonDiffKV:
    """
    Manager for Triton-optimized (or PyTorch-fallback) KV reconstruction.
    Uses a class-level buffer pool to eliminate aten::empty allocation churn.
    """
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
        """
        Reconstruct KV delta: result = U @ V * scale + anchor.

        Uses the class-level reuse buffer then clones so callers get an
        independent tensor that won't be overwritten by a future call.
        """
        out_buf = TritonDiffKV._get_recon_buffer(
            U.shape[0], V.shape[1], U.device, U.dtype
        )
        try:
            out = triton_fused_reconstruct(U, V, anchor, out=out_buf, scale=scale)
            return out.clone()  # decouple from the shared buffer
        except Exception as e:
            print(f"[DiffKV] Triton reconstruct failed, falling back to PyTorch: {e}")
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
