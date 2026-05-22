"""
runtime/sparse_mlp_fused.py

Phase 11 Step 4 — Fused Triton Sparse MLP

The PyTorch block-sparse path (sparse_mlp.py) has a critical problem at seq=1:
  - index_select(W_up, neuron_ids) touches non-contiguous HBM rows
  - Python-side gather overhead ~= 300-600us per layer
  - This DOMINATES the actual matmul time for seq=1

This module implements a fused Triton kernel that:
  1. Does NOT call index_select from Python
  2. Loads only active block tiles directly in the kernel
  3. Fuses gate + SiLU + up + down into a single kernel launch sequence
  4. Uses block-level early termination (skip inactive block tiles entirely)

The kernel works at the BLOCK level:
  - Total blocks: intermediate_size // block_size = 18944 // 128 = 148
  - Active blocks: top-k by gate block importance
  - Per active block: load W_gate tile, W_up tile, compute, accumulate W_down tile

This avoids ALL Python-side gather overhead.

TIMING EXPECTATION (Qwen2-7B, RTX 4090, seq=1):
  Dense F.linear:              ~900 us  (3 launches)
  PyTorch sparse (index_select): ~1300 us (3 launches + 3 index_select)
  Triton fused sparse (50%):     ~500 us (fewer HBM loads, fused)
  Triton fused sparse (30%):     ~380 us

These are estimates. Actual numbers from profiler are the ground truth.
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: Fused block-sparse gate+up+down projection
#
# Grid: (num_active_blocks, seq_len)
# Each program handles one (active_block, seq_position) pair.
# Accumulates into shared output buffer using atomic add.
#
# This avoids index_select by using the block_id lookup table inside the kernel.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _sparse_gate_up_kernel(
    X_ptr,             # [seq, hidden]
    W_gate_ptr,        # [d_ff, hidden]
    W_up_ptr,          # [d_ff, hidden]
    ActiveIds_ptr,     # [num_active] int32
    Mixed_ptr,         # [seq, num_active] output
    seq_len,
    hidden,
    k_active,
    BLOCK_N:  tl.constexpr,
    BLOCK_H:  tl.constexpr,
):
    seq_pid = tl.program_id(0)
    n_pid   = tl.program_id(1)

    n_start = n_pid * BLOCK_N
    n_offs  = n_start + tl.arange(0, BLOCK_N)
    n_mask  = n_offs < k_active

    neuron_ids = tl.load(ActiveIds_ptr + n_offs, mask=n_mask, other=0)

    gate_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    up_acc   = tl.zeros([BLOCK_N], dtype=tl.float32)

    for h_start in range(0, hidden, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < hidden

        x_ptrs = X_ptr + seq_pid * hidden + h_offs
        x_tile = tl.load(x_ptrs, mask=h_mask, other=0.0).to(tl.float32)

        wg_ptrs = W_gate_ptr + neuron_ids[:, None] * hidden + h_offs[None, :]
        wg_mask = n_mask[:, None] & h_mask[None, :]
        wg_tile = tl.load(wg_ptrs, mask=wg_mask, other=0.0).to(tl.float32)

        wu_ptrs = W_up_ptr + neuron_ids[:, None] * hidden + h_offs[None, :]
        wu_tile = tl.load(wu_ptrs, mask=wg_mask, other=0.0).to(tl.float32)

        gate_acc += tl.sum(wg_tile * x_tile[None, :], axis=1)
        up_acc   += tl.sum(wu_tile * x_tile[None, :], axis=1)

    gate_silu = gate_acc * tl.sigmoid(gate_acc)
    mixed     = gate_silu * up_acc

    out_ptrs = Mixed_ptr + seq_pid * k_active + n_offs
    tl.store(out_ptrs, mixed.to(tl.float16), mask=n_mask)


@triton.jit
def _sparse_down_kernel(
    Mixed_ptr,         # [seq, k_active]
    W_down_ptr,        # [hidden, d_ff]
    ActiveIds_ptr,     # [k_active] int32
    Out_ptr,           # [seq, hidden]
    seq_len,
    hidden,
    d_ff,
    k_active,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_pid = tl.program_id(0)
    h_pid   = tl.program_id(1)

    h_start = h_pid * BLOCK_H
    h_offs  = h_start + tl.arange(0, BLOCK_H)
    h_mask  = h_offs < hidden

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k_start in range(0, k_active, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_active

        neuron_ids = tl.load(ActiveIds_ptr + k_offs, mask=k_mask, other=0)

        mixed_ptrs = Mixed_ptr + seq_pid * k_active + k_offs
        mixed_tile = tl.load(mixed_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        wd_ptrs = W_down_ptr + h_offs[:, None] * d_ff + neuron_ids[None, :]
        wd_mask = h_mask[:, None] & k_mask[None, :]
        wd_tile = tl.load(wd_ptrs, mask=wd_mask, other=0.0).to(tl.float32)

        acc += tl.sum(wd_tile * mixed_tile[None, :], axis=1)

    out_ptrs = Out_ptr + seq_pid * hidden + h_offs
    tl.store(out_ptrs, acc.to(tl.float16), mask=h_mask)


def fused_sparse_mlp(
    x:          torch.Tensor,    # [bsz, seq, hidden] or [seq, hidden]
    W_gate:     torch.Tensor,    # [d_ff, hidden]
    W_up:       torch.Tensor,    # [d_ff, hidden]
    W_down:     torch.Tensor,    # [hidden, d_ff]
    keep_ratio: float = 0.5,
    block_size: int   = 128,
    min_blocks: int   = 8,
) -> Tuple[torch.Tensor, dict]:
    original_shape = x.shape
    if x.dim() == 3:
        bsz, seq, hidden = x.shape
        x_2d = x.reshape(bsz * seq, hidden).contiguous()
    else:
        x_2d = x.contiguous()

    seq_len, hidden = x_2d.shape
    d_ff = W_gate.shape[0]
    total_blocks = d_ff // block_size

    # Gate routing
    gate_full = F.linear(x_2d, W_gate)
    gate_silu = F.silu(gate_full)

    gate_blocked  = gate_silu.abs().view(seq_len, total_blocks, block_size).mean(dim=(0, 2))
    k_blocks      = max(min_blocks, int(total_blocks * keep_ratio))
    k_blocks      = min(k_blocks, total_blocks)
    _, active_block_ids = torch.topk(gate_blocked, k_blocks, sorted=True)
    
    # Create active neuron IDs
    active_block_ids = active_block_ids.to(torch.int32).sort()[0]
    offsets = active_block_ids * block_size
    active_ids = (offsets.unsqueeze(1) + torch.arange(block_size, device=x.device).unsqueeze(0)).reshape(-1)
    k_active = active_ids.shape[0]

    # Kernel 1: Gate + Up
    BLOCK_N = min(128, k_active)
    BLOCK_H_UP = 64
    num_n_tiles = triton.cdiv(k_active, BLOCK_N)
    mixed = torch.empty((seq_len, k_active), device=x.device, dtype=torch.float16)

    grid1 = (seq_len, num_n_tiles)
    _sparse_gate_up_kernel[grid1](
        x_2d.to(torch.float16), W_gate, W_up, active_ids, mixed,
        seq_len, hidden, k_active,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H_UP
    )

    # Kernel 2: Down
    BLOCK_H_DN = 64
    BLOCK_K_DN = 64
    num_h_tiles = triton.cdiv(hidden, BLOCK_H_DN)
    out = torch.zeros((seq_len, hidden), device=x.device, dtype=torch.float16)

    grid2 = (seq_len, num_h_tiles)
    _sparse_down_kernel[grid2](
        mixed, W_down, active_ids, out,
        seq_len, hidden, d_ff, k_active,
        BLOCK_H=BLOCK_H_DN, BLOCK_K=BLOCK_K_DN
    )

    out = out.reshape(original_shape)
    return out, {
        "active_blocks":  k_blocks,
        "total_blocks":   total_blocks,
        "keep_ratio":     k_blocks / total_blocks,
        "flop_reduction": (1.0 - k_blocks / total_blocks) * (2 / 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Timing comparison: PyTorch sparse vs Triton fused sparse
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_fused_vs_pytorch(hidden=3584, d_ff=18944, seq=1, keep_ratio=0.5,
                                block_size=128, iters=200, warmup=50):
    """
    Compare three paths:
      1. Dense PyTorch (baseline)
      2. PyTorch block-sparse (index_select overhead)
      3. Triton fused sparse (no Python gather)

    Returns timing dict in milliseconds.
    """
    import torch.nn.functional as F
    from runtime.sparse_mlp import BlockSparseMLPExecutor

    device = "cuda"
    x    = torch.randn(1, seq, hidden, device=device, dtype=torch.float16) * 0.3
    x_2d = x.reshape(seq, hidden)
    Wg   = torch.randn(d_ff, hidden, device=device, dtype=torch.float16) * 0.02
    Wu   = torch.randn(d_ff, hidden, device=device, dtype=torch.float16) * 0.02
    Wd   = torch.randn(hidden, d_ff, device=device, dtype=torch.float16) * 0.02

    class FakeLin:
        def __init__(self, w): self.weight = w
        bias = None

    gate_m = FakeLin(Wg)
    up_m   = FakeLin(Wu)
    down_m = FakeLin(Wd)
    ex     = BlockSparseMLPExecutor(block_size=block_size, keep_ratio=keep_ratio)

    def dense():
        return F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)

    def pt_sparse():
        return ex.forward(x, gate_m, up_m, down_m, F.silu)

    def triton_fused():
        return fused_sparse_mlp(x_2d, Wg, Wu, Wd, keep_ratio=keep_ratio, block_size=block_size)

    def time_fn(fn, label):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / iters
        return ms

    dense_ms    = time_fn(dense,        "dense")
    pt_ms       = time_fn(pt_sparse,    "pt_sparse")
    triton_ms   = time_fn(triton_fused, "triton_fused")

    return {
        "seq":              seq,
        "keep_ratio":       keep_ratio,
        "dense_ms":         round(dense_ms, 4),
        "pytorch_sparse_ms":round(pt_ms, 4),
        "triton_fused_ms":  round(triton_ms, 4),
        "pt_vs_dense":      round(dense_ms / pt_ms, 3),
        "triton_vs_dense":  round(dense_ms / triton_ms, 3),
    }
