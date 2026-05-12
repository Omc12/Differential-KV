"""
runtime/kernel_fusion_benchmarks.py

Benchmarks the performance gains of fused resonance attention kernels 
against standard runtime-managed stabilization.
"""

import torch
import time
import numpy as np
from runtime.fused_resonance_attention import FusedResonanceAttention
from runtime.flash_diffkv_attention import FlashDiffKVAttention

def benchmark_fusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, heads, seq, dim = 4, 32, 4096, 128
    rank = 16
    
    q = torch.randn(batch, heads, seq, dim, device=device)
    k_lr = torch.randn(batch, heads, seq, rank, device=device)
    k_v = torch.randn(batch, heads, rank, dim, device=device)
    k_a = torch.randn(batch, heads, dim, device=device)
    res = torch.randn(batch, heads, rank, device=device)
    
    fused_engine = FusedResonanceAttention(heads * dim, heads)
    flash_engine = FlashDiffKVAttention(dim, rank).to(device)
    
    # 1. Standard Unfused Baseline (Simulated)
    # Reconstruct -> Stabilize -> Attention
    start = time.perf_counter()
    for _ in range(10):
        k_recon = torch.matmul(k_lr, k_v) + k_a.unsqueeze(2)
        # Separate stabilization pass
        k_stable = k_recon + 0.1 * torch.randn_like(k_recon)
        # Attention pass
        _ = torch.matmul(q, k_stable.transpose(-2, -1))
    unfused_time = (time.perf_counter() - start) / 10
    
    # 2. Fused FlashDiffKV
    start = time.perf_counter()
    for _ in range(10):
        _, m = flash_engine(q, k_lr, k_v, k_a, k_lr, k_v, k_a, res)
    fused_time = (time.perf_counter() - start) / 10
    
    print("-" * 40)
    print("KERNEL FUSION BENCHMARK RESULTS")
    print("-" * 40)
    print(f"Unfused Baseline: {unfused_time*1000:.4f} ms")
    print(f"Fused FlashDiffKV: {fused_time*1000:.4f} ms")
    print(f"Speedup: {unfused_time / fused_time:.2f}x")
    print(f"Kernel Overhead: {m.get('tiling_overhead_reduction', 0)*100:.1f}% reduction")
    print("-" * 40)

if __name__ == "__main__":
    benchmark_fusion()
