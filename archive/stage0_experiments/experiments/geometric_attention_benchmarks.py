"""
experiments/geometric_attention_benchmarks.py

Benchmarks NCAA throughput, FLOPs, and overhead.
"""

import torch
import time
from runtime.attractor_attention import AttractorNativeAttention
from runtime.sparse_geometry_attention import SparseGeometryAttention
from runtime.geometric_attention_router import GeometricAttentionRouter

def benchmark_ncaa():
    print("Benchmarking Native Cognitive Attention Architecture...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, S, D = 1, 32, 4096, 128
    
    # Components
    ana = AttractorNativeAttention(H*D, H).to(device)
    dsga = SparseGeometryAttention(D, sparsity_target=0.6).to(device)
    router = GeometricAttentionRouter(H*D, H).to(device)
    
    q, k, v = torch.randn(B, H, S, D).to(device), torch.randn(B, H, S, D).to(device), torch.randn(B, H, S, D).to(device)
    manifold = k.clone()
    importance = torch.rand(B, H, S).to(device)
    # q_mean should be [B, H*D] (total feature dim)
    q_mean = q.transpose(1, 2).reshape(B, S, H * D).mean(dim=1)
    stats = torch.tensor([[0.05, 1.0, 0.5]]).to(device)
    
    # 1. Warmup
    for _ in range(5):
        _ = ana(q, k, v, manifold)
        
    # 2. Benchmark ANA
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(100):
        _ = ana(q, k, v, manifold)
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()
    ana_latency = (t1 - t0) / 100 * 1000
    
    # 3. Benchmark DSGA (Sparse)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = dsga(q, k, v, importance)
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()
    dsga_latency = (t1 - t0) / 100 * 1000
    
    # 4. Benchmark Router
    t0 = time.perf_counter()
    for _ in range(100):
        _ = router(q_mean, stats)
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()
    router_latency = (t1 - t0) / 100 * 1000
    
    print(f"Results for B={B}, H={H}, S={S}, D={D}:")
    print(f"ANA Latency: {ana_latency:.4f} ms")
    print(f"DSGA Latency: {dsga_latency:.4f} ms")
    print(f"Router Overhead: {router_latency:.4f} ms")
    
    flop_reduction = 0.6 # From DSGA target
    throughput_gain = 1.0 / (dsga_latency / ana_latency) if ana_latency > 0 else 0
    
    print(f"Estimated FLOP Reduction: {flop_reduction:.0%}")
    print(f"Estimated Throughput Gain vs Dense: {throughput_gain:.2f}x")
    
    return {
        "ana_ms": ana_latency,
        "dsga_ms": dsga_latency,
        "router_ms": router_latency,
        "throughput_gain": throughput_gain
    }

if __name__ == "__main__":
    benchmark_ncaa()
