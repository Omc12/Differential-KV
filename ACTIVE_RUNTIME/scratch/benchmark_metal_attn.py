import os
import sys
import math
import time
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import diffkv_core

DEVICE = torch.device("mps")
torch.manual_seed(42)

# Dimensions matching Qwen-like configs
num_heads = 8
num_kv_heads = 2
head_dim = 64
rank = 16
S_max = 256
num_blocks = 40
K_active = 20  # Number of active blocks
scale = 1.0 / math.sqrt(head_dim)

# Initialize tensors
Q = torch.randn((num_heads, head_dim), dtype=torch.float16, device=DEVICE)
U_pool = torch.randint(-128, 127, (num_blocks, S_max, rank), dtype=torch.int8, device=DEVICE)
U_scale_pool = torch.rand((num_blocks,), dtype=torch.float16, device=DEVICE) * 0.05
VK_pool = torch.randn((num_blocks, rank, num_kv_heads, head_dim), dtype=torch.float16, device=DEVICE) * 0.1
VV_pool = torch.randn((num_blocks, rank, num_kv_heads, head_dim), dtype=torch.float16, device=DEVICE) * 0.1
anchors_K = torch.randn((num_blocks, num_kv_heads, head_dim), dtype=torch.float16, device=DEVICE) * 0.5
anchors_V = torch.randn((num_blocks, num_kv_heads, head_dim), dtype=torch.float16, device=DEVICE) * 0.5
seq_lens = torch.randint(64, S_max, (num_blocks,), dtype=torch.int32, device=DEVICE)
slot_indices = torch.randint(0, num_blocks, (K_active,), dtype=torch.int32, device=DEVICE)

# Sweep over active block counts
print("=" * 70)
print(f"{'Active Blocks (K)':<20} | {'C++ ATen (ms)':<15} | {'Metal Shader (ms)':<18} | {'Speedup':<10}")
print("-" * 70)

for K_active in [2, 4, 8, 16, 32]:
    slot_indices = torch.randint(0, num_blocks, (K_active,), dtype=torch.int32, device=DEVICE)
    
    # Warmup
    for _ in range(20):
        _ = diffkv_core.decode_attention_aten_lse(
            Q, U_pool, U_scale_pool, VK_pool, VV_pool, anchors_K, anchors_V, seq_lens, slot_indices,
            scale, num_heads, num_kv_heads, rank
        )
        _ = diffkv_core.decode_attention_metal(
            Q, U_pool, U_scale_pool, VK_pool, VV_pool, anchors_K, anchors_V, seq_lens, slot_indices,
            scale, num_heads, num_kv_heads, rank
        )
    torch.mps.synchronize()

    # Benchmark C++ ATen
    N_ITERS = 1000
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = diffkv_core.decode_attention_aten_lse(
            Q, U_pool, U_scale_pool, VK_pool, VV_pool, anchors_K, anchors_V, seq_lens, slot_indices,
            scale, num_heads, num_kv_heads, rank
        )
    torch.mps.synchronize()
    t_cpp = ((time.perf_counter() - t0) / N_ITERS) * 1000

    # Benchmark Metal Compute Shader
    t1 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = diffkv_core.decode_attention_metal(
            Q, U_pool, U_scale_pool, VK_pool, VV_pool, anchors_K, anchors_V, seq_lens, slot_indices,
            scale, num_heads, num_kv_heads, rank
        )
    torch.mps.synchronize()
    t_metal = ((time.perf_counter() - t1) / N_ITERS) * 1000

    speedup = t_cpp / t_metal
    print(f"{K_active:<20d} | {t_cpp:>13.4f} | {t_metal:>16.4f} | {speedup:>8.2f}x")

print("=" * 70)

