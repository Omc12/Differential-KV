"""
test_phase10_native_overhead.py

Phase 10: Native Block Pool Validation
Measures the overhead of NativeBlockPool index lookup vs the previous
Python torch.stack() approach (SparseBatch).
"""

import sys, time, torch
sys.path.insert(0, ".")

from runtime.native_block_pool import NativeBlockPool
from runtime.triton_sparse_attn import native_triton_sparse_attn_decode
from runtime.kv_runtime_manager import KVBlock
from compression.lowrank import compress_lowrank

def make_compressed_block(seed=0):
    torch.manual_seed(seed)
    k = torch.randn(1, NUM_KV, BLOCK_S + 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, NUM_KV, BLOCK_S + 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    anchor_kv = torch.stack([k[:, :, 0], v[:, :, 0]], dim=1)
    blk = KVBlock(anchor_idx=0, anchor_kv=anchor_kv, token_indices=list(range(BLOCK_S+1)))

    feat_dim  = 2 * NUM_KV * HEAD_DIM
    stacked   = torch.stack([k[0, :, 1:].transpose(0, 1), v[0, :, 1:].transpose(0, 1)], dim=1)
    flat      = stacked.reshape(BLOCK_S, feat_dim).float()
    anc_flat  = anchor_kv.view(-1).float()
    deltas    = flat - anc_flat.unsqueeze(0)
    lr        = compress_lowrank(deltas, rank=RANK)
    blk.U, blk.V, blk.scale = lr.U, lr.V, lr.scale
    return blk

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
HEADS    = 28     # Qwen2.5-7B query heads
NUM_KV   = 4      # GQA KV heads
KV_GRP   = HEADS // NUM_KV   # = 7
HEAD_DIM = 128
RANK     = 16
BLOCK_S  = 63     # tokens per compressed block

if DEVICE != "cuda":
    print("Phase 10 Triton kernel requires CUDA.")
    sys.exit(0)

print("=" * 60)
print("PHASE 10 — NATIVE BLOCK POOL TIMING VALIDATION")
print("=" * 60)

# 1. Initialize Pool
max_blocks = 2048
pool = NativeBlockPool(max_blocks, NUM_KV, HEAD_DIM, RANK, BLOCK_S+1, device=DEVICE)

# 2. Simulate Background Compression (Write blocks into pool)
# We make 64 compressed blocks and put them in the pool
N = 64
block_indices = []
print(f"Pre-allocating {N} blocks into NativeBlockPool...")

for i in range(N):
    blk = make_compressed_block(i)
    pool_idx = pool.allocate_block()
    pool.write_block(
        pool_idx, blk.U, blk.V, blk.anchor_kv[:, 0], blk.anchor_kv[:, 1], blk.scale, blk.U.shape[0]
    )
    block_indices.append(pool_idx)

block_indices_t = torch.tensor(block_indices, device=DEVICE, dtype=torch.int32)
q_t = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)

# 3. Correctness & Compilation
print("\nCompiling Triton Kernel...")
_ = native_triton_sparse_attn_decode(q_t, block_indices_t, pool, [], None, None, KV_GRP, R=RANK, S_MAX=64)
torch.cuda.synchronize()

# 4. Measure Pure Triton Decode Latency vs Phase 8 Python Stacking
print("\n== Native Decode Orchestration vs Phase 8/9 Python ==")
print(f"  {'N':>4}  {'Python Stacking ms':>20}  {'Native Pool ms':>15}  {'Speedup':>10}")
print("  " + "-"*56)

# (We use the previous Python stacking time we measured for Phase 8/9 as baseline: ~0.8ms for N=64)
from runtime.batched_sparse_attn import build_sparse_batch

for n_test in [8, 16, 32, 64]:
    sub_indices = block_indices_t[:n_test]
    
    # Time Python stack
    blks_subset = [make_compressed_block(i) for i in range(n_test)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        bat = build_sparse_batch(blks_subset, device=DEVICE)
    torch.cuda.synchronize()
    p8_ms = (time.perf_counter() - t0) / 100 * 1000
    
    # Warmup Native to trigger compilation
    _ = native_triton_sparse_attn_decode(q_t, sub_indices, pool, [], None, None, KV_GRP, R=RANK, S_MAX=64)
    
    # Time Native
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        # The python loop now only consists of the triton launch!
        _ = native_triton_sparse_attn_decode(q_t, sub_indices, pool, [], None, None, KV_GRP, R=RANK, S_MAX=64)
    torch.cuda.synchronize()
    p10_ms = (time.perf_counter() - t0) / 100 * 1000
    
    speedup = p8_ms / p10_ms if p10_ms > 0 else float('inf')
    print(f"  {n_test:>4}  {p8_ms:>20.3f}  {p10_ms:>15.3f}  {speedup:>9.1f}x")

print("\n" + "=" * 60)
print("PHASE 10 NATIVE VALIDATION COMPLETE")
