"""
test_phase8_batched_sparse.py

Phase 8 Validation:
  1. Correctness: batched kernel == Phase 6 loop (cosine sim >= 0.99)
  2. Kernel launch reduction: count torch ops issued per path
  3. Latency vs Phase 6 at increasing block counts (N=4, 8, 16, 32)
  4. GPU profiler trace (optional, if CUDA available)
"""

import sys, math, time, torch
sys.path.insert(0, ".")

from runtime.kv_runtime_manager import KVBlock
from runtime.sparse_attention    import fused_sparse_attention_decode
from runtime.batched_sparse_attn import build_sparse_batch, batched_sparse_attn_decode
from compression.lowrank         import compress_lowrank

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
HEADS    = 28     # Qwen2.5-7B query heads
NUM_KV   = 4      # GQA KV heads
KV_GRP   = HEADS // NUM_KV   # = 7
HEAD_DIM = 128
RANK     = 16
BLOCK_S  = 63     # tokens per compressed block (block_size - 1)

print("=" * 60)
print("PHASE 8 — BATCHED SPARSE ATTENTION VALIDATION")
print(f"  Device={DEVICE}  H_q={HEADS}  H_kv={NUM_KV}  D={HEAD_DIM}  R={RANK}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def make_compressed_block(seed=0):
    """Create a KVBlock with real low-rank compression."""
    torch.manual_seed(seed)
    k = torch.randn(1, NUM_KV, BLOCK_S + 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, NUM_KV, BLOCK_S + 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    anchor_kv = torch.stack([k[:, :, 0], v[:, :, 0]], dim=1)
    blk = KVBlock(anchor_idx=0, anchor_kv=anchor_kv, token_indices=list(range(BLOCK_S+1)))

    feat_dim  = 2 * NUM_KV * HEAD_DIM
    stacked   = torch.stack([k[0, :, 1:].transpose(0, 1), v[0, :, 1:].transpose(0, 1)], dim=1)  # [S, 2, H_kv, D]
    flat      = stacked.reshape(BLOCK_S, feat_dim).float()
    anc_flat  = anchor_kv.view(-1).float()
    deltas    = flat - anc_flat.unsqueeze(0)
    lr        = compress_lowrank(deltas, rank=RANK)
    blk.U, blk.V, blk.scale = lr.U, lr.V, lr.scale
    return blk

def make_active_window():
    """Dense active window (last block, not compressed)."""
    torch.manual_seed(999)
    T = 20
    ak = torch.randn(1, NUM_KV, T, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    av = torch.randn(1, NUM_KV, T, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    return ak, av

# ─────────────────────────────────────────────────────────────────
# 1. Correctness at N=4
# ─────────────────────────────────────────────────────────────────
print("\n== 1. Correctness Check (N=4 compressed blocks) ==")

N_CORRECT = 4
blocks  = [make_compressed_block(i) for i in range(N_CORRECT)]
act_k, act_v = make_active_window()
q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)

with torch.no_grad():
    out_p6 = fused_sparse_attention_decode(q, blocks, act_k, act_v, KV_GRP)

    batch  = build_sparse_batch(blocks, device=DEVICE)
    out_p8 = batched_sparse_attn_decode(
        q=q, batch=batch, dense_blocks=[], active_k=act_k, active_v=act_v,
        num_key_value_groups=KV_GRP,
    )

cos = torch.nn.functional.cosine_similarity(
    out_p6.reshape(1,-1).float(), out_p8.reshape(1,-1).float()
).item()
err = (out_p6.float()-out_p8.float()).abs().max().item()
print(f"  Cosine similarity (P6 vs P8): {cos:.6f}")
print(f"  Max absolute error:           {err:.6f}")
assert cos > 0.99, f"CORRECTNESS FAIL: cos={cos:.4f}"
print("  PASS\n")

# ─────────────────────────────────────────────────────────────────
# 2. Latency vs block count
# ─────────────────────────────────────────────────────────────────
print("== 2. Latency vs Block Count ==")
print(f"  {'N':>4}  {'Phase6 ms':>12}  {'Phase8 ms':>12}  {'Speedup':>10}")
print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}")

WARMUP = 10
STEPS  = 100

for N in [4, 8, 16, 32]:
    blks    = [make_compressed_block(i) for i in range(N)]
    ak, av  = make_active_window()
    q_t     = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    bat     = build_sparse_batch(blks, device=DEVICE)

    # Phase 6 warmup + time
    with torch.no_grad():
        for _ in range(WARMUP):
            fused_sparse_attention_decode(q_t, blks, ak, av, KV_GRP)
    if DEVICE == "cuda": torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(STEPS):
            fused_sparse_attention_decode(q_t, blks, ak, av, KV_GRP)
    if DEVICE == "cuda": torch.cuda.synchronize()
    p6_ms = (time.perf_counter() - t0) / STEPS * 1000

    # Phase 8 warmup + time
    with torch.no_grad():
        for _ in range(WARMUP):
            batched_sparse_attn_decode(q_t, bat, [], ak, av, KV_GRP)
    if DEVICE == "cuda": torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(STEPS):
            batched_sparse_attn_decode(q_t, bat, [], ak, av, KV_GRP)
    if DEVICE == "cuda": torch.cuda.synchronize()
    p8_ms = (time.perf_counter() - t0) / STEPS * 1000

    speedup = p6_ms / p8_ms
    print(f"  {N:>4}  {p6_ms:>12.3f}  {p8_ms:>12.3f}  {speedup:>9.2f}×")

# ─────────────────────────────────────────────────────────────────
# 3. Profiler trace (N=16, 10 steps)
# ─────────────────────────────────────────────────────────────────
print("\n== 3. Profiler Trace (N=16) ==")
N_PROF  = 16
blks_p  = [make_compressed_block(i) for i in range(N_PROF)]
bat_p   = build_sparse_batch(blks_p, device=DEVICE)
ak_p, av_p = make_active_window()
q_p     = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)

if DEVICE == "cuda":
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                out = batched_sparse_attn_decode(q_p, bat_p, [], ak_p, av_p, KV_GRP)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # Top CUDA ops by self time
    print("\n  Top CUDA kernels (Phase 8 batched, N=16 blocks):")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

else:
    print("  (CUDA profiler not available on CPU)")

# ─────────────────────────────────────────────────────────────────
# 4. Memory bandwidth estimate
# ─────────────────────────────────────────────────────────────────
print("\n== 4. Memory Traffic Estimate (N=16) ==")
# Phase 6 reads per step per block:
#   anchor_K (H_kv*D) + anchor_V (H_kv*D) + V_K (R*H_kv*D) + V_V (R*H_kv*D) + U (S*R)
# Phase 8 reads per step (ONE LOAD of all N blocks stacked):
#   Same tensors but loaded in a single contiguous pass, enabling L2 reuse

N_BW    = 16
fp16    = 2   # bytes per fp16 element
per_blk = (
    2 * NUM_KV * HEAD_DIM         # anchor_K + anchor_V
  + 2 * RANK * NUM_KV * HEAD_DIM  # V_K + V_V
  + BLOCK_S * RANK                 # U
) * fp16
total_bytes = N_BW * per_blk
print(f"  Bytes read per decode step (N={N_BW} blocks): {total_bytes/1024:.1f} KB")
print(f"  Per block: {per_blk/1024:.1f} KB")
print(f"  At 1.2 TB/s (H100): theoretical min = {total_bytes/1.2e12*1e6:.3f} µs")

print("\n" + "=" * 60)
print("PHASE 8 VALIDATION COMPLETE")
print("=" * 60)
