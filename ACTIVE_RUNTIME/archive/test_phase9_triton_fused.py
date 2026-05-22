"""
test_phase9_triton_fused.py

Phase 9 Validation: True SRAM-Resident Fused Triton Kernel

Validates that the Triton kernel (Phase 9) is mathematically identical to 
the batched PyTorch kernel (Phase 8), and measures the reduction in VRAM 
allocation and execution latency.
"""

import sys, time, torch
sys.path.insert(0, ".")

from runtime.kv_runtime_manager import KVBlock
from runtime.batched_sparse_attn import build_sparse_batch, batched_sparse_attn_decode
from runtime.triton_sparse_attn import native_triton_sparse_attn_decode as triton_sparse_attn_decode
from compression.lowrank import compress_lowrank

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
HEADS    = 28     # Qwen2.5-7B query heads
NUM_KV   = 4      # GQA KV heads
KV_GRP   = HEADS // NUM_KV   # = 7
HEAD_DIM = 128
RANK     = 16
BLOCK_S  = 63     # tokens per compressed block

if DEVICE != "cuda":
    print("Phase 9 Triton kernel requires CUDA.")
    sys.exit(0)

print("=" * 60)
print("PHASE 9 — TRITON FUSED SPARSE ATTENTION VALIDATION")
print("=" * 60)

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

# 1. Correctness
print("\n== 1. Correctness Check (N=16 compressed blocks) ==")
N_CORRECT = 16
blocks = [make_compressed_block(i) for i in range(N_CORRECT)]
q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
batch = build_sparse_batch(blocks, device=DEVICE)

# Run P8 (batched PyTorch)
out_p8 = batched_sparse_attn_decode(
    q, batch, [], None, None, KV_GRP
)

# Run P9 (Triton)
out_p9 = triton_sparse_attn_decode(
    q, batch, [], None, None, KV_GRP
)

cos = torch.nn.functional.cosine_similarity(
    out_p8.reshape(1,-1).float(), out_p9.reshape(1,-1).float()
).item()
err = (out_p8.float()-out_p9.float()).abs().max().item()

print(f"  Cosine similarity (P8 vs P9): {cos:.6f}")
print(f"  Max absolute error:           {err:.6f}")
if cos > 0.99:
    print("  PASS")
else:
    print("  FAIL")

# 2. Latency vs Block Count
print("\n== 2. Latency vs Block Count ==")
print(f"  {'N':>4}  {'Phase8 ms':>12}  {'Phase9 ms':>12}  {'Speedup':>10}")
print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}")

WARMUP = 10
STEPS = 100

for N in [8, 16, 32, 64]:
    blks = [make_compressed_block(i) for i in range(N)]
    bat = build_sparse_batch(blks, device=DEVICE)
    q_t = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # P8
    for _ in range(WARMUP): batched_sparse_attn_decode(q_t, bat, [], None, None, KV_GRP)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS): batched_sparse_attn_decode(q_t, bat, [], None, None, KV_GRP)
    torch.cuda.synchronize()
    p8_ms = (time.perf_counter() - t0) / STEPS * 1000

    # P9
    for _ in range(WARMUP): triton_sparse_attn_decode(q_t, bat, [], None, None, KV_GRP)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS): triton_sparse_attn_decode(q_t, bat, [], None, None, KV_GRP)
    torch.cuda.synchronize()
    p9_ms = (time.perf_counter() - t0) / STEPS * 1000

    speedup = p8_ms / p9_ms
    print(f"  {N:>4}  {p8_ms:>12.3f}  {p9_ms:>12.3f}  {speedup:>9.2f}×")

# 3. VRAM allocation tracing (Intermediate tensors)
print("\n== 3. Peak VRAM Allocation (N=64) ==")
blks = [make_compressed_block(i) for i in range(64)]
bat = build_sparse_batch(blks, device=DEVICE)
q_t = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)

torch.cuda.reset_peak_memory_stats()
batched_sparse_attn_decode(q_t, bat, [], None, None, KV_GRP)
p8_peak = torch.cuda.max_memory_allocated() / 1e6

torch.cuda.reset_peak_memory_stats()
triton_sparse_attn_decode(q_t, bat, [], None, None, KV_GRP)
p9_peak = torch.cuda.max_memory_allocated() / 1e6

print(f"  Phase 8 (PyTorch batched): {p8_peak:.2f} MB")
print(f"  Phase 9 (Triton fused)   : {p9_peak:.2f} MB")
print(f"  VRAM Traffic Reduction   : {p8_peak/p9_peak:.1f}x less allocation")
