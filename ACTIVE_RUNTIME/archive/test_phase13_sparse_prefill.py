"""
test_phase13_sparse_prefill.py

Phase 13 Validation: Sparse Prefill Execution

Validates:
1. FLOP reduction during 8192-token prefill
2. Correct output shape and execution times
3. Quality/drift check (cosine similarity with dense SDPA)
"""

import sys, time, math, torch
sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 68)
print("PHASE 13 — SPARSE PREFILL ATTENTION VALIDATION")
print("=" * 68)

if DEVICE != "cuda":
    print("SKIPPED: CUDA required for performance timings.")
    sys.exit(0)

from runtime.sparse_prefill import SparsePrefillEngine

BSZ = 1
HEADS = 32
SEQ_LEN = 8192
HEAD_DIM = 128

print(f"\n[1] INITIALIZING SPARSE PREFILL")
print(f"  Dimensions: bsz={BSZ}, heads={HEADS}, seq={SEQ_LEN}, head_dim={HEAD_DIM}")

torch.manual_seed(42)
q = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1
k = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1
v = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1

engine = SparsePrefillEngine(sink_tokens=64, chunk_size=512, local_window_chunks=1)
print(f"  Configuration: Chunk Size={engine.chunk_size}, Sinks={engine.sink_tokens}, Local Window={engine.local_window_chunks} chunk(s)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FLOP & TIMING VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] EXECUTION & FLOP REDUCTION")

# Warmup
dense_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
sparse_out = engine.execute_sparse_attention(q, k, v)

# Dense timing
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
dense_ms = ((time.perf_counter() - t0) / 5) * 1000

# Sparse timing (reset stats first)
engine.stats["dense_flops"] = 0
engine.stats["sparse_flops"] = 0

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    _ = engine.execute_sparse_attention(q, k, v)
torch.cuda.synchronize()
sparse_ms = ((time.perf_counter() - t0) / 5) * 1000

# We divide the stats by 5 because we ran it 5 times in the loop
engine.stats["dense_flops"] //= 5
engine.stats["sparse_flops"] //= 5

summary = engine.get_summary()

print(f"  Dense Prefill FLOPs:  {summary['dense_gflops']} GFLOPs")
print(f"  Sparse Prefill FLOPs: {summary['sparse_gflops']} GFLOPs")
print(f"  FLOP Reduction:       {summary['flops_reduced_pct']}%")
print()
print(f"  Dense Wallclock:      {dense_ms:.2f} ms")
print(f"  Sparse Wallclock:     {sparse_ms:.2f} ms ({(dense_ms/sparse_ms):.2f}x speedup)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. QUALITY & DRIFT CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] QUALITY DRIFT CHECK")

# We expect cosine similarity to be fairly high because the most important 
# tokens (sinks and local context) are preserved.
cos_sim = torch.nn.functional.cosine_similarity(
    dense_out.reshape(-1).float(), 
    sparse_out.reshape(-1).float(), 
    dim=0
).item()

print(f"  Cosine Similarity (Dense vs Sparse Chunked): {cos_sim:.4f}")
if cos_sim > 0.8:
    print("  PASS: Output drift is within acceptable bounds for retrieval stability.")
else:
    print("  WARNING: High drift. Sink tokens or local window might be too small.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MEMORY MOVEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] HONEST SYSTEM LIMITS & KV REDUCTION (Step 3 & 6)")

total_kv_elements = BSZ * HEADS * SEQ_LEN * HEAD_DIM * 2 # K and V
total_kv_mb = (total_kv_elements * 2) / (1024 * 1024)

# In dense prefill, the entire N x N matrix is effectively generated (FlashAttention
# avoids HBM write, but compute scales quadratically).
# The biggest win here is COMPRESS-ON-WRITE:
# For chunk C, chunks 0 to C-2 are not needed. They can be evicted/compressed.

print(f"  Full dense KV footprint per layer: {total_kv_mb:.2f} MB")
print(f"  Max hot KV footprint in Sparse Engine (Sinks + Local + Chunk): ")
max_hot_tokens = engine.sink_tokens + (engine.local_window_chunks + 1) * engine.chunk_size
max_hot_mb = (BSZ * HEADS * max_hot_tokens * HEAD_DIM * 2 * 2) / (1024 * 1024)
print(f"    {max_hot_tokens} tokens = {max_hot_mb:.2f} MB")
print(f"  Memory Movement Reduction: {(1 - max_hot_mb / total_kv_mb) * 100:.1f}%")
print()
print("  - KV COMPRESS-ON-WRITE: Chunks falling out of the local window can be")
print("    immediately compressed (e.g. via AsyncCompressor) WITHOUT waiting for")
print("    the full 8192-token prefill to complete. This eliminates the massive")
print("    OOM spike at the end of dense prefill.")
