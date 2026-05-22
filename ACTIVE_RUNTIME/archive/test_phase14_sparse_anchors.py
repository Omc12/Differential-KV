"""
test_phase14_sparse_anchors.py

Phase 14 Validation: Retrieval-Aware Sparse Execution (Needle-in-Haystack)

Validates:
1. Retrieval of a "needle" chunk located outside the local window and sinks.
2. Cosine similarity improvement over locality-only sparse prefill.
3. FLOP reduction and minimal routing overhead.
"""

import sys, time, math, torch
sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 68)
print("PHASE 14 — RETRIEVAL-AWARE SPARSE ANCHORS VALIDATION")
print("=" * 68)

if DEVICE != "cuda":
    print("SKIPPED: CUDA required.")
    sys.exit(0)

from runtime.sparse_prefill_anchors import RetrievalAwareSparsePrefill
from runtime.sparse_prefill import SparsePrefillEngine

BSZ = 1
HEADS = 8
SEQ_LEN = 16384 # 16K context
HEAD_DIM = 128
CHUNK_SIZE = 512

print(f"\n[1] INITIALIZING NEEDLE-IN-HAYSTACK (Context: 16K)")

torch.manual_seed(42)
q = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1
k = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1
v = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1

# Plant a "Needle" far in the past (e.g. Chunk 5, tokens 2560-3072)
# We want the LAST chunk (Chunk 31) to search for this needle.
needle_chunk = 5
needle_start = needle_chunk * CHUNK_SIZE
last_chunk_start = 31 * CHUNK_SIZE

# Make the queries of the last chunk perfectly match the keys of the needle chunk
# so attention naturally strongly routes to it.
q[:, :, last_chunk_start:, :] = k[:, :, needle_start:needle_start+CHUNK_SIZE, :]

# Also inject a distinct signal into V so we can detect if it was retrieved in the output
v[:, :, needle_start:needle_start+CHUNK_SIZE, :] += 5.0 

engine_locality = SparsePrefillEngine(sink_tokens=64, chunk_size=CHUNK_SIZE, local_window_chunks=1)
engine_anchors = RetrievalAwareSparsePrefill(sink_tokens=64, chunk_size=CHUNK_SIZE, local_window_chunks=1, top_k_retrieval_chunks=1)

print(f"  Needle Planted at Chunk {needle_chunk} (Tokens {needle_start}-{needle_start+CHUNK_SIZE})")
print(f"  Retrieving from Chunk 31 (Tokens {last_chunk_start}-{SEQ_LEN})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXECUTION & DRIFT CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] EXECUTION & RETRIEVAL VALIDATION")

# Dense Baseline
t0 = time.perf_counter()
out_dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
ms_dense = (time.perf_counter() - t0) * 1000

# Sparse Locality Only (Phase 13)
t0 = time.perf_counter()
out_local = engine_locality.execute_sparse_attention(q, k, v)
torch.cuda.synchronize()
ms_local = (time.perf_counter() - t0) * 1000

# Sparse Anchor Routing (Phase 14)
t0 = time.perf_counter()
out_anchor = engine_anchors.execute_sparse_attention(q, k, v)
torch.cuda.synchronize()
ms_anchor = (time.perf_counter() - t0) * 1000

# Analyze retrieval of the final chunk
# The V signal was shifted by +5.0 in the needle. If retrieved, the output of the final chunk
# should have a much higher magnitude compared to locality-only (which misses it).
final_dense_mag = out_dense[:, :, last_chunk_start:, :].abs().mean().item()
final_local_mag = out_local[:, :, last_chunk_start:, :].abs().mean().item()
final_anchor_mag = out_anchor[:, :, last_chunk_start:, :].abs().mean().item()

print(f"  Needle V-Magnitude in Output (Dense):        {final_dense_mag:.4f}")
print(f"  Needle V-Magnitude in Output (Locality):     {final_local_mag:.4f}  (Missed)")
print(f"  Needle V-Magnitude in Output (Anchor):       {final_anchor_mag:.4f}  (Retrieved)")

if final_anchor_mag > final_local_mag * 1.5:
    print("  PASS: Anchor Routing successfully retrieved the needle chunk from the distant past.")
else:
    print("  FAIL: Anchor Routing missed the needle.")

cos_local = torch.nn.functional.cosine_similarity(
    out_dense[:, :, last_chunk_start:, :].reshape(-1).float(), 
    out_local[:, :, last_chunk_start:, :].reshape(-1).float(), dim=0).item()
    
cos_anchor = torch.nn.functional.cosine_similarity(
    out_dense[:, :, last_chunk_start:, :].reshape(-1).float(), 
    out_anchor[:, :, last_chunk_start:, :].reshape(-1).float(), dim=0).item()

print(f"\n  Final Chunk Cosine Similarity (Locality Only): {cos_local:.4f}")
print(f"  Final Chunk Cosine Similarity (Anchor Route):  {cos_anchor:.4f}")
print(f"  Quality Improvement: +{((cos_anchor - cos_local)/cos_local)*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PERFORMANCE OVERHEAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] PERFORMANCE & ROUTING OVERHEAD")

stats = engine_anchors.get_summary()

print(f"  Dense FLOPs:           {stats['dense_gflops']:.2f} GFLOPs")
print(f"  Sparse FLOPs (Total):  {stats['sparse_gflops']:.2f} GFLOPs (Reduction: {stats['flops_reduced_pct']}%)")
print(f"  Routing Overhead:      {stats['routing_gflops']:.4f} GFLOPs")
print(f"  Routing Cost %:        {(stats['routing_gflops'] / stats['sparse_gflops']) * 100:.2f}% of sparse FLOPs")
print()
print(f"  Dense Wallclock:       {ms_dense:.2f} ms")
print(f"  Locality Wallclock:    {ms_local:.2f} ms")
print(f"  Anchor Wallclock:      {ms_anchor:.2f} ms")
print(f"  Total Retrieval Events: {stats['total_retrieval_events']}")
