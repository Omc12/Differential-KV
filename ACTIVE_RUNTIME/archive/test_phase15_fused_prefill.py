"""
test_phase15_fused_prefill.py

Phase 15 Validation: Real Sparse Prefill Execution (Orchestration Collapse)

Validates:
1. Speedup of Fused Sparse Prefill (flex_attention) over Dense SDPA at 16K.
2. Elimination of Python chunk loop overhead.
3. Successful Retrieval Routing validation on the fused kernel.
"""

import sys, time, math, torch
sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 68)
print("PHASE 15 — FUSED SPARSE PREFILL (ORCHESTRATION COLLAPSE)")
print("=" * 68)

if DEVICE != "cuda":
    print("SKIPPED: CUDA required.")
    sys.exit(0)

try:
    from runtime.fused_sparse_prefill import FusedSparsePrefill, FLEX_AVAILABLE
except Exception as e:
    print(f"Failed to import fused_sparse_prefill: {e}")
    FLEX_AVAILABLE = False

if not FLEX_AVAILABLE:
    print("SKIPPED: flex_attention not available in this PyTorch version.")
    sys.exit(0)

BSZ = 1
HEADS = 8
SEQ_LEN = 16384 # 16K
HEAD_DIM = 128
CHUNK_SIZE = 512

torch.manual_seed(42)
q = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1
k = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1
v = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.1

# Plant a Needle (same as Phase 14)
needle_chunk = 5
needle_start = needle_chunk * CHUNK_SIZE
last_chunk_start = 31 * CHUNK_SIZE
q[:, :, last_chunk_start:, :] = k[:, :, needle_start:needle_start+CHUNK_SIZE, :]
v[:, :, needle_start:needle_start+CHUNK_SIZE, :] += 5.0 

fused_engine = FusedSparsePrefill(sink_tokens=512, chunk_size=CHUNK_SIZE, local_chunks=1, top_k_retrieval=1)

print("\n[1] EXECUTION TIMING (16K Context)")

# Dense Baseline
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    out_dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
ms_dense = ((time.perf_counter() - t0) / 5) * 1000

# Fused Sparse
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    out_fused = fused_engine.execute(q, k, v)
torch.cuda.synchronize()
ms_fused = ((time.perf_counter() - t0) / 5) * 1000

print(f"  Dense Wallclock:           {ms_dense:.2f} ms")
print(f"  Fused Sparse Wallclock:    {ms_fused:.2f} ms ({(ms_dense/ms_fused):.2f}x speedup)")

stats = fused_engine.stats
print(f"    - Routing:      {stats['routing_time_ms']:.2f} ms")
print(f"    - Mask Compile: {stats['mask_time_ms']:.2f} ms")
print(f"    - Attention:    {stats['attn_time_ms']:.2f} ms")

print("\n[2] RETRIEVAL ROUTING VALIDATION (Fused Kernel)")

final_dense_mag = out_dense[:, :, last_chunk_start:, :].abs().mean().item()
final_fused_mag = out_fused[:, :, last_chunk_start:, :].abs().mean().item()

print(f"  Needle V-Magnitude in Output (Dense):  {final_dense_mag:.4f}")
print(f"  Needle V-Magnitude in Output (Fused):  {final_fused_mag:.4f}")

if final_fused_mag > 1.0:
    print("  PASS: Fused Anchor Routing successfully retrieved the needle chunk.")
else:
    print("  FAIL: Fused Anchor Routing missed the needle.")
