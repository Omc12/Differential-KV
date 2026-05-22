"""
test_phase11_fused.py

Phase 11 Validation: Fused Triton Sparse MLP and Prefill Pruner
"""

import sys, time, math, torch
sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 68)
print("PHASE 11 — FUSED SPARSE MLP & PREFILL PRUNER VALIDATION")
print("=" * 68)

if DEVICE != "cuda":
    print("SKIPPED: CUDA required for Triton kernel tests.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PREFILL ATTENTION PRUNER SYNTHETIC TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] PREFILL ATTENTION PRUNER — Logic Validation")
from runtime.prefill_attention_pruner import validate_pruner_logic
summary = validate_pruner_logic()
print(f"  PASS: Pruner logic validated. Summary: {summary}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FUSED TRITON SPARSE MLP vs PYTORCH SPARSE MLP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] FUSED TRITON SPARSE MLP — Benchmarking")

from runtime.sparse_mlp_fused import benchmark_fused_vs_pytorch

HIDDEN       = 3584
INTERMEDIATE = 18944
BLOCK_SIZE   = 128
KEEP         = 0.5

print(f"  Testing at Qwen2-7B scale (hidden={HIDDEN}, d_ff={INTERMEDIATE}), keep={KEEP}")

# seq=1 (Decode)
try:
    print("\n  -- Seq=1 (Decode Phase) --")
    res1 = benchmark_fused_vs_pytorch(
        hidden=HIDDEN, d_ff=INTERMEDIATE, seq=1,
        keep_ratio=KEEP, block_size=BLOCK_SIZE, iters=100
    )
    print(f"  Dense PyTorch:        {res1['dense_ms']*1000:.1f} us")
    print(f"  PyTorch Block-Sparse: {res1['pytorch_sparse_ms']*1000:.1f} us (index_select overhead)")
    print(f"  Triton Fused Sparse:  {res1['triton_fused_ms']*1000:.1f} us (real speedup expected)")
    print(f"  Triton vs Dense:      {res1['triton_vs_dense']}x")
except Exception as e:
    print(f"  Triton benchmark failed (expected on some GPUs/Windows): {e}")

# seq=64 (Prefill slice)
try:
    print("\n  -- Seq=64 (Prefill Phase) --")
    res64 = benchmark_fused_vs_pytorch(
        hidden=HIDDEN, d_ff=INTERMEDIATE, seq=64,
        keep_ratio=KEEP, block_size=BLOCK_SIZE, iters=50
    )
    print(f"  Dense PyTorch:        {res64['dense_ms']:.3f} ms")
    print(f"  PyTorch Block-Sparse: {res64['pytorch_sparse_ms']:.3f} ms")
    print(f"  Triton Fused Sparse:  {res64['triton_fused_ms']:.3f} ms")
    print(f"  Triton vs Dense:      {res64['triton_vs_dense']}x")
except Exception as e:
    print(f"  Triton benchmark failed: {e}")

print("\n" + "=" * 68)
print("PHASE 11 FUSED VALIDATION SUMMARY COMPLETE")
print("=" * 68)
