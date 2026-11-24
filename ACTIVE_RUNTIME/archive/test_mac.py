#!/usr/bin/env python3
"""
test_mac.py — DKV Mac/Apple Silicon Integration Test

Validates that the entire DKV pipeline (compression, attention patch,
decode, streaming ingest) works correctly on Apple Silicon (MPS) and CPU,
using the tiny Qwen/Qwen2.5-0.5B-Instruct model.

Usage:
    cd ACTIVE_RUNTIME/
    python test_mac.py

Environment variables:
    DKV_USE_TORCH_COMPILE=0   skip torch.compile (faster startup)
    DKV_TELEMETRY=1           verbose block-state logging
"""

import os
import sys
import time

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Disable tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Skip torch.compile for faster test startup
os.environ.setdefault("DKV_USE_TORCH_COMPILE", "0")

import torch

# ── 1. Platform diagnostics ───────────────────────────────────────────────────
print("=" * 60)
print("  DKV Mac Integration Test")
print("=" * 60)
print(f"  PyTorch : {torch.__version__}")
print(f"  Python  : {sys.version.split()[0]}")
print(f"  Platform: {sys.platform}")

cuda_ok = torch.cuda.is_available()
mps_ok  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"  CUDA    : {'YES' if cuda_ok else 'no'}")
print(f"  MPS     : {'YES' if mps_ok  else 'no'}")

try:
    import mlx
    try:
        ver = mlx.__version__
    except AttributeError:
        try:
            import mlx.core as mx
            ver = mx.__version__
        except AttributeError:
            ver = "installed"
    print(f"  MLX     : YES ({ver})")
except ImportError:
    print("  MLX     : not installed (optional — install with: pip install mlx)")

from native_core.mac_utils import get_best_device
DEVICE = get_best_device()
print(f"  Device  : {DEVICE}")
print()

# ── 2. Low-level unit tests ───────────────────────────────────────────────────

def test_lowrank_compression():
    print("[TEST] Low-rank compression (compress_lowrank) ...")
    from native_core.compression.lowrank import compress_lowrank
    import torch

    # Generate a truly low-rank matrix (rank 8) so that rank 8 approximation has small error
    u = torch.randn(32, 8, dtype=torch.float32)
    v = torch.randn(8, 256, dtype=torch.float32)
    deltas = u @ v
    lr = compress_lowrank(deltas, rank=8)

    assert lr.U.shape == (32, 8), f"U shape mismatch: {lr.U.shape}"
    assert lr.V.shape == (8, 256), f"V shape mismatch: {lr.V.shape}"
    assert lr.energy_retained > 0, "Energy retained should be > 0"

    recon = (lr.U.float() @ lr.V.float()) * lr.scale
    rel_err = (recon - deltas).norm() / (deltas.norm() + 1e-12)
    print(f"    rank={lr.dynamic_rank}  energy_retained={lr.energy_retained:.3f}  rel_err={rel_err.item():.4f}")
    assert rel_err < 0.5, f"Reconstruction error too high: {rel_err.item():.4f}"
    print("    PASS\n")


def test_native_block_pool():
    print("[TEST] NativeBlockPool allocation/write/free ...")
    from runtime.native_block_pool import NativeBlockPool

    pool = NativeBlockPool(
        max_blocks=64,
        num_kv_heads=4,
        head_dim=64,
        rank=8,
        max_seq_len=32,
        device=DEVICE,
        dtype=torch.float16,
        initial_blocks=8,
    )

    idx = pool.allocate_block()
    assert idx is not None

    U  = torch.zeros((32, 8), dtype=torch.float16, device=DEVICE)
    V  = torch.zeros((8, 4 * 64 * 2), dtype=torch.float16, device=DEVICE)
    aK = torch.zeros((4, 64), dtype=torch.float16, device=DEVICE)
    aV = torch.zeros((4, 64), dtype=torch.float16, device=DEVICE)
    pool.write_block(idx, U, V, aK, aV, scale=1.0, seq_len=32)

    pool.free_block(idx)
    print("    PASS\n")


def test_pytorch_sparse_attn_decode():
    print("[TEST] PyTorch vectorized sparse attention decode fallback ...")
    from native_core.sparse_decode.triton_fused_decode import _pytorch_vectorized_sparse_attn_decode
    from runtime.native_block_pool import NativeBlockPool

    H_q, D = 8, 64
    R, S_MAX = 8, 16
    N_BLOCKS = 4

    pool = NativeBlockPool(
        max_blocks=64, num_kv_heads=4, head_dim=D,
        rank=R, max_seq_len=S_MAX,
        device=DEVICE, dtype=torch.float16, initial_blocks=8,
    )

    idxs = pool.allocate_blocks(N_BLOCKS)
    for idx in idxs:
        U  = torch.randn((S_MAX, R), dtype=torch.float16, device=DEVICE) * 0.01
        V  = torch.randn((R, 4 * D * 2), dtype=torch.float16, device=DEVICE) * 0.01
        aK = torch.randn((4, D), dtype=torch.float16, device=DEVICE) * 0.01
        aV = torch.randn((4, D), dtype=torch.float16, device=DEVICE) * 0.01
        pool.write_block(idx, U, V, aK, aV, scale=1.0, seq_len=S_MAX)

    block_indices = torch.tensor(idxs, dtype=torch.int32, device=DEVICE)
    q = torch.randn((1, H_q, 1, D), dtype=torch.float16, device=DEVICE)

    out = _pytorch_vectorized_sparse_attn_decode(
        q=q,
        block_indices=block_indices,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=2,
        R=R,
        S_MAX=S_MAX,
    )
    assert out.shape == (1, H_q, 1, D), f"Output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf!"
    print(f"    out.shape={out.shape}  max={out.abs().max().item():.4f}")
    print("    PASS\n")


# ── 3. Full end-to-end model test ─────────────────────────────────────────────

def test_full_pipeline():
    print("[TEST] Full pipeline with Qwen/Qwen2.5-0.5B-Instruct ...")
    print("  (This will download ~1 GB on first run — using HuggingFace cache)")

    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    from serving.hf_dkv_wrapper import DKVHFWrapper

    config = {
        "rank": 32,
        "micro_block_size": 256,
        "serving_mode": "balanced",
    }

    t0 = time.perf_counter()
    wrapper = DKVHFWrapper(
        model_id=MODEL_ID,
        config=config,
        device=DEVICE,
    )
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Short prompt — just a few tokens to exercise the full pipeline
    prompt = "Hello! Tell me one interesting fact about the Eiffel Tower."
    print(f"  Prompt: {prompt!r}")

    t1 = time.perf_counter()
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=64,
        temperature=0.0,   # greedy for determinism
        top_p=1.0,
        repetition_penalty=1.0,
    )
    gen_time = time.perf_counter() - t1
    print(f"  Response ({gen_time:.1f}s): {response!r}")

    assert isinstance(response, str) and len(response) > len(prompt), \
        "Model generated no meaningful output!"

    # Multi-turn: run again on the same wrapper to exercise decode caching
    print("\n  Multi-turn test (second prompt) ...")
    multi_turn_prompt = response + "\nWhat year was it built?"
    response2 = wrapper.generate(
        prompt=multi_turn_prompt,
        max_new_tokens=32,
        temperature=0.0,
        repetition_penalty=1.0,
    )
    print(f"  Response2: {response2!r}")

    wrapper.stop()
    print("    PASS\n")


# ── 4. Run all tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    errors = []

    tests = [
        ("Low-rank compression",     test_lowrank_compression),
        ("NativeBlockPool",          test_native_block_pool),
        ("PyTorch sparse attn decode", test_pytorch_sparse_attn_decode),
        ("Full pipeline (0.5B model)", test_full_pipeline),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  [FAIL] {name}: {exc}")
            traceback.print_exc()
            errors.append(name)
            print()

    print("=" * 60)
    if errors:
        print(f"  FAILED: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("  All tests PASSED ✓")
        print(f"  Device used: {DEVICE}")
    print("=" * 60)
