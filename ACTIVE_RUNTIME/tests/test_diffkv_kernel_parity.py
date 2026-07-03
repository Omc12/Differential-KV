"""
Phase 1 — DiffKV Kernel Parity Test
====================================
Validates that `compute_decode_attention_static` (the real DiffKV sparse decode kernel)
produces attention outputs close to exact dense attention.

No language model required — pure MLX math test.

Run:
    cd ACTIVE_RUNTIME
    python tests/test_diffkv_kernel_parity.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from serving.mlx_diffkv_wrapper import (
    compute_decode_attention_static,
    MLXKVBlockManager,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def cosine_sim(a: mx.array, b: mx.array) -> float:
    """Cosine similarity between two flat vectors."""
    a_np = np.array(a.reshape(-1).astype(mx.float32))
    b_np = np.array(b.reshape(-1).astype(mx.float32))
    denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom < 1e-9:
        return 1.0
    return float(np.dot(a_np, b_np) / denom)


def exact_attention(q, k, v, scale):
    """
    Exact multi-head attention.
    q: [H_q, D]
    k: [H_kv, S, D]
    v: [H_kv, S, D]
    Returns: [H_q, D]
    """
    gpk = q.shape[0] // k.shape[0]
    if gpk > 1:
        k = mx.repeat(k, gpk, axis=0)  # [H_q, S, D]
        v = mx.repeat(v, gpk, axis=0)
    scores = mx.sum(mx.expand_dims(q, 1) * k, axis=-1) * scale  # [H_q, S]
    weights = mx.softmax(scores, axis=-1)
    out = mx.sum(mx.expand_dims(weights, -1) * v, axis=1)       # [H_q, D]
    return out


def build_diffkv_store_from_kv(
    manager: MLXKVBlockManager,
    session_id: str,
    layer_idx: int,
    K: mx.array,  # [1, H_kv, S, D]
    V: mx.array,  # [1, H_kv, S, D]
):
    """Ingest a full K/V sequence into the DiffKV manager token-by-token."""
    S = K.shape[2]
    for t in range(S):
        k_t = K[:, :, t:t+1, :]
        v_t = V[:, :, t:t+1, :]
        manager.ingest_streaming(session_id, layer_idx, k_t, v_t)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_dense_only_no_compression():
    """
    Sequence fits entirely in the recency window — no block is flushed.
    Dense path only. Should be essentially exact (cosine > 0.999).
    """
    print("\n─── Test 1: Dense-only path (no compression triggered) ───")

    H_kv, H_q, D = 2, 4, 64
    RANK, BLOCK_SIZE, RECENCY = 16, 256, 512
    S = 100  # well under recency window

    rng = np.random.default_rng(17)
    K_np = rng.standard_normal((1, H_kv, S, D)).astype(np.float32) * 0.1
    V_np = rng.standard_normal((1, H_kv, S, D)).astype(np.float32) * 0.1
    Q_np = rng.standard_normal((H_q, D)).astype(np.float32) * 0.1

    K = mx.array(K_np.astype(np.float16))
    V = mx.array(V_np.astype(np.float16))
    Q = mx.array(Q_np.astype(np.float16))
    scale = 1.0 / math.sqrt(D)

    out_exact = exact_attention(Q, K[0], V[0], scale)
    mx.eval(out_exact)

    mgr = MLXKVBlockManager(
        num_layers=1, heads=H_q, kv_heads=H_kv, head_dim=D,
        rank=RANK, block_size=BLOCK_SIZE, recency_window=RECENCY
    )
    mgr.max_blocks = 256
    mgr.max_dense_len = RECENCY + BLOCK_SIZE

    sid = "dense_only"
    mgr.init_session(sid)
    build_diffkv_store_from_kv(mgr, sid, 0, K, V)

    nb = mgr.sessions[sid]["num_blocks"][0]
    dl = mgr.sessions[sid]["dense_lens"][0]
    print(f"  Compressed blocks: {nb} (expect 0), Dense tokens: {dl} (expect {S})")
    assert nb == 0, f"Expected 0 compressed blocks, got {nb}"

    out_4d = mgr.execute_decode_attention(
        sid, 0, Q.reshape(1, H_q, 1, D), rope=None,
        scale=scale, num_key_value_groups=H_q // H_kv
    )
    out_diffkv = out_4d[0, :, 0, :]
    mx.eval(out_diffkv)

    sim = cosine_sim(out_exact, out_diffkv)
    print(f"  Cosine similarity: {sim:.6f}")
    assert sim > 0.999, f"FAIL: dense-only cosine {sim:.6f} < 0.999"
    print("  PASS ✓")
    return sim


def test_single_block_parity():
    """
    Exactly one 256-token block compressed to SVD rank 16 + empty dense window.
    Expected cosine similarity > 0.99.
    """
    print("\n─── Test 2: Single compressed block ───")

    H_kv, H_q, D = 2, 4, 64
    RANK, BLOCK_SIZE, RECENCY = 16, 256, 512
    S = BLOCK_SIZE

    rng = np.random.default_rng(42)
    K_np = rng.standard_normal((1, H_kv, S, D)).astype(np.float32) * 0.1
    V_np = rng.standard_normal((1, H_kv, S, D)).astype(np.float32) * 0.1
    Q_np = rng.standard_normal((H_q, D)).astype(np.float32) * 0.1

    K = mx.array(K_np.astype(np.float16))
    V = mx.array(V_np.astype(np.float16))
    Q = mx.array(Q_np.astype(np.float16))
    scale = 1.0 / math.sqrt(D)

    out_exact = exact_attention(Q, K[0], V[0], scale)
    mx.eval(out_exact)

    mgr = MLXKVBlockManager(
        num_layers=1, heads=H_q, kv_heads=H_kv, head_dim=D,
        rank=RANK, block_size=BLOCK_SIZE, recency_window=RECENCY
    )
    mgr.max_blocks = 256
    mgr.max_dense_len = RECENCY + BLOCK_SIZE

    sid = "single_block"
    mgr.init_session(sid)
    build_diffkv_store_from_kv(mgr, sid, 0, K, V)

    nb = mgr.sessions[sid]["num_blocks"][0]
    dl = mgr.sessions[sid]["dense_lens"][0]
    print(f"  Compressed blocks: {nb}, Dense tokens: {dl}")

    out_4d = mgr.execute_decode_attention(
        sid, 0, Q.reshape(1, H_q, 1, D), rope=None,
        scale=scale, num_key_value_groups=H_q // H_kv
    )
    out_diffkv = out_4d[0, :, 0, :]
    mx.eval(out_diffkv)

    sim = cosine_sim(out_exact, out_diffkv)
    print(f"  Cosine similarity: {sim:.6f}")
    assert sim > 0.99, f"FAIL: single-block cosine {sim:.4f} < 0.99"
    print("  PASS ✓")
    return sim


def test_multi_block_parity():
    """
    8 compressed blocks (2048 tokens) + partial dense window (128 tokens).
    Expected cosine similarity > 0.90.
    """
    print("\n─── Test 3: Multi-block (8 blocks + dense window) ───")

    H_kv, H_q, D = 2, 4, 64
    RANK, BLOCK_SIZE, RECENCY = 16, 256, 512
    N_BLOCKS = 8
    S_DENSE = 128
    S = N_BLOCKS * BLOCK_SIZE + S_DENSE

    rng = np.random.default_rng(7)
    K_np = rng.standard_normal((1, H_kv, S, D)).astype(np.float32) * 0.1
    V_np = rng.standard_normal((1, H_kv, S, D)).astype(np.float32) * 0.1
    Q_np = rng.standard_normal((H_q, D)).astype(np.float32) * 0.1

    K = mx.array(K_np.astype(np.float16))
    V = mx.array(V_np.astype(np.float16))
    Q = mx.array(Q_np.astype(np.float16))
    scale = 1.0 / math.sqrt(D)

    out_exact = exact_attention(Q, K[0], V[0], scale)
    mx.eval(out_exact)

    mgr = MLXKVBlockManager(
        num_layers=1, heads=H_q, kv_heads=H_kv, head_dim=D,
        rank=RANK, block_size=BLOCK_SIZE, recency_window=RECENCY
    )
    mgr.max_blocks = 256
    mgr.max_dense_len = RECENCY + BLOCK_SIZE

    sid = "multi_block"
    mgr.init_session(sid)
    build_diffkv_store_from_kv(mgr, sid, 0, K, V)

    nb = mgr.sessions[sid]["num_blocks"][0]
    dl = mgr.sessions[sid]["dense_lens"][0]
    print(f"  Compressed blocks: {nb}, Dense tokens: {dl}")

    out_4d = mgr.execute_decode_attention(
        sid, 0, Q.reshape(1, H_q, 1, D), rope=None,
        scale=scale, num_key_value_groups=H_q // H_kv
    )
    out_diffkv = out_4d[0, :, 0, :]
    mx.eval(out_diffkv)

    sim = cosine_sim(out_exact, out_diffkv)
    print(f"  Cosine similarity: {sim:.6f}")
    assert sim > 0.90, f"FAIL: multi-block cosine {sim:.4f} < 0.90"
    print("  PASS ✓")
    return sim


def test_needle_in_compressed_block():
    """
    Plant a distinctive 'needle' token at position 128 of a 256-token block.
    Use a query aligned with the needle's K. DiffKV must preserve the signal.
    Expected cosine similarity > 0.90.
    """
    print("\n─── Test 4: Needle survival through SVD compression ───")

    H_kv, H_q, D = 2, 4, 64
    RANK, BLOCK_SIZE, RECENCY = 16, 256, 512
    S = BLOCK_SIZE
    needle_pos = 128

    rng = np.random.default_rng(99)
    K_np = rng.standard_normal((1, H_kv, S, D)).astype(np.float32) * 0.1
    V_np = rng.standard_normal((1, H_kv, S, D)).astype(np.float32) * 0.1

    # Plant needle: very large, distinctive K and V
    K_np[:, :, needle_pos:needle_pos+1, :] = 5.0
    V_np[:, :, needle_pos:needle_pos+1, :] = 3.0

    # Query aligned with needle's K direction
    Q_np = np.ones((H_q, D), dtype=np.float32) * 5.0

    K = mx.array(K_np.astype(np.float16))
    V = mx.array(V_np.astype(np.float16))
    Q = mx.array(Q_np.astype(np.float16))
    scale = 1.0 / math.sqrt(D)

    out_exact = exact_attention(Q, K[0], V[0], scale)
    mx.eval(out_exact)

    mgr = MLXKVBlockManager(
        num_layers=1, heads=H_q, kv_heads=H_kv, head_dim=D,
        rank=RANK, block_size=BLOCK_SIZE, recency_window=RECENCY
    )
    mgr.max_blocks = 256
    mgr.max_dense_len = RECENCY + BLOCK_SIZE

    sid = "needle"
    mgr.init_session(sid)
    build_diffkv_store_from_kv(mgr, sid, 0, K, V)

    out_4d = mgr.execute_decode_attention(
        sid, 0, Q.reshape(1, H_q, 1, D), rope=None,
        scale=scale, num_key_value_groups=H_q // H_kv
    )
    out_diffkv = out_4d[0, :, 0, :]
    mx.eval(out_diffkv)

    sim = cosine_sim(out_exact, out_diffkv)
    exact_norm = float(np.linalg.norm(np.array(out_exact.astype(mx.float32))))
    diffkv_norm = float(np.linalg.norm(np.array(out_diffkv.astype(mx.float32))))
    print(f"  Output norm — exact: {exact_norm:.4f}, DiffKV: {diffkv_norm:.4f}")
    print(f"  Cosine similarity: {sim:.6f}")
    assert sim > 0.90, f"FAIL: needle cosine {sim:.4f} < 0.90"
    print("  PASS ✓")
    return sim
# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DiffKV Kernel Parity Test — Phase 1")
    print("=" * 60)

    tests = [
        ("dense_only",      test_dense_only_no_compression),
        ("single_block",    test_single_block_parity),
        ("multi_block",     test_multi_block_parity),
        ("needle_survival", test_needle_in_compressed_block),
    ]

    results = {}
    failures = []

    for name, fn in tests:
        try:
            sim = fn()
            results[name] = sim
        except AssertionError as e:
            print(f"  ✗ {e}")
            failures.append((name, str(e)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append((name, str(e)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, fn in tests:
        if name in results:
            print(f"  PASS  {name:<25}  cosine_sim = {results[name]:.6f}")
        else:
            msg = next((m for n, m in failures if n == name), "exception")
            print(f"  FAIL  {name:<25}  {msg}")

    if failures:
        print(f"\n{len(failures)} test(s) FAILED.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} tests PASSED ✓")
        print("→ Kernel is sound. Safe to wire into live inference.")
