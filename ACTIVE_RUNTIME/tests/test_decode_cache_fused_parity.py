"""Decode-cache FUSED-path parity test (DIFFKV_DECODE_FUSED).

Validates that the fused persistent-buffer decode path (fp16 storage, exact-length
slicing, -3e4 mask floor, one-row in-place appends) matches the legacy
concat-per-token fp32 decode-cache path on synthetic sessions, including across
simulated multi-token decode (dense-window growth between route intervals).

No language model required — pure MLX math test.

Run:
    cd ACTIVE_RUNTIME
    python tests/test_decode_cache_fused_parity.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from serving.mlx_diffkv_wrapper import MLXKVBlockManager


def cosine_sim(a: mx.array, b: mx.array) -> float:
    a_np = np.array(a.reshape(-1).astype(mx.float32))
    b_np = np.array(b.reshape(-1).astype(mx.float32))
    denom = float(np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom < 1e-9:
        return 1.0
    return float(np.dot(a_np, b_np) / denom)


def build_synthetic_session(mgr: MLXKVBlockManager, nb: int, dense_len: int, seed: int = 0):
    """Fill layer 0 of a fresh session with random-but-consistent compressed blocks."""
    mx.random.seed(seed)
    sid = f"parity_{nb}_{dense_len}_{seed}"
    mgr.sessions.pop(sid, None)
    sess = mgr._get_or_create_session(sid)
    # The hint-less fallback session now starts SMALL (16 blocks) and relies on
    # geometric growth; this test writes `nb` blocks directly, so request the
    # capacity through the same growth path production uses.
    mgr._ensure_block_capacity(sess, nb)
    l = 0
    H, D, bs, R_ = mgr.kv_heads, mgr.head_dim, mgr.block_size, mgr.max_residual
    S_comp = bs - 1
    rank = mgr.rank
    dt = mx.float16

    sess["num_blocks"][l] = nb
    sess["comp_U"][l][:nb] = (mx.random.normal((nb, S_comp, rank)) * 0.05).astype(dt)
    sess["comp_VK"][l][:nb] = mx.random.normal((nb, H, rank, D)).astype(dt)
    sess["comp_VV"][l][:nb] = mx.random.normal((nb, H, rank, D)).astype(dt)
    sess["comp_anc_k"][l][:nb] = mx.random.normal((nb, H, D)).astype(dt)
    sess["comp_anc_v"][l][:nb] = mx.random.normal((nb, H, D)).astype(dt)
    sess["comp_scale"][l][:nb] = mx.abs(mx.random.normal((nb,))) + 0.5
    sess["comp_seq_len"][l][:nb] = S_comp
    sess["comp_res_k"][l][:nb] = mx.random.normal((nb, R_, H, D)).astype(dt)
    sess["comp_res_v"][l][:nb] = mx.random.normal((nb, R_, H, D)).astype(dt)
    for b in range(nb):
        sess["comp_res_n"][l][b] = R_
    # mark a few residual positions per block as exact-captured
    rm = np.zeros((nb, S_comp), dtype=bool)
    rm[:, :: max(1, S_comp // R_)] = True
    sess["comp_res_mask"][l][:nb] = mx.array(rm)

    sess["dense_keys"][l][0, :, :dense_len] = mx.random.normal((H, dense_len, D)).astype(dt)
    sess["dense_values"][l][0, :, :dense_len] = mx.random.normal((H, dense_len, D)).astype(dt)
    sess["dense_lens"][l] = dense_len
    sess["dense_lens_mx"][l] = mx.array(dense_len, dtype=mx.int32)
    mx.eval(sess["comp_U"][l], sess["comp_VK"][l], sess["comp_VV"][l],
            sess["comp_anc_k"][l], sess["comp_anc_v"][l], sess["comp_scale"][l],
            sess["comp_res_k"][l], sess["comp_res_v"][l],
            sess["dense_keys"][l], sess["dense_values"][l])
    return sid, sess


def run_case(mgr, nb, dense_len, steps=20, seed=0):
    """Simulate `steps` decode tokens on the same session with fused ON and OFF.

    Each step appends one new dense row (as ingest_streaming would), then runs
    _execute_decode_cache with a fresh query. Returns min cosine across steps.
    """
    H_q, H, D = mgr.heads, mgr.kv_heads, mgr.head_dim
    scale = 1.0 / (D ** 0.5)
    gpk = H_q // H

    sims = []
    for fused_first in (True,):
        sid, sess = build_synthetic_session(mgr, nb, dense_len, seed)
        # Pre-generate shared inputs for both paths
        mx.random.seed(seed + 1000)
        qs = [mx.random.normal((H_q, D)).astype(mx.float16) for _ in range(steps)]
        rows_k = [mx.random.normal((H, 1, D)).astype(mx.float16) for _ in range(steps)]
        rows_v = [mx.random.normal((H, 1, D)).astype(mx.float16) for _ in range(steps)]
        mx.eval(*qs, *rows_k, *rows_v)

        outs = {}
        for fused in (True, False):
            sid, sess = build_synthetic_session(mgr, nb, dense_len, seed)
            mgr._decode_fused = fused
            sess.pop("_cache_kv", None)
            per_step = []
            dl = dense_len
            for s in range(steps):
                # simulate ingest of this token
                sess["dense_keys"][0][0, :, dl:dl + 1] = rows_k[s]
                sess["dense_values"][0][0, :, dl:dl + 1] = rows_v[s]
                dl += 1
                sess["dense_lens"][0] = dl
                sess["dense_lens_mx"][0] = mx.array(dl, dtype=mx.int32)
                out = mgr._execute_decode_cache(
                    sess, 0, qs[s],
                    sess["dense_keys"][0][0], sess["dense_values"][0][0],
                    sess["dense_lens_mx"][0], scale, gpk)
                mx.eval(out)
                per_step.append(out)
            outs[fused] = per_step
        for s in range(steps):
            sims.append(cosine_sim(outs[True][s], outs[False][s]))
    return min(sims)


def main():
    os.environ.setdefault("DIFFKV_TOPK_BLOCKS", "16")
    mgr = MLXKVBlockManager(num_layers=1, heads=12, kv_heads=2, head_dim=64,
                            rank=16, block_size=64, recency_window=128)
    mgr.max_residual = 16
    mgr.route_residuals = 16
    mgr._decode_cache = True
    mgr._decode_cache_interval = 8   # force a mid-run re-route (steps > interval)

    failures = 0
    # (nb, dense_len): below top-K (no routing), above top-K (routing), tiny dense
    for nb, dl in ((4, 100), (24, 100), (24, 3)):
        sim = run_case(mgr, nb, dl, steps=20)
        ok = sim > 0.999
        print(f"nb={nb:3d} dense_len={dl:3d}  min cosine(fused, legacy) = {sim:.6f}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures += 1
    if failures:
        print(f"\n{failures} case(s) FAILED")
        sys.exit(1)
    print("\nAll fused-parity cases PASS")


if __name__ == "__main__":
    main()
