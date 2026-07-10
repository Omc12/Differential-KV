"""F2 gather-equivalence test for the Triton decode dispatchers (CPU, no GPU needed).

The F2 fix replaced per-token WHOLE-POOL clones (anchors_K, V_K, res_k) with a
gather of only the N routed rows + block_indices remapped to arange(N)
(`_gather_routed_blocks_for_kernel`). The Triton kernels address every per-block
tensor as tensor[block_indices[n]], so the fix is correct iff for every n:

    gathered[x][n] == old_full_pool_rotated[x][block_indices[n]]

This test certifies exactly that, plus the generation-keyed cache semantics.

Run:
    cd ACTIVE_RUNTIME
    python tests/test_triton_gather_equiv.py
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.sparse_decode.triton_fused_decode import (  # noqa: E402
    _gather_routed_blocks_for_kernel,
    _gathered_rot_cache,
    rotate_half,
)

P, N, H_KV, D, R, S, MR, NF, T = 32, 7, 2, 64, 8, 31, 5, 3, 4096


def make_pool(seed=0, with_res=True, with_fact=True):
    torch.manual_seed(seed)
    pool = types.SimpleNamespace()
    pool.anchors_K = torch.randn(P, H_KV, D)
    pool.anchors_V = torch.randn(P, H_KV, D)
    pool.V_K = torch.randn(P, R, H_KV, D)
    pool.V_V = torch.randn(P, R, H_KV, D)
    pool.U = torch.randn(P, S, R)
    pool.U_scale = torch.rand(P) + 0.5
    pool.scales = torch.rand(P) + 0.5
    pool.seq_lens = torch.randint(1, S + 1, (P,), dtype=torch.int32)
    if with_res:
        pool.residual_K_values = torch.randn(P, MR, H_KV, D)
        pool.residual_V_values = torch.randn(P, MR, H_KV, D)
        pool.residual_K_positions = torch.randint(-1, S, (P, MR), dtype=torch.int16)
        pool.residual_V_positions = torch.randint(-1, S, (P, MR), dtype=torch.int16)
    if with_fact:
        pool.fact_anchor_positions = torch.randint(-1, S, (P, NF), dtype=torch.int16)
        pool.fact_anchors_K = torch.randn(P, NF, H_KV, D)
        pool.fact_anchors_V = torch.randn(P, NF, H_KV, D)
    pool._stratified_generation = 0
    return pool


def old_formulation(pool, block_indices, anchor_indices, cos, sin):
    """The pre-F2 dispatcher math: full-pool clone + scatter-rotate at [indices]."""
    indices = block_indices.long()
    anchors_K_rot = pool.anchors_K.clone()
    V_K_rot = pool.V_K.clone()
    res_k = getattr(pool, "residual_K_values", None)
    res_k_rot = res_k.clone() if res_k is not None else None
    if anchor_indices is not None and cos is not None and sin is not None:
        cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
        sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
        clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1)
        cos_anc = cos_flat[clamped].to(dtype=pool.V_K.dtype).unsqueeze(1).unsqueeze(2)
        sin_anc = sin_flat[clamped].to(dtype=pool.V_K.dtype).unsqueeze(1).unsqueeze(2)
        cos_2d = cos_flat[clamped].to(dtype=pool.anchors_K.dtype).unsqueeze(1)
        sin_2d = sin_flat[clamped].to(dtype=pool.anchors_K.dtype).unsqueeze(1)
        V_K_rot[indices] = pool.V_K[indices] * cos_anc + rotate_half(pool.V_K[indices]) * sin_anc
        anchors_K_rot[indices] = pool.anchors_K[indices] * cos_2d + rotate_half(pool.anchors_K[indices]) * sin_2d
        if res_k_rot is not None:
            res_k_rot[indices] = res_k_rot[indices] * cos_anc + rotate_half(res_k_rot[indices]) * sin_anc
    return anchors_K_rot, V_K_rot, res_k_rot


def check_case(with_rot, with_res, with_fact, seed):
    pool = make_pool(seed, with_res=with_res, with_fact=with_fact)
    torch.manual_seed(seed + 99)
    block_indices = torch.randperm(P)[:N].to(torch.int32)
    anchor_indices = torch.randint(0, T, (N,)) if with_rot else None
    cos = torch.randn(T, D) if with_rot else None
    sin = torch.randn(T, D) if with_rot else None

    _gathered_rot_cache.clear()
    g = _gather_routed_blocks_for_kernel(pool, block_indices, anchor_indices, cos, sin)
    anchors_K_old, V_K_old, res_k_old = old_formulation(pool, block_indices, anchor_indices, cos, sin)

    idx = block_indices.long()
    assert torch.equal(g["idx"], torch.arange(N, dtype=block_indices.dtype)), "idx remap"
    checks = [
        ("anchors_K", g["anchors_K"], anchors_K_old[idx]),
        ("V_K", g["V_K"], V_K_old[idx]),
        ("anchors_V", g["anchors_V"], pool.anchors_V[idx]),
        ("V_V", g["V_V"], pool.V_V[idx]),
        ("U", g["U"], pool.U[idx]),
        ("U_scale", g["U_scale"], pool.U_scale[idx]),
        ("scales", g["scales"], pool.scales[idx]),
        ("seq_lens", g["seq_lens"], pool.seq_lens[idx]),
    ]
    if with_res:
        checks += [
            ("res_k", g["res_k"], res_k_old[idx]),
            ("res_v", g["res_v"], pool.residual_V_values[idx]),
            ("res_pos", g["res_pos"], pool.residual_K_positions[idx]),
            ("res_pos_v", g["res_pos_v"], pool.residual_V_positions[idx]),
            ("res_n", g["res_n"], (pool.residual_K_positions[idx] >= 0).sum(dim=-1).to(torch.int32)),
        ]
    else:
        assert g["res_n"].shape == (N,), "empty-res res_n must be N-sized"
    if with_fact:
        checks += [
            ("fact_pos", g["fact_pos"], pool.fact_anchor_positions[idx]),
            ("fact_ak", g["fact_ak"], pool.fact_anchors_K[idx]),
            ("fact_av", g["fact_av"], pool.fact_anchors_V[idx]),
        ]
    for name, new, old in checks:
        assert torch.equal(new, old), f"{name} mismatch (rot={with_rot}, res={with_res})"
    return pool, block_indices, anchor_indices, cos, sin


def check_cache():
    pool, bi, ai, cos, sin = check_case(True, True, True, seed=7)
    g1 = _gather_routed_blocks_for_kernel(pool, bi, ai, cos, sin)
    g2 = _gather_routed_blocks_for_kernel(pool, bi, ai, cos, sin)
    assert g1 is g2, "same gen + same indices must be a cache HIT"

    # Generation bump (pool write) must invalidate
    pool._stratified_generation += 1
    g3 = _gather_routed_blocks_for_kernel(pool, bi, ai, cos, sin)
    assert g3 is not g1, "generation bump must invalidate the cache"

    # Routing change must invalidate AND evict the old entry (bounded cache).
    # anchor_indices are paired with block_indices, so flip both together.
    bi2 = torch.flip(bi, dims=[0])
    ai2 = torch.flip(ai, dims=[0])
    g4 = _gather_routed_blocks_for_kernel(pool, bi2, ai2, cos, sin)
    assert g4 is not g3
    assert len([k for k in _gathered_rot_cache if k[0] == id(pool)]) == 1, \
        "stale entries must be evicted (one live entry per pool)"

    # Order sensitivity: reversed (indices, anchors) produce reversed rows
    assert torch.equal(g4["anchors_K"], torch.flip(g3["anchors_K"], dims=[0])), \
        "cache key must be order-sensitive"


def main():
    for with_rot in (True, False):
        for with_res in (True, False):
            check_case(with_rot, with_res, with_fact=with_res, seed=3)
    check_cache()
    print("All F2 gather-equivalence cases PASS")


if __name__ == "__main__":
    main()
