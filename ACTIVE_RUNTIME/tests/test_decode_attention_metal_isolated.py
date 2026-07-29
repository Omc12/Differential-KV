"""
Isolated unit test for dkv_core.decode_attention_metal (the compiled Metal shader
in native_core/dkv_core/metal/dkv_decode.metal -- the CUDA/PyTorch track's MPS
acceleration kernel, NOT MLX).

Compares the real compiled kernel's output against a faithful line-by-line
NumPy reference of the same algorithm (transliterated directly from the .metal
source), isolated from the model/tokenizer/routing/compression stack entirely.
Reference math runs in float64; both sides start from identical fp16-rounded
inputs so the comparison isolates kernel arithmetic bugs from fp16 precision
noise. Tolerance is magnitude-scaled (atol + rtol*|ref|), not a flat cap --
fp16 error scales with value magnitude.

Covers, as separate cases: partial RoPE, rank < pool_rank (the addressing-
stride class of bug), GQA, residual K/V corrections, fact anchor overrides,
dense window with/without RoPE, zero active compressed slots with a populated
dense window (the has_dense_rope/has_rope decoupling class of bug), and a
realistic-scale case matching Qwen3.5-2B's actual mid-layer config.

Run:
    cd ACTIVE_RUNTIME
    python tests/test_decode_attention_metal_isolated.py
"""
import os
import sys

# Exercise the STRICT dense bound (DKV_DENSE_VALID_LEN=1). The kernel defaults
# this OFF in production because assemble_dense_window_kv's cached write offsets
# can park live tokens past the valid COUNT -- see the .metal comment. These are
# kernel-correctness tests with a densely-packed workspace, which is the layout
# the strict bound assumes, so strict is the right mode here. Must be set before
# dkv_core is imported: the flag is cached on first kernel launch.
os.environ.setdefault("DKV_DENSE_VALID_LEN", "1")

# DKV_RESIDUAL_EXACT_KEYS switches residual semantics from "correct the low-rank
# reconstruction" to "substitute the exact value" (MLX parity). The kernel caches
# it on first launch, so it cannot be toggled mid-process -- the suite runs the
# exact-keys cases by re-executing this file in a subprocess with it set (see
# __main__). The reference implementation branches on the same flag.
# Default "1" MIRRORS THE RUNTIME DEFAULT (compression/lowrank._exact_keys_enabled
# and metal_runtime.dkv_residual_exact_keys, both ON to match MLX). If this drifts
# from those, the reference computes one residual convention while the kernel
# applies the other and every residual case fails for the wrong reason.
EXACT_KEYS = os.environ.get("DKV_RESIDUAL_EXACT_KEYS", "1") == "1"

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# There is a stale `dkv_core` editable pip install registered in site-packages
# (__editable__.dkv_core-*.pth) pointing at an old, unrelated build at the
# repo root -- a bare `import dkv_core` resolves there instead of the actual
# development build below, silently running stale kernel code with no error.
# Insert this path FIRST so it always wins.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "native_core", "dkv_core"))

dkv_core = pytest.importorskip(
    "dkv_core", reason="dkv_core native extension not built (run setup.py build_ext --inplace)"
)
if not getattr(dkv_core, "HAS_METAL_ATTN", False):
    pytest.skip("Metal decode kernel not available on this platform", allow_module_level=True)

import torch

DEVICE = "mps"


def rope_rotate_dim_ref(vec, cos_row, sin_row, rotary_dim, has_rope):
    """Faithful copy of dkv_rope_rotate_dim (float64, one vector at a time)."""
    out = vec.copy()
    if not has_rope:
        return out
    half_r = rotary_dim // 2
    for d in range(rotary_dim):
        partner = d + half_r if d < half_r else d - half_r
        partner_contrib = -vec[partner] if d < half_r else vec[partner]
        out[d] = vec[d] * cos_row[d] + partner_contrib * sin_row[d]
    return out


def reference_decode_attention(
    Q, U_pool, U_scale_pool, VK_pool, VV_pool, anchors_K, anchors_V,
    seq_lens, scales, cos_anc, sin_anc, slot_indices, scale,
    n_q_heads, n_kv_heads, rank,
    res_pos_K, res_val_K, res_pos_V, res_val_V,
    fact_pos, fact_val_K, fact_val_V,
    dense_K, dense_V, cos_dense, sin_dense, rotary_dim,
    cos_full=None, sin_full=None, anchor_pos=None,
):
    """
    Direct (non-online) softmax reference -- mathematically identical to the
    kernel's incremental merge_softmax_states approach, just computed in one
    pass since we don't need numerical-stability tricks for tiny test tensors.
    All accumulation in float64. Returns [H_q, D] float64.
    """
    H_q, D = Q.shape
    g = n_q_heads // n_kv_heads
    K_active = slot_indices.shape[0]
    max_residual = res_pos_K.shape[1] if res_pos_K.size else 0
    # dense_K may be a PADDED workspace; the number of REAL tokens is the
    # cos/sin table's row count (that is how the host sizes it). Mirrors the
    # kernel's L_dense_valid.
    L_dense_padded = dense_K.shape[1] if dense_K.size else 0
    L_dense = (cos_dense.shape[0]
               if (cos_dense is not None and cos_dense.size > 0
                   and cos_dense.shape[0] <= L_dense_padded)
               else L_dense_padded)
    has_rope = cos_anc.size > 0
    has_dense_rope = cos_dense.size > 0
    # Exact-position RoPE for residual/fact overrides: rotate them at their TRUE
    # absolute token position (anchor + within-block offset) using the raw
    # full-sequence tables, instead of at the block anchor. Mirrors the kernel's
    # has_exact_res_rope branch.
    has_exact = (cos_full is not None and cos_full.size > 0 and
                 sin_full is not None and sin_full.size > 0 and
                 anchor_pos is not None and anchor_pos.size > 0)

    def _ovr_rope(k, offset):
        """Return (cos_row, sin_row) for a position-specific override."""
        if has_exact:
            row = int(np.clip(int(anchor_pos[k]) + int(offset), 0, cos_full.shape[0] - 1))
            return cos_full[row], sin_full[row]
        return cos_anc[k], sin_anc[k]

    out = np.zeros((H_q, D), dtype=np.float64)

    for h in range(H_q):
        kv_head = h // g
        q = Q[h].astype(np.float64)

        # source tags: ("anc", k) | ("tok", k, t) | ("dense", t)
        terms = []
        per_slot_data = []  # per-k: (slot_id, score_anc, q_proj[rank], slen, scale_u, block_scale)

        for k in range(K_active):
            slot_id = int(slot_indices[k])
            slen = int(seq_lens[slot_id])
            scale_u = float(U_scale_pool[slot_id])
            block_scale = float(scales[slot_id])

            ak_rot = rope_rotate_dim_ref(
                anchors_K[slot_id, kv_head].astype(np.float64),
                cos_anc[k] if has_rope else None, sin_anc[k] if has_rope else None,
                rotary_dim, has_rope)
            score_anc = float(np.dot(q, ak_rot))

            q_proj = np.zeros(rank, dtype=np.float64)
            for r in range(rank):
                vk_rot = rope_rotate_dim_ref(
                    VK_pool[slot_id, r, kv_head].astype(np.float64),
                    cos_anc[k] if has_rope else None, sin_anc[k] if has_rope else None,
                    rotary_dim, has_rope)
                q_proj[r] = np.dot(q, vk_rot)
            per_slot_data.append((slot_id, score_anc, q_proj, slen, scale_u, block_scale))

            terms.append(("anc", k, score_anc * scale))

            for t in range(slen):
                # Only the first `rank` of U_pool's real pool_rank-wide columns
                # (this is exactly the addressing distinction Bug 2 got wrong).
                u_row = U_pool[slot_id, t, :rank].astype(np.float64)
                delta_sum = float(np.dot(q_proj, u_row))
                t_score = (delta_sum * scale_u * block_scale + score_anc) * scale

                final_t_score = t_score
                for fi in range(3):
                    fpos = int(fact_pos[slot_id, fi]) if fact_pos.size else -1
                    if fpos == t:
                        _c, _s = _ovr_rope(k, fpos)
                        fk_rot = rope_rotate_dim_ref(
                            fact_val_K[slot_id, fi, kv_head].astype(np.float64),
                            _c if has_rope else None, _s if has_rope else None,
                            rotary_dim, has_rope)
                        final_t_score = float(np.dot(q, fk_rot)) * scale
                        break
                for r in range(max_residual):
                    rpos = int(res_pos_K[slot_id, r]) if res_pos_K.size else -1
                    if rpos == t:
                        # Exact-keys mode rotates at the block ANCHOR (the stored
                        # delta already carries the within-block offset), and
                        # SUBSTITUTES rather than corrects.
                        if EXACT_KEYS:
                            _c = cos_anc[k] if has_rope else None
                            _s = sin_anc[k] if has_rope else None
                        else:
                            _c, _s = _ovr_rope(k, rpos)
                            _c = _c if has_rope else None
                            _s = _s if has_rope else None
                        rk = res_val_K[slot_id, r, kv_head].astype(np.float64)
                        if EXACT_KEYS:
                            rk = rk + anchors_K[slot_id, kv_head].astype(np.float64)
                        rk_rot = rope_rotate_dim_ref(rk, _c, _s, rotary_dim, has_rope)
                        if EXACT_KEYS:
                            final_t_score = float(np.dot(q, rk_rot)) * scale
                        else:
                            final_t_score += float(np.dot(q, rk_rot)) * scale
                        break

                terms.append(("tok", k, t, final_t_score))

        for t in range(L_dense):
            k_rot = rope_rotate_dim_ref(
                dense_K[kv_head, t].astype(np.float64),
                cos_dense[t] if has_dense_rope else None, sin_dense[t] if has_dense_rope else None,
                rotary_dim, has_dense_rope)
            score = float(np.dot(q, k_rot)) * scale
            terms.append(("dense", t, score))

        # Softmax over ALL terms -- equivalent to the kernel's online merge.
        raw_scores = np.array([term[-1] for term in terms], dtype=np.float64)
        m = raw_scores.max()
        w = np.exp(raw_scores - m)
        weights = w / w.sum()
        w_idx = {term[:-1]: weights[i] for i, term in enumerate(terms)}

        # Value accumulation, mirroring Pass 2 exactly.
        acc = np.zeros(D, dtype=np.float64)
        for k, (slot_id, score_anc, q_proj, slen, scale_u, block_scale) in enumerate(per_slot_data):
            w_anc = w_idx[("anc", k)]
            sum_delta_w = sum(w_idx[("tok", k, t)] for t in range(slen))
            w_total_anc = w_anc + sum_delta_w
            acc += w_total_anc * anchors_V[slot_id, kv_head].astype(np.float64)

            w_proj = np.zeros(rank, dtype=np.float64)
            for t in range(slen):
                w_t = w_idx[("tok", k, t)]
                w_proj += w_t * U_pool[slot_id, t, :rank].astype(np.float64) * scale_u
            svd_v = np.zeros(D, dtype=np.float64)
            for r in range(rank):
                svd_v += w_proj[r] * VV_pool[slot_id, r, kv_head].astype(np.float64)
            acc += svd_v * block_scale

            for r in range(max_residual):
                rpos = int(res_pos_V[slot_id, r]) if res_pos_V.size else -1
                if 0 <= rpos < slen:
                    if EXACT_KEYS and fact_pos.size and rpos in set(
                            int(x) for x in fact_pos[slot_id]):
                        continue  # substituted by the fact loop below
                    w_res = w_idx[("tok", k, rpos)]
                    res_v = res_val_V[slot_id, r, kv_head].astype(np.float64)
                    if EXACT_KEYS:
                        # Back out the token's low-rank value estimate, so the
                        # exact value replaces it instead of stacking on top.
                        # anchors_V cancels between exact and estimate.
                        u_row = U_pool[slot_id, rpos, :rank].astype(np.float64)
                        svd_v_r = (u_row * scale_u) @ VV_pool[slot_id, :rank, kv_head].astype(np.float64)
                        res_v = res_v - svd_v_r * block_scale
                    acc += w_res * res_v

            for fi in range(3):
                fpos = int(fact_pos[slot_id, fi]) if fact_pos.size else -1
                if 0 <= fpos < slen:
                    w_fact = w_idx[("tok", k, fpos)]
                    u_row = U_pool[slot_id, fpos, :rank].astype(np.float64)
                    svd_v_fpos = (u_row * scale_u) @ VV_pool[slot_id, :rank, kv_head].astype(np.float64)
                    svd_v_fpos = svd_v_fpos * block_scale + anchors_V[slot_id, kv_head].astype(np.float64)
                    exact_fact_v = fact_val_V[slot_id, fi, kv_head].astype(np.float64)
                    acc += w_fact * (exact_fact_v - svd_v_fpos)

        for t in range(L_dense):
            acc += w_idx[("dense", t)] * dense_V[kv_head, t].astype(np.float64)

        out[h] = acc

    return out


_FP16_FIELDS = {
    "Q", "U_scale_pool", "VK_pool", "VV_pool", "anchors_K", "anchors_V", "scales",
    "res_val_K", "res_val_V", "fact_val_K", "fact_val_V", "dense_K", "dense_V",
}


def run_case(name, **kw):
    print(f"\n=== {name} ===", flush=True)

    # Round every fp16-typed field through actual fp16 BEFORE computing the
    # reference, so both sides start from bit-identical inputs -- otherwise a
    # "discrepancy" could just be fp16 input quantization noise (the kernel
    # never sees full-precision values in the first place).
    kw = dict(kw)
    for field in _FP16_FIELDS:
        if kw[field].size:
            kw[field] = kw[field].astype(np.float16).astype(np.float32)

    ref = reference_decode_attention(**kw)

    def t(x, dtype):
        return torch.tensor(x, dtype=dtype, device=DEVICE)

    def opt(x, dtype):
        return t(x, dtype) if x.size else torch.empty(0, device=DEVICE, dtype=dtype)

    out, _lse = dkv_core.decode_attention_metal(
        Q=t(kw["Q"], torch.float16), U_pool=t(kw["U_pool"], torch.int8),
        U_scale_pool=t(kw["U_scale_pool"], torch.float16), VK_pool=t(kw["VK_pool"], torch.float16),
        VV_pool=t(kw["VV_pool"], torch.float16), anchors_K=t(kw["anchors_K"], torch.float16),
        anchors_V=t(kw["anchors_V"], torch.float16), seq_lens=t(kw["seq_lens"], torch.int32),
        scales=t(kw["scales"], torch.float16), cos_anc=opt(kw["cos_anc"], torch.float32),
        sin_anc=opt(kw["sin_anc"], torch.float32), slot_indices=t(kw["slot_indices"], torch.int32),
        scale=kw["scale"], n_q_heads=kw["n_q_heads"], n_kv_heads=kw["n_kv_heads"], rank=kw["rank"],
        res_pos_K=opt(kw["res_pos_K"], torch.int16), res_val_K=opt(kw["res_val_K"], torch.float16),
        res_pos_V=opt(kw["res_pos_V"], torch.int16), res_val_V=opt(kw["res_val_V"], torch.float16),
        fact_pos=opt(kw["fact_pos"], torch.int16), fact_val_K=opt(kw["fact_val_K"], torch.float16),
        fact_val_V=opt(kw["fact_val_V"], torch.float16), dense_K=opt(kw["dense_K"], torch.float16),
        dense_V=opt(kw["dense_V"], torch.float16), cos_dense=opt(kw["cos_dense"], torch.float32),
        sin_dense=opt(kw["sin_dense"], torch.float32), rotary_dim=kw["rotary_dim"],
        # None (not an empty tensor) when absent -- the kernel's optional args
        # take c10::nullopt, and passing None is what exercises the fallback.
        cos_full=t(kw["cos_full"], torch.float32) if kw["cos_full"].size else None,
        sin_full=t(kw["sin_full"], torch.float32) if kw["sin_full"].size else None,
        anchor_pos=t(kw["anchor_pos"], torch.int32) if kw["anchor_pos"].size else None,
    )
    kernel_out = out.cpu().to(torch.float64).numpy()

    abs_diff = np.abs(kernel_out - ref)
    max_abs = abs_diff.max()
    max_rel = (abs_diff / (np.abs(ref) + 1e-4)).max()
    has_nan = np.isnan(kernel_out).any()
    print(f"kernel has_nan={has_nan}  max_abs_diff={max_abs:.5f}  max_rel_diff={max_rel:.5f}")

    # Combined tolerance (like np.allclose): atol + rtol*|ref|.
    within_tol = abs_diff <= (0.02 + 0.01 * np.abs(ref))
    ok = (not has_nan) and bool(within_tol.all())
    print("PASS" if ok else "FAIL")
    if not ok:
        print("ref[:2,:8]   =", ref[:2, :8])
        print("kernel[:2,:8]=", kernel_out[:2, :8])
    return ok


def make_config(
    D, rotary_dim, n_kv_heads, n_q_heads, rank, pool_rank, N_pool, S_max,
    K_active, max_residual, L_dense, use_res=True, use_fact=True, use_dense_rope=True,
    seed=0, exact_res_rope=False, max_pos=4096, dense_pad=0,
    fact_overlaps_res=False,
):
    """Builds a random, self-consistent set of decode_attention_metal inputs."""
    r = np.random.default_rng(seed)

    def randh(*shape):
        return (r.standard_normal(shape) * 0.3).astype(np.float32)

    U_pool = r.integers(-100, 100, size=(N_pool, S_max, pool_rank)).astype(np.int8)
    seq_lens = r.integers(1, S_max + 1, size=N_pool).astype(np.int32)
    slot_indices = (
        r.choice(N_pool, size=K_active, replace=False).astype(np.int32)
        if K_active > 0 else np.zeros(0, dtype=np.int32)
    )

    def make_rope_table(n_rows):
        pos = r.uniform(0, 6.28, size=n_rows)
        half = rotary_dim // 2
        freqs = 1.0 / (10000 ** (np.arange(half) / max(half, 1)))
        angles = pos[:, None] * freqs[None, :]
        angles_full = np.concatenate([angles, angles], axis=-1)
        cos_r, sin_r = np.cos(angles_full), np.sin(angles_full)
        pad = D - rotary_dim
        if pad > 0:
            cos_r = np.concatenate([cos_r, np.ones((n_rows, pad))], axis=-1)
            sin_r = np.concatenate([sin_r, np.zeros((n_rows, pad))], axis=-1)
        return cos_r.astype(np.float32), sin_r.astype(np.float32)

    if K_active > 0:
        cos_anc, sin_anc = make_rope_table(K_active)
    else:
        cos_anc = np.zeros((0, D), dtype=np.float32)
        sin_anc = np.zeros((0, D), dtype=np.float32)

    if max_residual > 0 and use_res:
        res_pos_K = np.full((N_pool, max_residual), -1, dtype=np.int16)
        res_pos_V = np.full((N_pool, max_residual), -1, dtype=np.int16)
        for n in range(N_pool):
            n_res = min(max_residual, int(seq_lens[n]))
            chosen = r.choice(int(seq_lens[n]), size=n_res, replace=False)
            res_pos_K[n, :n_res] = chosen
            res_pos_V[n, :n_res] = chosen
        res_val_K = randh(N_pool, max_residual, n_kv_heads, D)
        res_val_V = randh(N_pool, max_residual, n_kv_heads, D)
    else:
        res_pos_K = np.zeros((0, 0), dtype=np.int16)
        res_pos_V = np.zeros((0, 0), dtype=np.int16)
        res_val_K = np.zeros((0, 0, 0, 0), dtype=np.float32)
        res_val_V = np.zeros((0, 0, 0, 0), dtype=np.float32)

    if use_fact:
        fact_pos = np.full((N_pool, 3), -1, dtype=np.int16)
        for n in range(N_pool):
            if fact_overlaps_res and res_pos_K.size:
                # Deliberately place facts ON residual positions. Both mechanisms
                # substitute a token's value, so under exact-keys mode the two
                # must not each back out the low-rank estimate.
                candidates = [int(p) for p in res_pos_K[n] if 0 <= int(p) < int(seq_lens[n])]
            else:
                avoid = set(res_pos_K[n].tolist()) if res_pos_K.size else set()
                candidates = [p for p in range(int(seq_lens[n])) if p not in avoid]
            n_fact = min(3, len(candidates))
            chosen = r.choice(candidates, size=n_fact, replace=False) if candidates else []
            fact_pos[n, :len(chosen)] = chosen
        fact_val_K = randh(N_pool, 3, n_kv_heads, D)
        fact_val_V = randh(N_pool, 3, n_kv_heads, D)
    else:
        fact_pos = np.zeros((0, 0), dtype=np.int16)
        fact_val_K = np.zeros((0, 0, 0, 0), dtype=np.float32)
        fact_val_V = np.zeros((0, 0, 0, 0), dtype=np.float32)

    if L_dense > 0:
        dense_K = randh(n_kv_heads, L_dense + dense_pad, D)
        dense_V = randh(n_kv_heads, L_dense + dense_pad, D)
        if dense_pad > 0:
            # Padding rows carry LARGE values so that attending them at all
            # produces an obviously wrong result rather than a subtle drift.
            dense_K[:, L_dense:, :] = 25.0
            dense_V[:, L_dense:, :] = 25.0
        if use_dense_rope:
            cos_dense, sin_dense = make_rope_table(L_dense)
        else:
            cos_dense = np.zeros((0, D), dtype=np.float32)
            sin_dense = np.zeros((0, D), dtype=np.float32)
    else:
        dense_K = np.zeros((0, 0, 0), dtype=np.float32)
        dense_V = np.zeros((0, 0, 0), dtype=np.float32)
        cos_dense = np.zeros((0, D), dtype=np.float32)
        sin_dense = np.zeros((0, D), dtype=np.float32)

    # Full-sequence RoPE tables (raw: [max_pos, rotary_dim], NOT padded to D)
    # + absolute anchor position per routed slot, for exact-position rotation
    # of residual/fact overrides.
    if exact_res_rope and K_active > 0:
        _pos = np.arange(max_pos)
        _half = rotary_dim // 2
        _freqs = 1.0 / (10000 ** (np.arange(_half) / max(_half, 1)))
        _ang = _pos[:, None] * _freqs[None, :]
        _ang = np.concatenate([_ang, _ang], axis=-1)         # [max_pos, rotary_dim]
        cos_full = np.cos(_ang).astype(np.float32)
        sin_full = np.sin(_ang).astype(np.float32)
        # Anchors spread through the sequence, leaving room for within-block offsets.
        anchor_pos = (r.integers(0, max(1, max_pos - S_max - 1), size=K_active)).astype(np.int32)
    else:
        cos_full = np.zeros((0, 0), dtype=np.float32)
        sin_full = np.zeros((0, 0), dtype=np.float32)
        anchor_pos = np.zeros(0, dtype=np.int32)

    return dict(
        cos_full=cos_full, sin_full=sin_full, anchor_pos=anchor_pos,
        Q=randh(n_q_heads, D), U_pool=U_pool,
        U_scale_pool=r.uniform(0.5, 1.5, size=N_pool).astype(np.float32),
        VK_pool=randh(N_pool, pool_rank, n_kv_heads, D), VV_pool=randh(N_pool, pool_rank, n_kv_heads, D),
        anchors_K=randh(N_pool, n_kv_heads, D), anchors_V=randh(N_pool, n_kv_heads, D),
        seq_lens=seq_lens, scales=r.uniform(0.3, 0.7, size=N_pool).astype(np.float32),
        cos_anc=cos_anc, sin_anc=sin_anc, slot_indices=slot_indices,
        scale=1.0 / np.sqrt(D), n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, rank=rank,
        res_pos_K=res_pos_K, res_val_K=res_val_K, res_pos_V=res_pos_V, res_val_V=res_val_V,
        fact_pos=fact_pos, fact_val_K=fact_val_K, fact_val_V=fact_val_V,
        dense_K=dense_K, dense_V=dense_V, cos_dense=cos_dense, sin_dense=sin_dense,
        rotary_dim=rotary_dim,
    )


CASES = [
    ("1_simplest", "1. simplest (1 slot, full rotary, no extras)", dict(
        D=8, rotary_dim=8, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=2,
        N_pool=1, S_max=4, K_active=1, max_residual=0, L_dense=0,
        use_res=False, use_fact=False, use_dense_rope=False, seed=1)),
    ("2_partial_rope", "2. partial RoPE (rotary_dim=4 < D=8)", dict(
        D=8, rotary_dim=4, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=2,
        N_pool=1, S_max=4, K_active=1, max_residual=0, L_dense=0,
        use_res=False, use_fact=False, use_dense_rope=False, seed=2)),
    ("3_rank_lt_pool_rank", "3. rank(2) < pool_rank(5), 4 slots -- addressing-stride regression check", dict(
        D=8, rotary_dim=8, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=5,
        N_pool=6, S_max=4, K_active=4, max_residual=0, L_dense=0,
        use_res=False, use_fact=False, use_dense_rope=False, seed=3)),
    ("4_gqa", "4. GQA (n_q_heads=6, n_kv_heads=2)", dict(
        D=8, rotary_dim=8, n_kv_heads=2, n_q_heads=6, rank=2, pool_rank=2,
        N_pool=3, S_max=4, K_active=2, max_residual=0, L_dense=0,
        use_res=False, use_fact=False, use_dense_rope=False, seed=4)),
    ("5_residual", "5. residual K/V corrections", dict(
        D=8, rotary_dim=8, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=2,
        N_pool=2, S_max=6, K_active=2, max_residual=3, L_dense=0,
        use_res=True, use_fact=False, use_dense_rope=False, seed=5)),
    ("6_fact", "6. fact anchor overrides", dict(
        D=8, rotary_dim=8, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=2,
        N_pool=2, S_max=6, K_active=2, max_residual=0, L_dense=0,
        use_res=False, use_fact=True, use_dense_rope=False, seed=6)),
    ("7_dense", "7. dense window + partial RoPE", dict(
        D=8, rotary_dim=4, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=2,
        N_pool=2, S_max=4, K_active=2, max_residual=0, L_dense=5,
        use_res=False, use_fact=False, use_dense_rope=True, seed=7)),
    ("8_zero_slots_dense_rope", "8. K_active=0, dense window WITH rope -- has_dense_rope regression check", dict(
        D=8, rotary_dim=4, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=2,
        N_pool=2, S_max=4, K_active=0, max_residual=0, L_dense=6,
        use_res=False, use_fact=False, use_dense_rope=True, seed=8)),
    ("9_no_fact_tensors", "9. use_fact=False (empty fact buffers) -- has_fact regression check", dict(
        D=8, rotary_dim=8, n_kv_heads=1, n_q_heads=1, rank=1, pool_rank=1,
        N_pool=1, S_max=1, K_active=1, max_residual=0, L_dense=0,
        use_res=False, use_fact=False, use_dense_rope=False, seed=11)),
    ("10_combined", "10. everything combined, rank<pool_rank, GQA", dict(
        D=16, rotary_dim=8, n_kv_heads=2, n_q_heads=4, rank=3, pool_rank=6,
        N_pool=8, S_max=10, K_active=5, max_residual=2, L_dense=7,
        use_res=True, use_fact=True, use_dense_rope=True, seed=9)),
    ("11_realistic_scale", "11. realistic scale (D=256, rotary_dim=64, rank=32, pool_rank=48)", dict(
        D=256, rotary_dim=64, n_kv_heads=2, n_q_heads=8, rank=32, pool_rank=48,
        N_pool=40, S_max=257, K_active=20, max_residual=16, L_dense=100,
        use_res=True, use_fact=True, use_dense_rope=True, seed=10)),
    # ── Exact-position residual/fact RoPE (has_exact_res_rope) ────────────────
    # These pass the full-sequence rope tables + anchor positions, so residual-K
    # and fact-K must be rotated at anchor+offset rather than at the anchor.
    # Anchor-position rotation (the old behavior) fails these by a wide margin.
    ("12_padded_dense_workspace", "12. PADDED dense workspace -- L_dense_valid regression check", dict(
        D=8, rotary_dim=8, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=2,
        N_pool=2, S_max=4, K_active=2, max_residual=0, L_dense=6,
        use_res=False, use_fact=False, use_dense_rope=True, seed=15, dense_pad=10)),
    ("13_exact_res_rope", "12. exact-position residual/fact RoPE -- partial rotary", dict(
        D=8, rotary_dim=4, n_kv_heads=1, n_q_heads=1, rank=2, pool_rank=2,
        N_pool=3, S_max=8, K_active=3, max_residual=4, L_dense=0,
        use_res=True, use_fact=True, use_dense_rope=False, seed=12,
        exact_res_rope=True, max_pos=512)),
    ("13_exact_res_rope_full_rotary", "13. exact-position residual/fact RoPE -- full rotary + GQA", dict(
        D=16, rotary_dim=16, n_kv_heads=2, n_q_heads=4, rank=3, pool_rank=5,
        N_pool=6, S_max=10, K_active=4, max_residual=5, L_dense=6,
        use_res=True, use_fact=True, use_dense_rope=True, seed=13,
        exact_res_rope=True, max_pos=1024)),
    ("14_exact_res_rope_realistic", "14. exact-position residual/fact RoPE -- realistic Qwen3.5 scale", dict(
        D=256, rotary_dim=64, n_kv_heads=2, n_q_heads=8, rank=32, pool_rank=48,
        N_pool=40, S_max=257, K_active=20, max_residual=16, L_dense=100,
        use_res=True, use_fact=True, use_dense_rope=True, seed=14,
        exact_res_rope=True, max_pos=8192)),
    # REGRESSION: dense window LONGER than the kernel's dense_w_shared width (768).
    # This used to silently drop every dense token past row 767 -- not
    # down-weighted, never scored -- so a needle in the newest part of the recency
    # window was invisible to the model. max_dense_len is recency_window +
    # block_size (1419 at the mid preset), so production routinely exceeded it.
    ("15_dense_exceeds_shared", "15. dense window longer than dense_w_shared (>768)", dict(
        D=64, rotary_dim=64, n_kv_heads=2, n_q_heads=4, rank=8, pool_rank=16,
        N_pool=6, S_max=32, K_active=3, max_residual=4, L_dense=993,
        use_res=True, use_fact=True, use_dense_rope=True, seed=15, max_pos=8192)),
    ("16_dense_exceeds_shared_no_slots", "16. long dense window, zero compressed slots", dict(
        D=32, rotary_dim=16, n_kv_heads=1, n_q_heads=4, rank=4, pool_rank=8,
        N_pool=4, S_max=16, K_active=0, max_residual=0, L_dense=1419,
        use_res=False, use_fact=False, use_dense_rope=True, seed=16, max_pos=8192)),
    # REGRESSION: the ACTUAL production configuration after the MLX alignment --
    # max_residual=128 (MLX's default), block seq_len 256, recency window 4096.
    # Every bug in this investigation has been a constant sized for the OLD
    # defaults that nothing validated against the new ones (the residual scratch
    # was 64 wide while reads ran to max_residual; the pool divided its budget by
    # a cost excluding residuals; the dense workspace was 2.24x over-allocated).
    # This case pins the shipped config so the next such drift fails here rather
    # than as unexplained garbage output.
    ("17_production_config", "17. production config: max_residual=128, S_max=257, L_dense=4065", dict(
        D=256, rotary_dim=64, n_kv_heads=2, n_q_heads=8, rank=32, pool_rank=48,
        N_pool=20, S_max=257, K_active=16, max_residual=128, L_dense=4065,
        use_res=True, use_fact=True, use_dense_rope=True, seed=17, max_pos=8448)),
]

# Cases that only make sense under DKV_RESIDUAL_EXACT_KEYS. They deliberately
# force residual K and V onto the SAME positions (as the compressor now does),
# because substitution is only well-defined when a token's score and its value
# are replaced together.
EXACT_KEY_CASES = [
    ("E1_exact_keys_basic", "E1. exact-keys substitution -- small", dict(
        D=16, rotary_dim=16, n_kv_heads=1, n_q_heads=1, rank=3, pool_rank=3,
        N_pool=3, S_max=8, K_active=2, max_residual=4, L_dense=0,
        use_res=True, use_fact=False, use_dense_rope=False, seed=101,
        max_pos=1024)),
    ("E2_exact_keys_gqa_partial", "E2. exact-keys substitution -- GQA + partial rotary", dict(
        D=32, rotary_dim=8, n_kv_heads=2, n_q_heads=8, rank=4, pool_rank=6,
        N_pool=5, S_max=12, K_active=4, max_residual=5, L_dense=7,
        use_res=True, use_fact=False, use_dense_rope=True, seed=102,
        max_pos=2048)),
    ("E3_exact_keys_with_facts", "E3. exact-keys substitution -- overlapping fact anchors", dict(
        D=32, rotary_dim=32, n_kv_heads=2, n_q_heads=4, rank=4, pool_rank=8,
        N_pool=5, S_max=12, K_active=4, max_residual=5, L_dense=6,
        use_res=True, use_fact=True, use_dense_rope=True, seed=103,
        max_pos=2048, fact_overlaps_res=True)),
    ("E4_exact_keys_realistic", "E4. exact-keys substitution -- realistic Qwen3.5 scale", dict(
        D=256, rotary_dim=64, n_kv_heads=2, n_q_heads=8, rank=32, pool_rank=48,
        N_pool=40, S_max=257, K_active=20, max_residual=16, L_dense=100,
        use_res=True, use_fact=True, use_dense_rope=True, seed=104,
        max_pos=8192, fact_overlaps_res=True)),
]

# Every case runs under BOTH residual conventions. make_config already gives the
# K and V residuals the same positions, which is what exact-keys substitution
# requires, so the general cases are valid in either mode. The E-cases add the
# fact/residual overlap that only matters when substituting.
ACTIVE_CASES = CASES + EXACT_KEY_CASES


@pytest.mark.parametrize("key,label,cfg_kwargs", ACTIVE_CASES, ids=[c[0] for c in ACTIVE_CASES])
def test_decode_attention_metal_case(key, label, cfg_kwargs):
    cfg = make_config(**cfg_kwargs)
    assert run_case(label, **cfg)


if __name__ == "__main__":
    results = {}
    for key, label, cfg_kwargs in ACTIVE_CASES:
        cfg = make_config(**cfg_kwargs)
        results[key] = run_case(label, **cfg)

    print("\n" + "=" * 60)
    for key, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {key}")
    n_pass = sum(results.values())
    print(f"\n{n_pass}/{len(results)} passed (DKV_RESIDUAL_EXACT_KEYS="
          f"{'1' if EXACT_KEYS else '0'})")
    ok_all = n_pass == len(results)

    # The flag is cached in a C++ static on first kernel launch, so the other
    # convention needs a fresh process. The default run covers the runtime
    # default (exact keys); re-exec once with =0 to keep the legacy correction
    # form covered too.
    if EXACT_KEYS and os.environ.get("DKV_RESIDUAL_EXACT_KEYS") is None:
        import subprocess
        print("\n" + "=" * 60)
        print("Re-running the same cases in correction form (DKV_RESIDUAL_EXACT_KEYS=0)...")
        env = dict(os.environ, DKV_RESIDUAL_EXACT_KEYS="0")
        rc = subprocess.run([sys.executable, __file__], env=env).returncode
        ok_all = ok_all and rc == 0

    if not ok_all:
        sys.exit(1)
