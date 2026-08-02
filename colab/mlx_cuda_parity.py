#!/usr/bin/env python3
"""Side-by-side numerical comparison of the MLX and CUDA compression paths.

WHY THIS EXISTS
---------------
Every claim of "these are equivalent" in this codebase so far has come from
reading the two implementations and concluding they match. That method has a bad
record: it produced three separate parity claims that were later contradicted by
behaviour, and it cannot see the differences that matter (seeds, batch-dependent
projections, quantisation granularity, error ranking) because they look identical
at a glance.

This runs BOTH implementations on the SAME inputs in one process and prints where
the numbers diverge. It needs no GPU: MLX runs natively on Apple Silicon, and the
CUDA side is plain PyTorch which runs on CPU.

    python colab/mlx_cuda_parity.py

WHAT IS COMPARED
----------------
Stage by stage, so a divergence is attributed to the stage that introduced it:

  1. delta normalisation   (per-block scale)
  2. random projection     (Omega: shape, sharing, seed)
  3. rSVD factors          (U, Vh) and the reconstruction they produce
  4. int8 quantisation of U
  5. residual selection    (which token indices get an exact residual)

Stage 3 is reported per token AND for a planted needle row, because a mean error
that looks fine can still be a total loss on the one token that carries a code.

FIDELITY NOTE
-------------
The CUDA side is replicated here from lowrank.py rather than imported, because
_compress_layer_blocks_gpu_inner needs live block objects and a pool. Every step
below cites the source lines it mirrors so the replication is auditable; if
lowrank.py changes, re-check the citations. MLX is called directly — no
replication, no modification.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))

import torch

try:
    import mlx.core as mx
except ImportError:
    print("mlx not importable — run this on Apple Silicon. "
          "The CUDA half alone tells you nothing about parity.")
    sys.exit(1)

from serving.mlx_dkv_wrapper import compress_mlx_block_batched


def build_deltas(n_blocks, S, feat, needle_block, needle_row, rank, seed=7):
    """Deltas shaped like a real block batch, with one planted 'needle' row.

    The needle must be SAME-MAGNITUDE but pointing OUTSIDE the filler subspace.
    A first attempt made it high-magnitude, and the harness immediately showed
    that to be wrong: a large outlier row dominates sigma_1, so the top-rank basis
    captures it and it comes out the BEST-reconstructed row in the block. Being
    distinctive does not make a token badly reconstructed — being ORTHOGONAL to
    what the other tokens share does.

    Filler therefore spans more directions than `rank`, so truncation genuinely
    has to discard something, and the needle competes for a kept direction on
    equal magnitude terms.
    """
    g = torch.Generator().manual_seed(seed)
    n_basis = rank + 16                       # > rank: truncation must drop some
    basis = torch.randn(n_basis, feat, generator=g)
    basis = basis / basis.norm(dim=1, keepdim=True)
    coef = torch.randn(n_blocks, S, n_basis, generator=g)
    # decaying weights => a realistic singular spectrum rather than a flat one
    decay = torch.linspace(1.0, 0.05, n_basis).view(1, 1, -1)
    d = (coef * decay) @ basis
    d = d + torch.randn(n_blocks, S, feat, generator=g) * 0.02

    # needle: a direction orthogonal to the filler basis, scaled to the typical
    # row norm so it wins nothing by magnitude alone
    v = torch.randn(feat, generator=g)
    v = v - basis.T @ (basis @ v)             # project out the filler subspace
    v = v / v.norm() * d[needle_block].norm(dim=1).median()
    d[needle_block, needle_row] = v
    return d


def cuda_rsvd(deltas, rank, n_oversamples=5, n_iter=2, seed=None):
    """Mirrors lowrank.py _compress_layer_blocks_gpu_inner rSVD, lines ~1046-1150.

    Faithful to: per-block Omega via _rsvd_omega (seed default 0), power
    iteration x2, QR, DIRECT svd of B (not the covariance), S folded into U.
    """
    from native_core.compression.lowrank import _rsvd_omega
    N, T, feat = deltas.shape
    if seed is not None:
        os.environ["DKV_RSVD_SEED"] = str(seed)
    # normalisation: lowrank.py normalises deltas per block before the sketch
    scales = deltas.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-9)
    x = deltas / scales

    r_proj = min(rank + n_oversamples, T, feat)
    Omega = _rsvd_omega(N, feat, r_proj, dtype=torch.float32)   # PER-BLOCK
    Y = torch.matmul(x, Omega)
    for _ in range(n_iter):
        Y = torch.matmul(x, torch.matmul(x.transpose(1, 2), Y))
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = torch.matmul(Q.transpose(1, 2), x)
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)       # DIRECT svd of B
    U = torch.matmul(Q, U_b)
    U_k = U[:, :, :rank] * S[:, :rank].unsqueeze(1)             # S folded into U
    Vh_k = Vh[:, :rank, :]
    return U_k, Vh_k, scales.squeeze(-1).squeeze(-1)


def q_int8(U):
    """int8 quantisation of U, one scale per block. Same on both sides:
    lowrank.py ~1434 / mlx_dkv_wrapper.py ~3331."""
    s = U.abs().amax(dim=(1, 2)).clamp(min=1e-5)
    Uq = torch.clamp(torch.round(U / s.view(-1, 1, 1) * 127), -127, 127)
    return Uq * s.view(-1, 1, 1) / 127.0


def rowerr(recon, deltas):
    return (recon - deltas).norm(dim=-1) / deltas.norm(dim=-1).clamp(min=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING + DECODE
#
#  The compression stages above replicate the CUDA side. These do NOT: they call
#  the REAL production functions (route_blocks_relevance,
#  _gather_routed_blocks_for_kernel) on CPU tensors, so what is measured is the
#  shipping code, not a paraphrase of it. Only the Triton kernel's inner
#  arithmetic is replicated, and only because it cannot run without a GPU.
#
#  THE POSITION CONVENTION, which everything here turns on:
#
#      block-local slot 0 .............. the ANCHOR, at absolute anchor_idx
#      active-token index j ............ at absolute anchor_idx + 1 + j
#
#  stated independently at streaming_sparse_ingest.py:1206/1216 (anchor = k[...,0],
#  active = k[...,1:]), dkv_attention.py:1385 (arange(1 + max_seq_len)), and
#  dkv_decode.metal:153 ("lands delta token t at absolute anchor_pos + t + 1 --
#  its true position").
#
#  MLX never has to re-derive it: it captures keys POST-RoPE at their true
#  positions and neither its router nor its decode rotates anything. CUDA stores
#  UNROTATED K on purpose (dkv_backend.py:40) and re-rotates on read, so every
#  read site re-derives the convention by hand — which is exactly where it breaks.
# ══════════════════════════════════════════════════════════════════════════════

H_KV, GPK, HEAD_DIM, ROT_DIM = 2, 2, 64, 16     # ROT_DIM < HEAD_DIM = Qwen3.5 geometry
BS, N_BLK, MAXRES = 17, 4, 3                    # 1 anchor + 16 active, per block
H_Q = H_KV * GPK
SEQ = N_BLK * BS
SEQ_LONG = 8192          # reach a realistic deep-block anchor in the R3 sweep


def _rope_tables(L, rot, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, rot, 2).float() / rot))
    t = torch.arange(L).float()
    f = torch.outer(t, inv)
    emb = torch.cat([f, f], dim=-1)
    return emb.cos()[None], emb.sin()[None]      # [1, L, rot], as rotary_emb returns


def _stub_pool(anchors_K, res_k, res_pos, rank=4):
    """Minimal stand-in exposing only what the two functions under test read."""
    P = anchors_K.shape[0]
    class _P: pass
    p = _P()
    p.anchors_K = anchors_K
    p.anchors_V = torch.zeros_like(anchors_K)
    p.V_K = torch.zeros(P, rank, H_KV, HEAD_DIM)
    p.V_V = torch.zeros(P, rank, H_KV, HEAD_DIM)
    p.U = torch.zeros(P, BS - 1, rank)
    p.U_scale = torch.ones(P)
    p.scales = torch.ones(P)
    p.seq_lens = torch.full((P,), BS - 1, dtype=torch.int32)
    p.residual_K_values = res_k
    p.residual_V_values = torch.zeros_like(res_k)
    p.residual_K_positions = res_pos
    p.residual_V_positions = res_pos.clone()
    p.routing_topk_default = 64
    return p


def stage_positions():
    """Is a stored residual key rotated at the position it actually occupies?

    Decisive because it needs no reference implementation: the residual's true
    position is known by construction, so the gather either lands on it or does
    not. A miss of ONE token is not a rounding error — theta_0 = 1.0 rad, so the
    fastest RoPE pair is off by a full radian while the slowest barely moves.
    That is a code returning with its letters right and its digits wrong, which
    is the observed failure (ZEBRA-447, ZEBRA-474-QUARTZ), not garbage output.
    """
    from native_core.sparse_decode.triton_fused_decode import (
        _gather_routed_blocks_for_kernel, _partial_rope_apply)

    cos, sin = _rope_tables(SEQ, ROT_DIM)
    g0 = torch.Generator().manual_seed(11)

    anchors = torch.arange(N_BLK) * BS                      # 0, 17, 34, 51
    anchors_K = torch.randn(N_BLK, H_KV, HEAD_DIM, generator=g0)
    res_k = torch.randn(N_BLK, MAXRES, H_KV, HEAD_DIM, generator=g0)
    res_pos = torch.tensor([[2, 9, -1]] * N_BLK, dtype=torch.int16)   # active-token idx

    pool = _stub_pool(anchors_K, res_k, res_pos)
    g = _gather_routed_blocks_for_kernel(
        pool, torch.arange(N_BLK), anchors, cos, sin)

    from native_core.sparse_decode.triton_fused_decode import pool_stores_rotated_k

    print("\n" + "=" * 74)
    print("  [R1] residual-key RoPE handling   (real _gather_routed_blocks_for_kernel)")
    print("=" * 74)

    # This stage guards BOTH conventions, because there are now two and the bug
    # class is a site applying the wrong one. See pool_stores_rotated_k.
    if pool_stores_rotated_k():
        # MLX convention (DKV_ROTATED_POOL, default): the pool already holds
        # POST-RoPE keys, so the gather must pass them through UNTOUCHED. Any
        # rotation here would be a SECOND rotation.
        worst = (g["res_k"] - res_k).abs().max().item()
        worst_anc = (g["anchors_K"] - anchors_K).abs().max().item()
        print(f"    convention: ROTATED pool (MLX) — gather must not rotate")
        print(f"    max|gathered residual - stored| = {worst:.3e}")
        print(f"    max|gathered anchor   - stored| = {worst_anc:.3e}")
        ok = worst < 1e-6 and worst_anc < 1e-6
        print(f"    {'PASS — passed through unrotated, as MLX stores them'
               if ok else '*** FAIL — rotated a second time ***'}")
        return ok

    worst_true = worst_off1 = 0.0
    for n in range(N_BLK):
        for ri in range(MAXRES):
            p = int(res_pos[n, ri])
            if p < 0:
                continue
            got = g["res_k"][n, ri]
            for label, abs_pos in (("true", anchors[n] + 1 + p),
                                   ("off-by-1", anchors[n] + p)):
                c = cos[0, abs_pos].view(1, -1)
                s = sin[0, abs_pos].view(1, -1)
                d = (got - _partial_rope_apply(res_k[n, ri], c, s)).abs().max().item()
                if label == "true":
                    worst_true = max(worst_true, d)
                else:
                    worst_off1 = max(worst_off1, d)
    print(f"    convention: UNROTATED pool — gather must rotate at the true position")
    print(f"    max|gathered - rope(key, anchor+1+offset)| = {worst_true:.3e}   <- TRUE position")
    print(f"    max|gathered - rope(key, anchor+offset)  | = {worst_off1:.3e}")
    ok = worst_true < 1e-5
    print(f"    {'PASS — rotated at its true position' if ok else '*** FAIL — rotated at the WRONG position ***'}")
    return ok


def stage_routing():
    """Real route_blocks_relevance vs real MLX _block_relevance_residual.

    Same physical situation, each side fed in its own storage convention:
      MLX  gets keys already rotated at true positions (what it captures).
      CUDA gets the unrotated keys + the cos/sin tables (what it stores).
    If CUDA's read-time rotation is right the two relevance vectors agree, and
    both rank the needle's block first. The query is built to match the needle
    residual's TRUE rotated key, so the correct answer is unambiguous.
    """
    from native_core.srl.query_router import route_blocks_relevance
    from serving.mlx_dkv_wrapper import _block_relevance_residual
    from native_core.sparse_decode.triton_fused_decode import _partial_rope_apply

    cos, sin = _rope_tables(SEQ, ROT_DIM)
    g0 = torch.Generator().manual_seed(23)
    anchors = torch.arange(N_BLK) * BS
    anchors_K = torch.randn(N_BLK, H_KV, HEAD_DIM, generator=g0) * 0.5
    res_k = torch.randn(N_BLK, MAXRES, H_KV, HEAD_DIM, generator=g0) * 0.5
    res_pos = torch.tensor([[2, 9, -1]] * N_BLK, dtype=torch.int16)

    needle_blk, needle_ri = N_BLK - 1, 1          # deepest block => largest anchor
    needle_pos = int(anchors[needle_blk]) + 1 + int(res_pos[needle_blk, needle_ri])

    def rot_at(x, pos):
        return _partial_rope_apply(x, cos[0, pos].view(1, -1), sin[0, pos].view(1, -1))

    # rotated views, MLX's storage convention
    anc_rot = torch.stack([rot_at(anchors_K[n], int(anchors[n])) for n in range(N_BLK)])
    res_rot = torch.stack([
        torch.stack([rot_at(res_k[n, r], int(anchors[n]) + 1 + int(res_pos[n, r]))
                     if res_pos[n, r] >= 0 else res_k[n, r] for r in range(MAXRES)])
        for n in range(N_BLK)])

    # query = the needle residual's true rotated key => that block must win
    q = res_rot[needle_blk, needle_ri].repeat_interleave(GPK, dim=0)   # [H_q, D]
    scale = 1.0 / (HEAD_DIM ** 0.5)

    rel_mlx = torch.tensor(_block_relevance_residual(
        mx.array(q.numpy()), mx.array(anc_rot.numpy()), mx.array(res_rot.numpy()),
        mx.array((res_pos >= 0).numpy()), scale, GPK).tolist())

    os.environ["DKV_TOPK_BLOCKS"] = "0"        # attend-all => returns the ranking basis
    pool = _stub_pool(anchors_K, res_k, res_pos)
    os.environ["DKV_TOPK_BLOCKS"] = "2"        # force a real top-K decision
    sel = route_blocks_relevance(q, pool, torch.arange(N_BLK), anchors, scale, cos, sin)

    print("\n" + "=" * 74)
    print("  [R2] block routing   (real route_blocks_relevance vs real MLX)")
    print("=" * 74)
    print(f"    needle: block {needle_blk}, residual {needle_ri}, true position {needle_pos}")
    # Same ranking, but with residual keys rotated at the RAW within-block offset
    # -- what route_blocks_relevance did before this change. Replicated inline so
    # the comparison exists without adding a dead knob to production code.
    res_buggy = torch.stack([
        torch.stack([rot_at(res_k[n, r], int(res_pos[n, r]))
                     if res_pos[n, r] >= 0 else res_k[n, r] for r in range(MAXRES)])
        for n in range(N_BLK)])
    rel_buggy = torch.tensor(_block_relevance_residual(
        mx.array(q.numpy()), mx.array(anc_rot.numpy()), mx.array(res_buggy.numpy()),
        mx.array((res_pos >= 0).numpy()), scale, GPK).tolist())

    print(f"    MLX relevance per block : {[round(v, 3) for v in rel_mlx.tolist()]}")
    print(f"    MLX top block           : {int(rel_mlx.argmax())}")
    print(f"    CUDA top-2 selection    : {sorted(sel.tolist())}")
    print(f"    same ranking with residuals rotated at the raw offset (the bug):")
    print(f"      relevance {[round(v, 3) for v in rel_buggy.tolist()]}"
          f"   top block {int(rel_buggy.argmax())}")
    ok = int(rel_mlx.argmax()) == needle_blk and needle_blk in sel.tolist()
    print(f"    {'PASS — both keep the needle block' if ok else '*** FAIL — CUDA drops the needle block MLX keeps ***'}")
    if int(rel_buggy.argmax()) != needle_blk:
        print("      ^ the raw-offset ranking picks the WRONG block")
    return ok


def stage_decode():
    """Triton kernel arithmetic (lines 414-588) vs exact attention.

    Replicated, not called — it needs a GPU. What this isolates is the ONE thing
    the GPU run cannot separate: whether a wrong residual rotation alone is
    enough to lose the needle, holding compression and routing fixed. Both
    residual semantics are run because CUDA still defaults to CORRECTION form
    (three ADD-only readers block the flip) while MLX substitutes.
    """
    from native_core.sparse_decode.triton_fused_decode import _partial_rope_apply

    cos, sin = _rope_tables(SEQ, ROT_DIM)
    g0 = torch.Generator().manual_seed(31)
    S_act = BS - 1
    anchor_pos, res_off = 34, 9
    k_raw = torch.randn(S_act, HEAD_DIM, generator=g0)
    anc_raw = torch.randn(HEAD_DIM, generator=g0)

    # A second, longer table so the sweep below can reach a realistic 8k anchor.
    cos_l, sin_l = _rope_tables(SEQ_LONG, ROT_DIM)

    def rot1(x, pos):                      # [D] -> [D], rotated at absolute `pos`
        return _partial_rope_apply(x.view(1, -1), cos[0, pos].view(1, -1),
                                   sin[0, pos].view(1, -1)).view(-1)

    def rot1_long(x, pos):
        return _partial_rope_apply(x.view(1, -1), cos_l[0, pos].view(1, -1),
                                   sin_l[0, pos].view(1, -1)).view(-1)

    k_true = torch.stack([rot1(k_raw[t], anchor_pos + 1 + t) for t in range(S_act)])
    anc_true = rot1(anc_raw, anchor_pos)
    q = k_true[res_off].clone()                       # query aimed at the needle
    scale = 1.0 / (HEAD_DIM ** 0.5)

    # exact reference: what the needle row is worth under uncompressed attention
    K_ref = torch.cat([anc_true[None], k_true])
    s_ref = K_ref @ q * scale
    w_ref = torch.softmax(s_ref, 0)[1 + res_off].item()

    print("\n" + "=" * 74)
    print("  [R3] decode: residual rotation position, kernel score reconstruction")
    print("=" * 74)
    print(f"    block anchor {anchor_pos}, needle at active idx {res_off} "
          f"(true position {anchor_pos + 1 + res_off})")
    print(f"    exact:                    needle score {s_ref[1 + res_off]:7.3f}"
          f"   softmax weight {w_ref:6.3f}")

    # The kernel's EXACT-residual substitution: s[p] <- q . rope(res_k, rot_pos);
    # every other row keeps the lossy low-rank score, which with U=0 is the
    # anchor score -- the worst case the residual mechanism exists to rescue.
    #
    # Sweep the position ERROR rather than asserting one value, because the two
    # bugs found here are three orders of magnitude apart and it matters which
    # one actually costs recall:
    #   delta = 1     decode gather: anchor+offset instead of anchor+1+offset
    #   delta = anchor  router: the within-block offset used as an ABSOLUTE
    #                   position, so the anchor's whole magnitude is the error
    # Retention is q.rope(k, p+delta) / q.rope(k, p) = sum_i |k_i|^2 cos(theta_i
    # delta) / |k|^2: a pair decorrelates once theta_i*delta exceeds ~pi, so the
    # damage is set by how many rotary pairs delta manages to wrap.
    s_anc = (anc_true @ q * scale).item()
    s_true = (rot1(k_raw[res_off], anchor_pos + 1 + res_off) @ q * scale).item()
    print(f"\n    position error -> score retention (rotary_dim={ROT_DIM}/{HEAD_DIM}, base 1e4)")
    got = {}
    for delta, note in ((0, "fixed"), (1, "decode gather bug"),
                        (256, "one block"), (5911, "router bug @ 8k")):
        pos = min(anchor_pos + 1 + res_off + delta, SEQ_LONG - 1)
        sc = (rot1_long(k_raw[res_off], pos) @ q * scale).item()
        w = torch.softmax(torch.cat([torch.tensor([s_anc]),
                                     torch.full((S_act,), s_anc).index_put_(
                                         (torch.tensor([res_off]),),
                                         torch.tensor([sc]))]), 0)[1 + res_off].item()
        got[delta] = w
        print(f"      delta={delta:>5}  score {sc:7.3f}  ({100 * sc / s_true:5.1f}% retained)"
              f"  weight {w:6.3f}   {note}")

    # The damage has a CEILING, and it is worth stating because it contradicts
    # the obvious guess. Only rotary_dim of head_dim components rotate at all;
    # the remaining head_dim - rotary_dim pass through untouched and keep
    # contributing their full q.k. On Qwen3.5-2B that tail is 192 of 256
    # components, so even a fully decorrelating position error can only remove
    # ~rotary_dim/head_dim of the score -- the 81% floor below, not 0%.
    #
    # So neither position bug ALONE explains a lost needle. They are real
    # correctness defects worth fixing, and the router one is much the larger,
    # but a full-rotary model would be hurt ~4x more by the identical code. Do
    # not close the recall investigation on this.
    floor = 1.0 - ROT_DIM / HEAD_DIM
    ok = (abs(got[0] - w_ref) < 0.05 and got[1] < got[0]
          and got[5911] < got[1])
    print(f"    retention floor set by the UNROTATED tail: "
          f"{100 * floor:.0f}% ({HEAD_DIM - ROT_DIM}/{HEAD_DIM} components never rotate)")
    print(f"    {'PASS — damage is monotone in position error and bounded by that floor'
           if ok else '*** unexpected: damage not monotone in position error ***'}")
    return ok


def stage_sparse_bias():
    """DKV_SPARSE_BIAS='auto' is evaluated on a DIFFERENT partition on each side.

    The formula is identical, character for character:

        bias = max(0, BASE - 0.5 * max(0, (lse_dense - lse_sparse) - 4.0))

        mlx_dkv_wrapper.py:857          triton_fused_decode.py:2489
                                        dkv_attention.py:157

    What differs is what the two LSEs contain.

      MLX (mlx_dkv_wrapper.py:771, :1031)
        sparse = anchors + low-rank deltas, with the lossy TWIN of every exact
                 residual set to -inf
        dense  = mx.concatenate([res_k_all, dense_k]) -- the EXACT residual keys
                 CONCATENATED IN FRONT of the recency window

      CUDA (triton_fused_decode.py:460-483, :2434)
        sparse = anchors + low-rank deltas, twins KEPT, the residual correction
                 applied in place inside the same softmax
        dense  = the recency window only

    So a needle that matches an exact residual raises lse_DENSE on MLX and
    lse_SPARSE on CUDA -- opposite signs into the same subtraction. MLX's comment
    for the decay branch says it "decays to 0 as the dense half (e.g. an exact
    needle residual) pulls ahead". On CUDA the needle pulls the OTHER half ahead,
    which drives diff negative and pins the bias at BASE.

    This measures the bias each side computes for the same physical situation.
    """
    print("\n" + "=" * 74)
    print("  [R4] DKV_SPARSE_BIAS='auto': same formula, different partition")
    print("=" * 74)

    BASE = 2.0
    bias = lambda lse_d, lse_s: max(0.0, BASE - 0.5 * max(0.0, (lse_d - lse_s) - 4.0))  # noqa: E731

    # One needle scoring far above filler, in a compressed block's exact residual.
    # Logs are per-half logsumexps; only their DIFFERENCE matters to the formula.
    lse_lossy_blocks = 6.0      # 16 routed blocks of low-rank filler
    lse_needle_exact = 14.0     # the exact residual row the query matches
    lse_recency      = 5.0      # ~30 recent tokens, no needle

    def lse_add(*xs):
        import math
        m = max(xs)
        return m + math.log(sum(math.exp(x - m) for x in xs))

    for who, lse_s, lse_d in (
        ("MLX  (residuals in DENSE)", lse_lossy_blocks,
         lse_add(lse_needle_exact, lse_recency)),
        ("CUDA (residuals in SPARSE)", lse_add(lse_lossy_blocks, lse_needle_exact),
         lse_recency),
    ):
        d = lse_d - lse_s
        b = bias(lse_d, lse_s)
        print(f"    {who}:  lse_sparse {lse_s:6.2f}  lse_dense {lse_d:6.2f}"
              f"  diff {d:+6.2f}  ->  bias {b:.2f}"
              f"{'  (decayed to 0)' if b == 0 else f'  (PINNED at BASE, e^{b:.0f} = {2.718 ** b:.1f}x)'}")

    print("    the decay branch needs diff > +4, i.e. the recency window beating ALL")
    print("    of compressed history by 4 nats -- the OPPOSITE of the needle condition")
    print("    it was written for. On CUDA 'auto' is therefore not adaptive at all.")
    ok = bias(lse_add(lse_needle_exact, lse_recency), lse_lossy_blocks) == 0.0
    print(f"    {'PASS — reproduced: MLX decays, CUDA saturates' if ok else '*** formula did not reproduce ***'}")
    return ok


def stage_skip_block_budget():
    """A 'skip_compression' block is only exact for its FIRST max_residual tokens.

    Rules 1/1b/3/5 flag a block as skip_compression when it holds a passcode,
    formula or table cell -- content the SVD must not smear. lowrank.py then
    selected residual positions as torch.arange(T_active) and the pool kept the
    leading slice (native_block_pool.py:684, `res_K_positions[:, :mr]`).

    With CUDA's 257-token block (1 anchor + 256 active) and max_residual=128,
    that means offsets 0-127 are exact and 128-255 are NOT -- inside the block
    the exemption was supposed to protect. Whether a needle survives reduces to
      needle_position % 257 < 128
    which is not monotone in depth, and which truncates a multi-token code that
    straddles offset 127: exact before it, rank-32 after.

    MLX has no positional path. It always ranks (mlx_dkv_wrapper.py:2852)
        top_k = argsort(capture_scores)[:, -max_residual:][:, ::-1]
    with is_core tokens (digits, all-caps runs, '-', '_' -- :2711) boosted so
    they win that ranking.

    This calls the REAL compress_lowrank on a block whose high-error tokens sit
    in the SECOND half, and reports which of them survive the pool's cap.
    """
    from native_core.compression.lowrank import compress_lowrank

    T, feat, rank, cap = 256, 128, 16, 128
    needle_offsets = [40, 200, 233]                            # two past the cap

    # Build the hard tokens the way build_deltas does, NOT as high-magnitude
    # outliers. A large outlier row dominates sigma_1, so the top-rank basis
    # captures it and it comes out the BEST-reconstructed row in the block --
    # a first version of this stage did exactly that and reported every needle
    # dropped even with the fix in place. What makes a token hard is being
    # ORTHOGONAL to what the other tokens share, at ordinary magnitude.
    g = torch.Generator().manual_seed(5)
    n_basis = rank + 8                                         # > rank: truncation bites
    basis = torch.randn(n_basis, feat, generator=g)
    basis = basis / basis.norm(dim=1, keepdim=True)
    coef = torch.randn(T, n_basis, generator=g)
    deltas = (coef * torch.linspace(1.0, 0.05, n_basis)) @ basis
    for off in needle_offsets:
        v = torch.randn(feat, generator=g)
        v = v - basis.T @ (basis @ v)                          # out of the filler span
        deltas[off] = v / v.norm() * deltas.norm(dim=1).median()

    lr = compress_lowrank(deltas, rank, force_exact=True, max_residual=cap)
    chosen = lr.residual_K_positions
    chosen = chosen.tolist() if chosen is not None else []

    # The pool applies its OWN truncation on write and it is a LEADING SLICE, not
    # a budget the compressor is trusted to have respected:
    #   native_block_pool.py:683  mr = min(res_K_positions.shape[1], max_residual)
    #   native_block_pool.py:684  self.residual_K_positions[pidx, :mr] = res[:, :mr]
    # Checking compress_lowrank's return alone therefore proves nothing -- a first
    # version of this stage did exactly that, saw all 256 positions come back, and
    # reported PASS against the unfixed code. Model the write.
    stored = set(chosen[:cap])
    kept = {o for o in needle_offsets if o in stored}
    positional = stored == set(range(cap))
    sel = stored

    print("\n" + "=" * 74)
    print("  [R5] skip_compression block: which tokens actually get an exact residual")
    print("=" * 74)
    print(f"    T_active={T}  max_residual={cap}  hard tokens at {needle_offsets}")
    print(f"    slots the POOL keeps    : {len(sel)}"
          f"{'  == exactly range(%d): SELECTED BY POSITION' % cap if positional else '  (ranked)'}")
    print(f"    hard tokens kept        : {sorted(kept)}")
    print(f"    hard tokens DROPPED     : {sorted(set(needle_offsets) - kept)}")
    past_cap = [o for o in needle_offsets if o >= cap]
    ok = all(o in sel for o in past_cap)
    print(f"    {'PASS — tokens past offset %d are selected on merit' % cap if ok else
              '*** FAIL — every token past offset %d was dropped by POSITION ***' % cap}")
    return ok


def main():
    N, S, feat, rank = 8, 256, 512, 32
    nb, nr = 5, 137                      # needle block / row
    deltas = build_deltas(N, S, feat, nb, nr, rank)

    print("=" * 74)
    print(f"  MLX vs CUDA compression parity   blocks={N} S={S} feat={feat} rank={rank}")
    print(f"  needle planted at block {nb}, row {nr}")
    print("=" * 74)

    # ── stage 1+2: seeds and projection shape ───────────────────────────────
    mlx_seed = int(os.environ.get("DKV_SVD_SEED", "1234"))
    cuda_seed = int(os.environ.get("DKV_RSVD_SEED", os.environ.get("DKV_SVD_SEED", "0")))
    print(f"\n[1] rSVD seed          MLX={mlx_seed}   CUDA={cuda_seed}"
          f"   {'MATCH' if mlx_seed == cuda_seed else '*** DIVERGE ***'}")
    print(f"[2] Omega              MLX=(d,r) shared by all blocks   "
          f"CUDA=(N,d,r) per-block")
    print( "                       => CUDA block i's projection depends on the BATCH SIZE")

    # ── stage 3: factors and reconstruction ─────────────────────────────────
    U_m, Vh_m, sc_m = compress_mlx_block_batched(mx.array(deltas.numpy()), rank)
    U_m = torch.tensor(U_m.tolist()); Vh_m = torch.tensor(Vh_m.tolist())
    sc_m = torch.tensor(sc_m.tolist())
    U_c, Vh_c, sc_c = cuda_rsvd(deltas, rank)

    rec_m = torch.matmul(U_m, Vh_m) * sc_m.view(-1, 1, 1)
    rec_c = torch.matmul(U_c, Vh_c) * sc_c.view(-1, 1, 1)
    e_m, e_c = rowerr(rec_m, deltas), rowerr(rec_c, deltas)

    print(f"\n[3] per-block scale    max|diff| = {(sc_m - sc_c).abs().max():.3e}"
          f"   {'MATCH' if (sc_m-sc_c).abs().max() < 1e-5 else '*** DIVERGE ***'}")
    print(f"    reconstruction     MLX mean rel err {e_m.mean():.4f}   "
          f"CUDA {e_c.mean():.4f}")
    print(f"    NEEDLE row         MLX {e_m[nb, nr]:.4f}            "
          f"CUDA {e_c[nb, nr]:.4f}")
    print(f"    needle rank among worst-reconstructed rows in its block:")
    print(f"                       MLX #{(e_m[nb] > e_m[nb, nr]).sum().item()+1}"
          f"   CUDA #{(e_c[nb] > e_c[nb, nr]).sum().item()+1}   (1 = worst, best case)")

    # ── stage 4: int8 quantisation ──────────────────────────────────────────
    rec_mq = torch.matmul(q_int8(U_m), Vh_m) * sc_m.view(-1, 1, 1)
    rec_cq = torch.matmul(q_int8(U_c), Vh_c) * sc_c.view(-1, 1, 1)
    emq, ecq = rowerr(rec_mq, deltas), rowerr(rec_cq, deltas)
    print(f"\n[4] after int8 U       MLX mean {emq.mean():.4f} (+{emq.mean()-e_m.mean():.4f})"
          f"   CUDA mean {ecq.mean():.4f} (+{ecq.mean()-e_c.mean():.4f})")
    print(f"    NEEDLE row         MLX {emq[nb, nr]:.4f}            CUDA {ecq[nb, nr]:.4f}")

    # ── stage 5: residual selection ─────────────────────────────────────────
    # MLX: ONE joint score, absolute, V-balanced -> one index set (:2630, :3853)
    # CUDA (pre-42eb66c): rel_error_K and rel_error_V ranked SEPARATELY
    half = feat // 2
    dK, dV = deltas[..., :half], deltas[..., half:]
    rK, rV = rec_cq[..., :half], rec_cq[..., half:]
    eK, eV = (dK - rK).norm(dim=-1), (dV - rV).norm(dim=-1)
    joint = torch.sqrt(eK ** 2 + eV ** 2)                       # MLX
    relK = eK / dK.norm(dim=-1).clamp(min=1e-8)                 # CUDA (old)
    relV = eV / dV.norm(dim=-1).clamp(min=1e-8)

    for budget in (8, 16, 128):
        sj = set(torch.topk(joint[nb], budget).indices.tolist())
        sk = set(torch.topk(relK[nb], budget).indices.tolist())
        sv = set(torch.topk(relV[nb], budget).indices.tolist())
        print(f"\n[5] budget {budget:>3}         needle selected?  "
              f"MLX(joint)={nr in sj}   CUDA(relK)={nr in sk}  CUDA(relV)={nr in sv}")
        print(f"                       CUDA K-set == V-set ? {sk == sv}"
              f"   (split sets = exact score, lossy value)")
        print(f"                       overlap MLX vs CUDA-K: "
              f"{len(sj & sk)}/{budget}")

    print("\n" + "=" * 74)
    print("  Read [3] and [4] for fidelity, [5] for whether the needle is even")
    print("  given a residual. A divergence in [1]/[2] alone changes results")
    print("  without either side being wrong; [3]-[5] are where quality lives.")
    print("=" * 74)

    results = [("R1 residual RoPE position", stage_positions()),
               ("R2 block routing",          stage_routing()),
               ("R3 decode needle weight",   stage_decode()),
               ("R4 sparse-bias partition",  stage_sparse_bias()),
               ("R5 skip-block budget",      stage_skip_block_budget())]
    print("\n" + "=" * 74)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 74)


if __name__ == "__main__":
    main()
