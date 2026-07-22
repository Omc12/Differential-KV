"""Parity: NativeBlockPool.write_blocks_batched == N × write_block().

The batched pool write replaces ~2,352 per-block write_block() calls per 13K
prefill (48 layers × ~49 blocks) with one vectorized scatter.  This asserts the
two produce bit-identical pool state — U (int8) + scale, V_KV, anchors, scales,
seq_lens, residuals, versions, has_any_residual, and (with W_proj set) the SRL
descriptor — so decode reads exactly the same bytes either way.

Runs on CPU; device-agnostic.
"""
import os
import sys
import torch

os.environ.setdefault("DKV_MAX_RESIDUAL_TOKENS", "64")
HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = os.path.abspath(os.path.join(HERE, ".."))
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

from runtime.native_block_pool import NativeBlockPool

KV, HD, RANK, MAXSEQ, MAXRES = 2, 16, 32, 256, 64
DESC_DIM = 64


def _make_pool(seed=0):
    p = NativeBlockPool(
        max_blocks=64, num_kv_heads=KV, head_dim=HD, rank=RANK,
        max_seq_len=MAXSEQ, device="cpu", dtype=torch.float16,
        initial_blocks=64, num_layers=1, lazy=False, max_residual_tokens=MAXRES,
    )
    p.ensure_allocated(64 * MAXSEQ)
    # Exercise the SRL descriptor path too.
    g = torch.Generator().manual_seed(seed)
    p.W_proj = torch.randn(DESC_DIM, HD, generator=g, dtype=torch.float32)
    return p


def test_batched_write_matches_per_block():
    N, S = 10, 255
    g = torch.Generator().manual_seed(42)
    # Per-block dynamic ranks (the hard case) — batched path pads to RANK.
    ranks = [4, 8, 16, 32, 5, 20, 31, 12, 1, 9]
    assert len(ranks) == N

    # Build per-block data.
    blocks = []
    for i in range(N):
        k = ranks[i]
        U = torch.randn(S, k, generator=g, dtype=torch.float16) * (i + 1)
        V = torch.randn(k, 2 * KV * HD, generator=g, dtype=torch.float16)
        aK = torch.randn(KV, HD, generator=g, dtype=torch.float16)
        aV = torch.randn(KV, HD, generator=g, dtype=torch.float16)
        scale = float(torch.rand(1, generator=g).item()) + 0.5
        n_res = [0, 3, 64, 17, 40, 1, 64, 8, 2, 33][i]   # variable residual counts
        rK_pos = torch.randperm(S, generator=g)[:n_res].to(torch.int16)
        rK_val = torch.randn(n_res, KV * HD, generator=g, dtype=torch.float16)
        rV_pos = torch.randperm(S, generator=g)[:n_res].to(torch.int16)
        rV_val = torch.randn(n_res, KV * HD, generator=g, dtype=torch.float16)
        blocks.append(dict(U=U, V=V, aK=aK, aV=aV, scale=scale,
                           rK_pos=rK_pos, rK_val=rK_val, rV_pos=rV_pos, rV_val=rV_val))

    # ── Pool A: per-block write_block ──
    pa = _make_pool()
    idx_a = pa.allocate_blocks(N)
    for i, b in enumerate(blocks):
        pa.write_block(
            pool_idx=idx_a[i], U=b["U"], V=b["V"], anchor_K=b["aK"], anchor_V=b["aV"],
            scale=b["scale"], seq_len=S,
            residual_K_positions=(b["rK_pos"] if b["rK_pos"].numel() else None),
            residual_K_values=(b["rK_val"] if b["rK_pos"].numel() else None),
            residual_V_positions=(b["rV_pos"] if b["rV_pos"].numel() else None),
            residual_V_values=(b["rV_val"] if b["rV_pos"].numel() else None),
        )

    # ── Pool B: one batched write, U/V padded to RANK, residuals padded to max ──
    pb = _make_pool()
    idx_b = pb.allocate_blocks(N)
    assert idx_a == idx_b, "allocation order must match for a fair comparison"

    U_pad = torch.zeros(N, S, RANK, dtype=torch.float16)
    V_pad = torch.zeros(N, RANK, 2 * KV * HD, dtype=torch.float16)
    aK = torch.stack([b["aK"] for b in blocks])
    aV = torch.stack([b["aV"] for b in blocks])
    scales = torch.tensor([b["scale"] for b in blocks])
    max_res = max(b["rK_pos"].numel() for b in blocks)
    rK_pos = torch.full((N, max_res), -1, dtype=torch.int16)
    rK_val = torch.zeros(N, max_res, KV, HD, dtype=torch.float16)
    rV_pos = torch.full((N, max_res), -1, dtype=torch.int16)
    rV_val = torch.zeros(N, max_res, KV, HD, dtype=torch.float16)
    for i, b in enumerate(blocks):
        k = ranks[i]
        U_pad[i, :, :k] = b["U"]
        V_pad[i, :k, :] = b["V"]
        nr = b["rK_pos"].numel()
        if nr:
            rK_pos[i, :nr] = b["rK_pos"]
            rK_val[i, :nr] = b["rK_val"].view(nr, KV, HD)
            rV_pos[i, :nr] = b["rV_pos"]
            rV_val[i, :nr] = b["rV_val"].view(nr, KV, HD)

    pb.write_blocks_batched(
        pool_indices=torch.tensor(idx_b, dtype=torch.long),
        U=U_pad, V=V_pad, anchor_K=aK, anchor_V=aV, scales=scales, seq_len=S,
        res_K_positions=rK_pos, res_K_values=rK_val,
        res_V_positions=rV_pos, res_V_values=rV_val,
    )

    # ── Compare every pool field, bit-identical where integer/exact ──
    idx = torch.tensor(idx_a, dtype=torch.long)
    assert torch.equal(pa.U[idx], pb.U[idx]), "U int8 mismatch"
    assert torch.equal(pa.U_scale[idx], pb.U_scale[idx]), "U_scale mismatch"
    assert torch.equal(pa.V_KV[idx], pb.V_KV[idx]), "V_KV mismatch"
    assert torch.equal(pa.anchors_KV[idx], pb.anchors_KV[idx]), "anchors mismatch"
    assert torch.equal(pa.scales[idx], pb.scales[idx]), "scales mismatch"
    assert torch.equal(pa.seq_lens[idx], pb.seq_lens[idx]), "seq_lens mismatch"
    assert torch.equal(pa.residual_K_positions[idx], pb.residual_K_positions[idx]), "resK pos mismatch"
    assert torch.equal(pa.residual_K_values[idx], pb.residual_K_values[idx]), "resK val mismatch"
    assert torch.equal(pa.residual_V_positions[idx], pb.residual_V_positions[idx]), "resV pos mismatch"
    assert torch.equal(pa.residual_V_values[idx], pb.residual_V_values[idx]), "resV val mismatch"
    assert [pa.version[j] for j in idx_a] == [pb.version[j] for j in idx_b], "version mismatch"
    assert pa.has_any_residual == pb.has_any_residual, "has_any_residual mismatch"
    # Descriptor: fp16 math, allow a tiny tolerance.
    assert torch.allclose(pa.desc[idx].float(), pb.desc[idx].float(), atol=2e-3), \
        f"descriptor mismatch, max {(pa.desc[idx].float()-pb.desc[idx].float()).abs().max()}"
    print(f"[parity] OK — batched pool write bit-identical to {N} per-block "
          f"write_block() calls (ranks {sorted(set(ranks))}, residuals 0..64)")


if __name__ == "__main__":
    test_batched_write_matches_per_block()
