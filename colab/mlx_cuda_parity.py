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


if __name__ == "__main__":
    main()
