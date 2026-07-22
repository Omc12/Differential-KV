#!/usr/bin/env python3
"""Attribute DKV's CUDA prefill-compression time, stage by stage.

Why this exists
---------------
On an A100 the NAT eval reports ~6.1 s of `compress` for a 13.4K prompt (~45% of
total prefill), while MLX spends ~13% of prefill compressing.  The rSVD matmuls
in `compress_layer_blocks_gpu` only account for ~0.04 s of that at these shapes:

    49 blocks/layer x 256 tokens x 2048 feat, r_proj = rank + 5, 48 layers
    ~= 784 GFLOP total  ->  0.04 s at A100 fp32, 0.005 s with TF32

So ~99% of compress is NOT arithmetic.  The prime suspect is
`torch.linalg.svd(B)` where B is [N, r_proj, 2048]: cuSOLVER's genuinely batched
SVD (`gesvdjBatched`) only covers matrices up to 32x32, so a 53x2048 batch falls
back to a per-element loop -> 49 x 48 = 2,352 sequential cuSOLVER SVDs per
prefill.  `torch.linalg.qr` may be looping for the same reason.

This script runs the real shapes with synthetic data (no model download) and
times every stage with proper CUDA synchronisation, so the 6 s is attributed by
measurement instead of by argument.  It also A/Bs a Gram-matrix replacement for
the wide SVD and checks that the factorisation it produces is equivalent.

Usage
-----
    python colab/profile_compress_stages.py                 # defaults to the eval's shapes
    python colab/profile_compress_stages.py --rank 32       # DKV_RANK_BOOST=off equivalent
    python colab/profile_compress_stages.py --layers 4      # quicker smoke run
"""

import argparse
import time

import torch


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Stage:
    """Accumulate wall time per named stage, synchronising around each."""

    def __init__(self):
        self.t = {}

    def run(self, name, fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0)
        return out

    def report(self, total_label="TOTAL"):
        total = sum(self.t.values())
        width = max(len(k) for k in self.t)
        print(f"\n{'stage'.ljust(width)}   {'seconds':>9}   {'share':>7}")
        print("-" * (width + 22))
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]):
            print(f"{k.ljust(width)}   {v:9.3f}   {100*v/total:6.1f}%")
        print("-" * (width + 22))
        print(f"{total_label.ljust(width)}   {total:9.3f}   {100.0:6.1f}%")
        return total


def profile_current(N, T, feat, rank, layers, dev, oversample=5):
    """Time compress_layer_blocks_gpu's stages exactly as it runs today."""
    s = Stage()
    r_proj = min(rank + oversample, T, feat)

    for _ in range(layers):
        # Stand in for the per-block active_k/active_v the real path concatenates.
        deltas = s.run("1. deltas (fp32 materialise)",
                       lambda: torch.randn(N, T, feat, device=dev, dtype=torch.float32))

        # V-side rebalancing (DKV_V_SCALE, default on).
        half = feat // 2
        def _vscale():
            eK = (deltas[:, :, :half] ** 2).sum(dim=(1, 2))
            eV = (deltas[:, :, half:] ** 2).sum(dim=(1, 2))
            g = torch.sqrt(eK / eV.clamp(min=1e-12)).clamp(1.0, 1e4)
            return torch.cat([deltas[:, :, :half], deltas[:, :, half:] * g.view(-1, 1, 1)], dim=2)
        dsvd = s.run("2. V-scale (extra fp32 copy)", _vscale)

        def _norm():
            n = dsvd.norm(dim=2).clamp(min=1e-5)
            return dsvd / n.unsqueeze(2)
        dn = s.run("3. token-norm (extra fp32 copy)", _norm)

        Omega = s.run("4. Omega randn",
                      lambda: torch.randn(N, feat, r_proj, device=dev, dtype=torch.float32))
        Y = s.run("5. Y = D @ Omega", lambda: torch.matmul(dn, Omega))

        def _power():
            y = Y
            for _ in range(2):
                y = torch.matmul(dn, torch.matmul(dn.transpose(1, 2), y))
            return y
        Y2 = s.run("6. 2x power iteration", _power)

        Q = s.run("7. linalg.qr  [N,T,r]", lambda: torch.linalg.qr(Y2, mode="reduced")[0])
        B = s.run("8. B = Q^T @ D", lambda: torch.matmul(Q.transpose(1, 2), dn))
        USV = s.run("9. linalg.svd [N,r,feat]  <-- SUSPECT",
                    lambda: torch.linalg.svd(B, full_matrices=False))
        s.run("10. U = Q @ U_b", lambda: torch.matmul(Q, USV[0]))
        s.run("11. S.cpu() sync", lambda: USV[1].cpu())

        del deltas, dsvd, dn, Omega, Y, Y2, Q, B, USV
    return s


def profile_gram_svd(N, T, feat, rank, layers, dev, oversample=5):
    """Same pipeline, but the wide SVD replaced by eigh on the small Gram matrix.

    SVD(B) for B = [r, feat] with r << feat only needs U_b [r,r], S [r] and
    Vh [r,feat].  Those fall out of the eigendecomposition of the r x r Gram
    matrix B B^T:  B B^T = U_b diag(S^2) U_b^T, then Vh = diag(1/S) U_b^T B.
    That trades a 53x2048 cuSOLVER SVD for a 53x53 symmetric eigh, which is far
    cheaper per call.

    Accuracy: forming B B^T squares the condition number, so the small singular
    values lose relative precision.  Measured on CPU with a decaying synthetic
    spectrum at these shapes ([N,53,2048]):

        spectrum decay   cond(B)    SVD recon   Gram recon
        1.00             1.0e+00    5.7e-07     4.9e-07
        0.90             2.4e+02    8.3e-07     5.5e-07
        0.80             1.1e+05    7.1e-07     2.4e-04
        0.70             1.1e+08    8.0e-07     2.2e-04

    So the Gram route's worst case here is ~2.2e-4 relative.  For scale, the
    pool already stores U as int8 with a per-block scale (write_block), which
    injects ~9.2e-3 relative error — 41x LARGER.  The Gram error therefore sits
    well under the quantization noise floor the pipeline already imposes, which
    makes it an unlikely candidate to move recall.  That is an argument, not a
    result: A/B it on real needle/synthesis recall before shipping.
    """
    s = Stage()
    r_proj = min(rank + oversample, T, feat)

    for _ in range(layers):
        dn = torch.randn(N, T, feat, device=dev, dtype=torch.float32)
        dn = dn / dn.norm(dim=2).clamp(min=1e-5).unsqueeze(2)
        Omega = torch.randn(N, feat, r_proj, device=dev, dtype=torch.float32)
        Y = torch.matmul(dn, Omega)
        for _ in range(2):
            Y = torch.matmul(dn, torch.matmul(dn.transpose(1, 2), Y))
        Q = torch.linalg.qr(Y, mode="reduced")[0]
        B = torch.matmul(Q.transpose(1, 2), dn)

        def _gram():
            G = torch.matmul(B, B.transpose(1, 2))                 # [N, r, r]
            evals, evecs = torch.linalg.eigh(G)                    # ascending
            evals = evals.flip(-1).clamp(min=0)
            evecs = evecs.flip(-1)
            S = evals.sqrt()
            Vh = torch.matmul(evecs.transpose(1, 2), B) / S.clamp(min=1e-8).unsqueeze(-1)
            return evecs, S, Vh
        s.run("9'. eigh(B B^T) [N,r,r]", _gram)
        del dn, Omega, Y, Q, B
    return s


def check_gram_equivalence(N, T, feat, rank, dev, oversample=5):
    """The Gram route must reconstruct B as well as the direct SVD does."""
    r_proj = min(rank + oversample, T, feat)
    torch.manual_seed(0)
    dn = torch.randn(N, T, feat, device=dev, dtype=torch.float32)
    dn = dn / dn.norm(dim=2).clamp(min=1e-5).unsqueeze(2)
    Q = torch.linalg.qr(torch.matmul(dn, torch.randn(N, feat, r_proj, device=dev)),
                        mode="reduced")[0]
    B = torch.matmul(Q.transpose(1, 2), dn)

    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    B_svd = torch.matmul(Ub * S.unsqueeze(1), Vh)

    G = torch.matmul(B, B.transpose(1, 2))
    ev, evec = torch.linalg.eigh(G)
    ev, evec = ev.flip(-1).clamp(min=0), evec.flip(-1)
    S_g = ev.sqrt()
    Vh_g = torch.matmul(evec.transpose(1, 2), B) / S_g.clamp(min=1e-8).unsqueeze(-1)
    B_gram = torch.matmul(evec * S_g.unsqueeze(1), Vh_g)

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()

    print("\n=== Gram-vs-SVD equivalence (does the cheap route reconstruct B?) ===")
    print(f"  ||B - U S Vh||   / ||B||   direct SVD  : {rel(B_svd, B):.3e}")
    print(f"  ||B - U S Vh||   / ||B||   Gram eigh   : {rel(B_gram, B):.3e}")
    print(f"  singular values  max abs diff          : {(S - S_g).abs().max().item():.3e}")
    print("  (singular VECTORS may differ in sign/degenerate subspaces — compare")
    print("   reconstruction, not the factors themselves.)")


def probe_batched_cliff(N, feat, layers, dev):
    """Find the r_proj at which cuSOLVER stops using its genuinely batched kernel.

    cuSOLVER's batched routines cap at 32x32 (gesvdjBatched for SVD,
    syevjBatched for eigh).  Above that, PyTorch loops over the batch, paying a
    launch (and, for the Jacobi solvers, a convergence-check sync) per element.
    That loop — not the arithmetic — is what makes compress 6 s.

    If a sharp cliff exists at 32, then keeping r_proj <= 32 is worth far more
    than any other change here, and it becomes a hard design constraint:
        r_proj = rank + oversamples
        rank 48 + 5 = 53  (today, because the rank boost fires on 100% of blocks)
        rank 32 + 5 = 37
        rank 32 + 0 = 32  <- batched
        rank 27 + 5 = 32  <- batched
    """
    print("\n=== BATCHED-KERNEL CLIFF PROBE (eigh on [N,r,r], svd on [N,r,feat]) ===")
    print(f"{'r':>4} {'eigh s':>9} {'eigh ms/call':>13} {'svd s':>9} {'svd ms/call':>12}")
    calls = N * layers
    for r in (16, 24, 28, 30, 32, 33, 36, 37, 40, 48, 53):
        G = torch.randn(N, r, r, device=dev)
        G = torch.matmul(G, G.transpose(1, 2))          # symmetric PSD
        B = torch.randn(N, r, feat, device=dev)
        sync()
        t0 = time.perf_counter()
        for _ in range(layers):
            torch.linalg.eigh(G)
        sync()
        te = time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        for _ in range(layers):
            torch.linalg.svd(B, full_matrices=False)
        sync()
        ts = time.perf_counter() - t0

        mark = "  <-- BATCHED" if r <= 32 else ""
        print(f"{r:>4} {te:9.3f} {1000*te/calls:13.4f} {ts:9.3f} {1000*ts/calls:12.4f}{mark}")
        del G, B
    print("\nA sharp drop at r<=32 confirms the batched-kernel cliff.")
    print("If it is there, the real fix is to keep r_proj <= 32, not to micro-")
    print("optimise the decomposition.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=49, help="blocks per layer (13.4K prompt -> 49)")
    ap.add_argument("--tokens", type=int, default=256, help="T_active per block")
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--rank", type=int, default=48,
                    help="48 = boost fires (default today); 32 = DKV_RANK_BOOST=off")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--tf32", action="store_true", help="enable TF32 for fp32 matmul")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This profiler needs CUDA — run it on the A100 box.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    dev = "cuda:0"
    feat = 2 * args.kv_heads * args.head_dim
    N, T = args.blocks, args.tokens
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shapes: N={N} blocks/layer, T={T}, feat={feat}, rank={args.rank}, "
          f"r_proj={min(args.rank+5, T, feat)}, layers={args.layers}, TF32={args.tf32}")

    # Warm up cuSOLVER/cuBLAS handles so their one-time init is not billed to stage 1.
    _w = torch.randn(4, T, feat, device=dev)
    torch.linalg.qr(torch.randn(4, T, 16, device=dev))
    torch.linalg.svd(torch.randn(4, 16, feat, device=dev), full_matrices=False)
    torch.linalg.eigh(torch.randn(4, 16, 16, device=dev).transpose(1, 2) @ torch.randn(4, 16, 16, device=dev))
    del _w
    sync()

    print("\n=== CURRENT PATH (compress_layer_blocks_gpu, as shipped) ===")
    cur = profile_current(N, T, feat, args.rank, args.layers, dev)
    total = cur.report()
    svd_t = cur.t.get("9. linalg.svd [N,r,feat]  <-- SUSPECT", 0.0)
    qr_t = cur.t.get("7. linalg.qr  [N,T,r]", 0.0)
    print(f"\n  cuSOLVER (qr + svd) = {qr_t + svd_t:.3f} s = {100*(qr_t+svd_t)/total:.1f}% of compress")
    print(f"  svd call count      = {N * args.layers} sequential decompositions")
    if svd_t:
        print(f"  per-svd             = {1000*svd_t/(N*args.layers):.3f} ms")

    print("\n=== ALTERNATIVE: wide SVD -> eigh on the [r,r] Gram matrix ===")
    gram = profile_gram_svd(N, T, feat, args.rank, args.layers, dev)
    gram_t = sum(gram.t.values())
    print(f"  eigh(B B^T) total   = {gram_t:.3f} s")
    if svd_t and gram_t:
        print(f"  vs linalg.svd       = {svd_t:.3f} s  ->  {svd_t/gram_t:.1f}x faster")
        print(f"  projected compress  = {total - svd_t + gram_t:.3f} s (was {total:.3f} s)")

    check_gram_equivalence(min(N, 8), T, feat, args.rank, dev)

    # The most important question: is the cost a genuine cuSOLVER cliff at r>32?
    # If eigh drops sharply at r<=32, capping r_proj there beats the Gram swap.
    probe_batched_cliff(N, feat, args.layers, dev)

    print("\nNext: re-run with --rank 32 (DKV_RANK_BOOST=off) and --tf32 to")
    print("separate the rank-boost cost from the cuSOLVER cost.")


if __name__ == "__main__":
    main()
