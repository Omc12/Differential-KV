#!/usr/bin/env python3
"""
Active-side twin of native's DIFFKV_RECON_CMP probe.

Generates the IDENTICAL deterministic high-rank synthetic block (smooth low-rank
filler made of NCOMP cosine components + an overwhelmingly dominant landmark at
token 0 + a distinctive tail-component "needle" at token 128), then runs it
through the live active (MLX) compression path (compress_mlx_block: joint [K|V]
rank-r rSVD, NO residuals) and reports per-token relative reconstruction error.

Diff the NEEDLE line against native's:
    DIFFKV_RECON_CMP=1            ./diffkv_native/build/diffkv_native   (native, residuals ON)
    DIFFKV_RECON_NORESID=1 ...    (native, pure lowrank — apples-to-apples)
    ./diffkv_venv/bin/python3 benchmarks/recon_cmp.py                   (active)

If native (pure lowrank) ≈ active here, the compression is equivalent and the
benchmark gap lives in the DECODE attention. If active is much lower, active's
compression is genuinely better on the same data.
"""
import os
import sys
import math
import numpy as np

S, KVH, D = 256, 2, 128
F = KVH * D            # 256
NCOMP = 28
NEEDLE = 128
RANK = int(os.environ.get("DIFFKV_RANK", "16"))

ACTIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME")


def build_block():
    """Identical formula to run_recon_cmp() in src/main.cpp."""
    K = np.zeros((S, F), dtype=np.float64)
    V = np.zeros((S, F), dtype=np.float64)
    s = np.arange(S)[:, None]
    f = np.arange(F)[None, :]
    for k in range(1, NCOMP + 1):
        ph = 2.0 * math.pi * k * (s + 1) / S
        K += np.cos(ph + 1.7 * k * f / F + 0.3 * k)
        V += np.sin(ph + 1.3 * k * f / F + 0.5 * k)
    # overwhelmingly dominant landmark @0 (anchor)
    K[0] = 60.0 * np.sin(0.1 * np.arange(F))
    V[0] = 60.0 * np.cos(0.1 * np.arange(F))
    # distinctive needle @128 (tail component k=45)
    K[NEEDLE] += 5.0 * np.cos(2.0 * math.pi * 45 * np.arange(F) / F + 0.9)
    V[NEEDLE] += 5.0 * np.sin(2.0 * math.pi * 45 * np.arange(F) / F + 0.4)
    return K.astype(np.float32), V.astype(np.float32)


def main():
    K, V = build_block()
    # Use the SAME anchor native picked (printed as landmark=L; pass via DIFFKV_RECON_ANCHOR).
    # Row order doesn't affect per-token reconstruction error, so we list originals excluding A.
    A = int(os.environ.get("DIFFKV_RECON_ANCHOR", "208"))
    orig = [i for i in range(S) if i != A]
    anchor_k, anchor_v = K[A], V[A]
    dK = K[orig] - anchor_k
    dV = V[orig] - anchor_v
    deltas = np.concatenate([dK, dV], axis=1)  # [S-1, 2F]

    sys.path.insert(0, ACTIVE_DIR)
    try:
        import mlx.core as mx
        from serving.mlx_diffkv_wrapper import compress_mlx_block
        backend = "compress_mlx_block (live MLX active path)"
        U_k, Vh_k, scale, k = compress_mlx_block(mx.array(deltas), RANK)
        U_k = np.array(U_k, dtype=np.float32)
        Vh_k = np.array(Vh_k, dtype=np.float32)
        recon = (U_k @ Vh_k) * scale            # [S-1, 2F]
    except Exception as e:
        # Fallback: lowrank.py (torch) path if MLX import fails
        print(f"[RECON_CMP] MLX path unavailable ({e}); falling back to numpy rSVD parity", file=sys.stderr)
        x = deltas.astype(np.float32)
        sc = float(np.max(np.abs(x))) or 1.0
        xn = x / sc
        U, Sv, Vh = np.linalg.svd(xn, full_matrices=False)
        kk = min(RANK, U.shape[1])
        recon = ((U[:, :kk] * Sv[:kk]) @ Vh[:kk, :]) * sc
        backend = "numpy truncated SVD (fallback)"

    recon_K = anchor_k + recon[:, :F]
    recon_V = anchor_v + recon[:, F:]
    true_K = K[orig]
    true_V = V[orig]

    relK = np.linalg.norm(recon_K - true_K, axis=1) / np.maximum(np.linalg.norm(true_K, axis=1), 1e-12)
    relV = np.linalg.norm(recon_V - true_V, axis=1) / np.maximum(np.linalg.norm(true_V, axis=1), 1e-12)

    ni = orig.index(NEEDLE)  # needle's row in the anchor-excluded list
    print(f"[RECON_CMP] active backend = {backend}")
    print(f"[RECON_CMP] active rank={RANK}  anchor=token{A}  residuals=NONE (MLX path)")
    print(f"[RECON_CMP] mean rel err   K={100*relK.mean():.4f}%  V={100*relV.mean():.4f}%")
    print(f"[RECON_CMP] NEEDLE(tok{NEEDLE}) rel err  K={100*relK[ni]:.4f}%  V={100*relV[ni]:.4f}%")


if __name__ == "__main__":
    main()
