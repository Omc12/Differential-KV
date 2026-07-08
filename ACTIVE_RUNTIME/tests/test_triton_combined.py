"""
tests/test_triton_combined.py

Parity test: native_triton_sparse_attn_decode_combined must produce output
numerically equivalent to the 3-step reference:
    sparse Triton (no dense) → dense SDPA → Python LSE-merge

Run:
    # On a CUDA machine:
    pytest ACTIVE_RUNTIME/tests/test_triton_combined.py -v
    # Or directly:
    python ACTIVE_RUNTIME/tests/test_triton_combined.py

Falls back gracefully on CPU (no Triton) by comparing both paths to the
PyTorch reference fallback.
"""

import sys
import os
import math
import types
import torch
import torch.nn.functional as F

# Allow running from repo root or from ACTIVE_RUNTIME/
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from native_core.sparse_decode.triton_fused_decode import (
    native_triton_sparse_attn_decode,
    native_triton_sparse_attn_decode_combined,
    HAS_TRITON,
)

# ── Minimal mock objects ────────────────────────────────────────────────────

class MockPool:
    """Minimal pool with enough attributes for the Triton kernel."""
    def __init__(self, N, S_MAX, R, H_kv, D, device, dtype=torch.float16):
        self.device     = device
        self.dtype      = dtype
        self.n_kv_heads = H_kv
        self.head_dim   = D
        # Compressed block data
        self.U        = (torch.randn(N, S_MAX, R, device=device) * 0.1).to(dtype)
        self.U_scale  = torch.ones(N, device=device, dtype=dtype)
        self.V_K      = torch.randn(N, R, H_kv, D, device=device, dtype=dtype) * 0.1
        self.V_V      = torch.randn(N, R, H_kv, D, device=device, dtype=dtype) * 0.1
        self.anchors_K = torch.randn(N, H_kv, D, device=device, dtype=dtype) * 0.1
        self.anchors_V = torch.randn(N, H_kv, D, device=device, dtype=dtype) * 0.1
        self.seq_lens = torch.full((N,), S_MAX, device=device, dtype=torch.int32)
        self.scales   = torch.ones(N, device=device, dtype=dtype)
        # No residuals or fact anchors for this test
        self.residual_K_values    = None
        self.residual_V_values    = None
        self.residual_K_positions = None
        self.residual_V_positions = None
        self.fact_anchor_positions = None
        self.fact_anchors_K        = None
        self.fact_anchors_V        = None


class MockDenseBlock:
    """Minimal dense block matching what diffkv_attention.py passes."""
    def __init__(self, H_kv, L, D, device, dtype=torch.float16):
        self.anchor_kv = None
        self.active_k  = torch.randn(1, H_kv, L, D, device=device, dtype=dtype) * 0.1
        self.active_v  = torch.randn(1, H_kv, L, D, device=device, dtype=dtype) * 0.1

# ── Reference 3-step path ───────────────────────────────────────────────────

def reference_three_step(q, block_indices, pool, dense_k, dense_v,
                          num_key_value_groups, R, S_MAX, device):
    """
    Replicates the 3-step path that native_triton_sparse_attn_decode uses:
      1. sparse Triton kernel  → (out_sparse, m_sparse, l_sparse)
      2. dense SDPA            → (out_dense, lse_dense)
      3. Python LSE-merge
    """
    H_q = q.shape[1]
    D   = q.shape[3]

    # Step 1: sparse only (no dense blocks passed in)
    out_sparse_full = native_triton_sparse_attn_decode(
        q=q,
        block_indices=block_indices,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=num_key_value_groups,
        R=R,
        S_MAX=S_MAX,
    )  # [1, H_q, 1, D]

    if dense_k is None or dense_k.shape[2] == 0:
        return out_sparse_full

    # Step 2: dense SDPA
    H_kv  = dense_k.shape[1]
    L     = dense_k.shape[2]
    q_sq  = q[0, :, 0, :].float()            # [H_q, D]
    dk    = dense_k[0].float()               # [H_kv, L, D]
    dv    = dense_v[0].float()
    n_rep = num_key_value_groups

    q_r = q_sq.view(H_kv, n_rep, D)
    s   = torch.bmm(q_r, dk.permute(0, 2, 1)).view(H_q, L) / math.sqrt(D)
    lse_dense = torch.logsumexp(s, dim=-1)   # [H_q]
    w_dense   = torch.softmax(s, dim=-1)
    w_r       = w_dense.view(H_kv, n_rep, L)
    out_dense = torch.bmm(w_r, dv).view(H_q, D)  # [H_q, D]

    # Reconstruct sparse lse from the sparse-only output.
    # native_triton_sparse_attn_decode merges dense internally, so we run the
    # sparse kernel again without dense to get an isolated sparse LSE.
    # (On CUDA the kernel stores m/l — here we compute logsumexp from scores.)
    # Simple approach: use the same kernel's implicit m+log(l) by comparing
    # the two outputs after the LSE-merge. For the test we instead compute
    # the reference purely in PyTorch.
    out_sparse_hd = out_sparse_full[0, :, 0, :].float()  # [H_q, D]

    # We don't have direct access to lse_sparse from the Triton kernel here,
    # so compute a pure-PyTorch reference score for the sparse side.
    # Reconstruct K as anchor + U @ V_K  (simplified, no RoPE for this unit test)
    N  = block_indices.shape[0]
    R_ = pool.U.shape[2]
    H_kv_r = pool.anchors_K.shape[1]
    n_rep = H_q // H_kv_r

    scores_list = []
    for i in range(N):
        ni   = block_indices[i].item()
        u_b  = pool.U[ni].float()                # [S_MAX, R]
        vk_b = pool.V_K[ni].float()              # [R, H_kv, D]
        ak_b = pool.anchors_K[ni].float()        # [H_kv, D]
        sc_b = pool.scales[ni].float()
        us_b = pool.U_scale[ni].float()

        u_b = u_b * us_b
        # q_sq is [H_q, D]; project per kv_head then expand for GQA
        q_kv = q_sq.view(H_kv_r, n_rep, D)                             # [H_kv, n_rep, D]
        # q_proj: [H_kv, n_rep, R]
        q_proj_kv = torch.einsum("hnd,rhd->hnr", q_kv, vk_b) / math.sqrt(D)
        # delta: [H_kv, n_rep, S_MAX]
        delta = torch.einsum("hnr,sr->hns", q_proj_kv, u_b) * sc_b
        # s_anc: [H_kv, n_rep]
        s_anc  = torch.einsum("hnd,hd->hn", q_kv, ak_b) / math.sqrt(D)
        # tok_s: [H_kv, n_rep, S_MAX]
        tok_s = s_anc.unsqueeze(-1) + delta
        # flatten H_kv*n_rep → H_q
        s_anc_flat = s_anc.reshape(H_q).unsqueeze(1)                   # [H_q, 1]
        tok_s_flat = tok_s.reshape(H_q, -1)                            # [H_q, S_MAX]
        all_s = torch.cat([s_anc_flat, tok_s_flat], dim=1)             # [H_q, S_MAX+1]
        scores_list.append(all_s)

    all_tok_scores = torch.cat(scores_list, dim=1)                       # [H_q, total]
    lse_sparse     = torch.logsumexp(all_tok_scores, dim=-1)             # [H_q]

    # Step 3: LSE-merge
    lse_max    = torch.maximum(lse_dense, lse_sparse)
    w_d        = torch.exp(lse_dense  - lse_max)
    w_s        = torch.exp(lse_sparse - lse_max)
    denom      = (w_d + w_s).clamp(min=1e-9)
    out_merged = (out_dense * w_d.unsqueeze(-1) +
                  out_sparse_hd * w_s.unsqueeze(-1)) / denom.unsqueeze(-1)

    return out_merged.to(q.dtype).unsqueeze(0).unsqueeze(2)


# ── Test cases ───────────────────────────────────────────────────────────────

def _make_inputs(N, S_MAX, R, H_kv, L_dense, D,
                 device="cpu", dtype=torch.float16):
    H_q   = H_kv * 4  # GQA ratio = 4
    q     = torch.randn(1, H_q, 1, D, device=device, dtype=dtype) * 0.1
    pool  = MockPool(N, S_MAX, R, H_kv, D, device, dtype)
    block_indices = torch.arange(N, device=device, dtype=torch.int32)
    dense_k = (torch.randn(1, H_kv, L_dense, D, device=device, dtype=dtype) * 0.1
               if L_dense > 0 else None)
    dense_v = (torch.randn(1, H_kv, L_dense, D, device=device, dtype=dtype) * 0.1
               if L_dense > 0 else None)
    return q, pool, block_indices, dense_k, dense_v, H_kv


def test_sparse_only(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Combined kernel with L_dense=0 must match sparse-only kernel."""
    print(f"[test_sparse_only] device={device}, HAS_TRITON={HAS_TRITON}")
    N, S_MAX, R, H_kv, D = 8, 32, 8, 4, 64
    q, pool, bidx, _, _, H_kv = _make_inputs(N, S_MAX, R, H_kv, 0, D, device)

    ref = native_triton_sparse_attn_decode(
        q=q, block_indices=bidx, pool=pool,
        dense_blocks=[], active_k=None, active_v=None,
        num_key_value_groups=4, R=R, S_MAX=S_MAX,
    )
    out = native_triton_sparse_attn_decode_combined(
        q=q, block_indices=bidx, pool=pool,
        dense_k=None, dense_v=None,
        num_key_value_groups=4, R=R, S_MAX=S_MAX,
    )

    atol = 5e-3 if device == "cuda" else 1e-2
    assert ref.shape == out.shape, f"Shape mismatch: {ref.shape} vs {out.shape}"
    max_err = (ref.float() - out.float()).abs().max().item()
    print(f"  max_err={max_err:.5f}  (atol={atol})")
    assert max_err < atol, f"FAIL: max_err={max_err} > atol={atol}"
    print("  PASSED")


def test_dense_only(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Combined kernel with N=0 (no compressed blocks) must match dense SDPA."""
    print(f"[test_dense_only] device={device}")
    H_kv, L_dense, D = 4, 128, 64
    H_q = H_kv * 4
    q       = torch.randn(1, H_q, 1, D, device=device, dtype=torch.float16) * 0.1
    dense_k = torch.randn(1, H_kv, L_dense, D, device=device, dtype=torch.float16) * 0.1
    dense_v = torch.randn(1, H_kv, L_dense, D, device=device, dtype=torch.float16) * 0.1

    # Reference: pure SDPA
    dk_r  = dense_k[0].float().unsqueeze(0).expand(H_q // H_kv, -1, -1, -1)
    # easier: just use F.sdpa with repeat_kv
    dk_rp = dense_k.repeat(1, H_q // H_kv, 1, 1).view(1, H_q, L_dense, D)
    dv_rp = dense_v.repeat(1, H_q // H_kv, 1, 1).view(1, H_q, L_dense, D)
    ref = F.scaled_dot_product_attention(q.float(), dk_rp.float(), dv_rp.float())
    ref = ref.to(torch.float16)

    # Empty pool
    pool = MockPool(1, 4, 4, H_kv, D, device, torch.float16)
    bidx = torch.zeros(0, device=device, dtype=torch.int32)

    out = native_triton_sparse_attn_decode_combined(
        q=q, block_indices=bidx, pool=pool,
        dense_k=dense_k, dense_v=dense_v,
        num_key_value_groups=4, R=4, S_MAX=4,
    )

    atol = 5e-2 if device == "cpu" else 5e-3
    max_err = (ref.float() - out.float()).abs().max().item()
    print(f"  max_err={max_err:.5f}  (atol={atol})")
    assert max_err < atol, f"FAIL: max_err={max_err} > atol={atol}"
    print("  PASSED")


def test_combined_parity(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Core parity test: combined kernel must be numerically close to the
    3-step reference (sparse + dense SDPA + Python LSE-merge).
    This does NOT require exact match — numerical ordering of float ops differs.
    """
    print(f"[test_combined_parity] device={device}, HAS_TRITON={HAS_TRITON}")
    N, S_MAX, R, H_kv, L_dense, D = 6, 32, 8, 4, 64, 64
    q, pool, bidx, dense_k, dense_v, H_kv_ = _make_inputs(
        N, S_MAX, R, H_kv, L_dense, D, device
    )

    ref = reference_three_step(q, bidx, pool, dense_k, dense_v,
                                num_key_value_groups=4, R=R, S_MAX=S_MAX,
                                device=device)
    out = native_triton_sparse_attn_decode_combined(
        q=q, block_indices=bidx, pool=pool,
        dense_k=dense_k, dense_v=dense_v,
        num_key_value_groups=4, R=R, S_MAX=S_MAX,
    )

    # Loose tolerance: the reference uses a different computation order for
    # the LSE merge vs. the in-kernel online softmax — results are equivalent
    # in exact arithmetic but may differ by up to ~1e-2 in float16.
    atol = 2e-2
    max_err = (ref.float() - out.float()).abs().max().item()
    mean_err = (ref.float() - out.float()).abs().mean().item()
    print(f"  max_err={max_err:.5f}  mean_err={mean_err:.6f}  (atol={atol})")
    assert max_err < atol, f"FAIL: max_err={max_err} > atol={atol}"
    print("  PASSED")


def test_shape_invariants(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Output shape must always be [1, H_q, 1, D] regardless of inputs."""
    print(f"[test_shape_invariants] device={device}")
    for N, L in [(0, 64), (4, 0), (4, 64), (16, 256)]:
        q, pool, bidx, dk, dv, _ = _make_inputs(4, 32, 8, 4, L, 64, device)
        if N == 0:
            bidx = torch.zeros(0, device=device, dtype=torch.int32)
        out = native_triton_sparse_attn_decode_combined(
            q=q, block_indices=bidx, pool=pool,
            dense_k=dk, dense_v=dv,
            num_key_value_groups=4, R=8, S_MAX=32,
        )
        assert out.shape == (1, 16, 1, 64), f"Bad shape for N={N},L={L}: {out.shape}"
        print(f"  N={N} L={L} → {out.shape}  OK")
    print("  PASSED")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not HAS_TRITON and device == "cpu":
        print("[WARNING] Triton not available — testing PyTorch fallback paths only.")

    test_shape_invariants(device)
    test_sparse_only(device)
    test_dense_only(device)
    test_combined_parity(device)

    print("\n✓ All tests passed.")
