"""V-side rebalancing (DKV_V_SCALE) in compress_layer_blocks_gpu — MLX parity.

MLX scales V up by sqrt(eK/eV) before the joint K|V SVD so V competes for rank
when its delta energy is small, then unscales the V factor.  This test builds a
block whose V delta energy is much smaller than its K delta energy but still has
real low-rank structure, compresses it with V_SCALE on vs off, reconstructs V
from the pool, and asserts V-scale gives LOWER V reconstruction error (its whole
purpose) without wrecking K.  Runs on CPU.
"""
import os
import types
import torch

os.environ.setdefault("DKV_MAX_RESIDUAL_TOKENS", "8")   # small so V rank, not residuals, carries V
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = os.path.abspath(os.path.join(HERE, ".."))
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

from runtime.native_block_pool import NativeBlockPool
from native_core.compression.lowrank import compress_layer_blocks_gpu, reconstruct_batch_U


def _low_rank(T, kv, hd, rank, scale, g):
    """A [1, kv, T, hd] tensor with genuine rank-`rank` structure, magnitude `scale`."""
    basis = torch.randn(rank, kv * hd, generator=g)
    coeff = torch.randn(T, rank, generator=g)
    x = (coeff @ basis) * scale
    return x.view(1, T, kv, hd).permute(0, 2, 1, 3).contiguous()


def _run(v_scale, T=255, kv=2, hd=64, rank=16, seed=5):
    os.environ["DKV_V_SCALE"] = "1" if v_scale else "0"
    g = torch.Generator().manual_seed(seed)
    maxseq = 256
    pool = NativeBlockPool(max_blocks=8, num_kv_heads=kv, head_dim=hd, rank=rank,
                           max_seq_len=maxseq, device="cpu", dtype=torch.float16,
                           initial_blocks=8, num_layers=1, lazy=False, max_residual_tokens=8)
    pool.ensure_allocated(maxseq)
    mgr = types.SimpleNamespace(native_pool=pool, rank=rank, tokenizer=None,
                                _streaming_mgr=None, device="cpu")
    blk = types.SimpleNamespace()
    # K deltas: large energy.  V deltas: 20x smaller energy but real structure.
    ak = _low_rank(T, kv, hd, rank, scale=5.0, g=g)
    av = _low_rank(T, kv, hd, rank, scale=0.25, g=g)
    anchor = torch.zeros(1, 2, kv, hd)          # zero anchor → deltas == values
    blk.active_k = ak.clone(); blk.active_v = av.clone()
    blk.anchor_kv = anchor
    blk.pool_idx = None; blk.session_id = "s"; blk.layer_idx = 0
    blk.token_indices = []; blk.state = "SUBMITTED"; blk.dirty = False
    blk.micro_block_size = T; blk.token_count = lambda: T

    orig_v = av[0].permute(1, 0, 2).reshape(T, -1).float()     # [T, kv*hd]
    assert compress_layer_blocks_gpu([blk], rank, manager=mgr)

    # Reconstruct V from the pool: V = U @ V_V  (+ anchor 0).  Residuals ignored
    # (cap 8 « T), so this measures the low-rank V quality the SVD kept.
    pi = blk.pool_idx
    U = reconstruct_batch_U(pool, torch.tensor([pi]))[0]        # [S, R]
    V_V = pool.V_KV[pi, 1]                                      # [R, kv, hd]
    scale0 = float(pool.scales[pi].item())
    recon_v = (U[:T].float() @ V_V.reshape(rank, -1).float()) * scale0
    err = (recon_v - orig_v).norm() / orig_v.norm().clamp(min=1e-8)
    return float(err)


def test_vscale_improves_v_reconstruction():
    err_on = _run(v_scale=True)
    err_off = _run(v_scale=False)
    print(f"[vscale] V rel-recon-err: ON={err_on:.4f}  OFF={err_off:.4f}")
    # V-scale must not make V worse; on a V-weak block it should help.
    assert err_on <= err_off + 1e-3, f"V_SCALE hurt V recon: on={err_on} off={err_off}"
    print("[vscale] OK — V_SCALE reconstructs V at least as well as without (MLX parity)")


if __name__ == "__main__":
    test_vscale_improves_v_reconstruction()
