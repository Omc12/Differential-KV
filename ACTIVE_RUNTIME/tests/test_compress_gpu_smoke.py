"""End-to-end smoke test for the batched compress_layer_blocks_gpu (lowrank.py).

Drives the real function on CPU with real StreamingKVBlock-like blocks and a
real NativeBlockPool — catching integration bugs (tensor shapes, dtypes,
pool.write_block, rank_mask) that the pure-arithmetic parity test can't see.
Runs anywhere torch runs; no CUDA/tokenizer needed (tokenizer=None disables the
content-boost path so the batched numeric core is exercised in isolation).
"""
import os
import types
import torch

os.environ.setdefault("DKV_MAX_RESIDUAL_TOKENS", "64")

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = os.path.abspath(os.path.join(HERE, ".."))
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

from runtime.native_block_pool import NativeBlockPool
from native_core.compression.lowrank import compress_layer_blocks_gpu


def _make_block(anchor_idx, T, kv_heads, head_dim, seed):
    g = torch.Generator().manual_seed(seed)
    blk = types.SimpleNamespace()
    blk.active_k = torch.randn(1, kv_heads, T, head_dim, generator=g)
    blk.active_v = torch.randn(1, kv_heads, T, head_dim, generator=g)
    blk.anchor_kv = torch.randn(1, 2, kv_heads, head_dim, generator=g)
    blk.pool_idx = None
    blk.session_id = "s"
    blk.layer_idx = 0
    blk.anchor_idx = anchor_idx
    blk.token_indices = []
    blk.state = "SUBMITTED"
    blk.dirty = False
    blk.micro_block_size = T
    blk.token_count = lambda: T
    return blk


def test_compress_gpu_end_to_end():
    N, T, kv_heads, head_dim, rank = 6, 255, 2, 128, 32
    max_seq_len = 256

    pool = NativeBlockPool(
        max_blocks=64, num_kv_heads=kv_heads, head_dim=head_dim, rank=rank,
        max_seq_len=max_seq_len, device="cpu", dtype=torch.float16,
        initial_blocks=64, num_layers=1, lazy=False, max_residual_tokens=64,
    )
    pool.ensure_allocated(N * max_seq_len)

    mgr = types.SimpleNamespace()
    mgr.native_pool = pool
    mgr.rank = rank
    mgr.tokenizer = None          # disables content-boost path
    mgr._streaming_mgr = None
    mgr.device = "cpu"

    blocks = [_make_block(i * max_seq_len, T, kv_heads, head_dim, seed=100 + i)
              for i in range(N)]

    # Snapshot block-0's raw K before compress nulls it (for the recon check).
    orig_K = blocks[0].active_k[0].permute(1, 0, 2).reshape(T, -1).float().clone()

    ok = compress_layer_blocks_gpu(blocks, rank, manager=mgr)
    assert ok is True, "compress_layer_blocks_gpu returned False"

    # Every block must be COMPRESSED, have a pool slot, and cleared raw KV.
    for i, blk in enumerate(blocks):
        assert blk.state == "COMPRESSED", f"block {i} state={blk.state}"
        assert blk.pool_idx is not None, f"block {i} not written to pool"
        assert blk.active_k is None and blk.active_v is None, \
            f"block {i} raw KV not released"

    # Pool must hold non-trivial compressed content for each written slot.
    for blk in blocks:
        pi = blk.pool_idx
        assert int(pool.seq_lens[pi].item()) == T, "seq_len not written"
        _ver = pool.version[pi]
        assert int(_ver.item() if hasattr(_ver, "item") else _ver) >= 1, "version not bumped"
        assert pool.U[pi].abs().sum().item() > 0, "U slot is all-zero"
        assert pool.V_KV[pi].abs().sum().item() > 0, "V slot is all-zero"

    # Reconstruction sanity: rebuild K for one block from the pool and check it
    # tracks the original active_k better than the anchor alone (SVD captured
    # real structure, not noise).
    from native_core.compression.lowrank import reconstruct_batch_U
    idx = torch.tensor([blocks[0].pool_idx], dtype=torch.long)
    U0 = reconstruct_batch_U(pool, idx)[0]                     # [S, R] (int8-dequant)
    V_K0 = pool.V_KV[blocks[0].pool_idx, 0]                    # [R, kv_heads, head_dim]
    scale0 = float(pool.scales[blocks[0].pool_idx].item())
    recon_flat = (U0[:T].float() @ V_K0.reshape(rank, -1).float()) * scale0  # [T, kv*hd]
    anchor_K = pool.anchors_KV[blocks[0].pool_idx, 0].reshape(-1).float()    # [kv*hd]
    recon_K = recon_flat + anchor_K.unsqueeze(0)              # add anchor back
    err = (recon_K - orig_K).norm() / orig_K.norm()
    anchor_err = (anchor_K.unsqueeze(0) - orig_K).norm() / orig_K.norm()
    assert err < anchor_err, \
        f"recon ({err:.3f}) no better than anchor-only ({anchor_err:.3f})"
    print(f"[smoke] OK — {N} blocks compressed + written to pool; "
          f"block0 recon rel-err {err:.3f} < anchor-only {anchor_err:.3f}")


if __name__ == "__main__":
    test_compress_gpu_end_to_end()
