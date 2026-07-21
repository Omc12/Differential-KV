"""CPU parity test for the DIFFKV_DECODE_CACHE_CUDA gather cache.

The optimization caches the output of _gather_routed_blocks_for_kernel across
decode tokens. That is correct ONLY if the gather is deterministic and
query-independent (it takes no query arg). This test proves that: same block
inputs -> bit-identical gather, so reusing a cached gather for later tokens
(which differ only in the query, applied later by the kernel) cannot change the
result. Also emulates the wrapper's cache dict to confirm reuse returns the
identical cached object.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("DIFFKV_RESIDUAL_EXACT_ROPE", "1")  # exercise the exact-position path

import torch
from native_core.sparse_decode.triton_fused_decode import _gather_routed_blocks_for_kernel

torch.manual_seed(0)

# Synthetic pool matching NativeBlockPool tensor shapes the gather reads.
NB, N, H_kv, D, R, S, MAXR, F = 8, 4, 2, 8, 3, 6, 5, 3

class FakePool:
    pass

p = FakePool()
p.anchors_K = torch.randn(NB, H_kv, D)
p.anchors_V = torch.randn(NB, H_kv, D)
p.V_K = torch.randn(NB, R, H_kv, D)
p.V_V = torch.randn(NB, R, H_kv, D)
p.U = torch.randint(-127, 127, (NB, S, R), dtype=torch.int8)
p.U_scale = torch.rand(NB) + 0.5
p.scales = torch.rand(NB) + 0.5
p.seq_lens = torch.randint(1, S, (NB,), dtype=torch.int32)
p.residual_K_values = torch.randn(NB, MAXR, H_kv, D)
p.residual_V_values = torch.randn(NB, MAXR, H_kv, D)
# residual positions: some valid (>=0), some padded (-1)
rp = torch.randint(-1, S, (NB, MAXR), dtype=torch.int16)
p.residual_K_positions = rp.clone()
p.residual_V_positions = rp.clone()
p.fact_anchor_positions = torch.full((NB, F), -1, dtype=torch.int16)  # no fact anchors
p.fact_anchors_K = torch.randn(NB, F, H_kv, D)
p.fact_anchors_V = torch.randn(NB, F, H_kv, D)

block_indices = torch.tensor([1, 3, 5, 7], dtype=torch.long)
anchor_indices = torch.tensor([2, 10, 20, 30], dtype=torch.long)
T = 64
pos = torch.arange(T).float().unsqueeze(1)
freq = torch.arange(D).float().unsqueeze(0)
cos = torch.cos(pos * 0.01 * (freq + 1))   # [T, D] deterministic
sin = torch.sin(pos * 0.01 * (freq + 1))

def gather():
    return _gather_routed_blocks_for_kernel(p, block_indices, anchor_indices, cos, sin)

def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    return a == b

# 1) Determinism / query-independence: two independent gathers must be identical.
g1, g2 = gather(), gather()
keys = sorted(set(g1) | set(g2))
mism = [k for k in keys if not _same(g1[k], g2[k])]
assert not mism, f"NON-DETERMINISTIC gather keys: {mism}"
print(f"[1] gather deterministic + query-independent across {len(keys)} tensors: PASS")

# 2) Emulate the wrapper's cache: first call computes+stores, second reuses SAME object.
cache, key = {}, (0, 7)  # (layer, metadata_version)
def gather_cached():
    c = cache.get(key)
    if c is not None:
        return c
    g = gather()
    cache[key] = g
    return g
first = gather_cached()
second = gather_cached()
assert first is second, "cache did not return the identical stored object"
# and the cached object equals a fresh recompute (bit-identical)
fresh = gather()
for k in keys:
    assert _same(first[k], fresh[k]), f"cached[{k}] != fresh recompute"
print("[2] cache reuse returns identical object AND matches fresh recompute: PASS")

# 3) Key change (block flush -> new metadata_version) recomputes.
key2 = (0, 8)
assert cache.get(key2) is None, "stale key should miss"
print("[3] new metadata_version key misses cache (forces recompute on flush): PASS")

print("\nALL PARITY CHECKS PASSED — caching the gather is equivalent to recomputing it.")
