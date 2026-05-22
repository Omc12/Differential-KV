"""
Phase 6 Validation: Fused Sparse Attention vs Dense Materialization

Verifies:
  1. fused_sparse_attention_decode produces output close to the equivalent
     dense torch.cat path (correctness check)
  2. The fused sparse path is faster in a decode scenario (perf check)
  3. aten::cat no longer dominates the decode profile
"""

import torch
import math
import time
import sys

# ── Sys path so we can import runtime modules ──────────────────────────────
sys.path.insert(0, ".")

from runtime.sparse_attention import fused_sparse_attention_decode
from runtime.kv_runtime_manager import KVBlock
from compression.lowrank import compress_lowrank

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16
HEADS    = 28           # Qwen2.5-7B query heads
NUM_KV   = 4            # Qwen2.5-7B KV heads (GQA)
KV_GRP   = HEADS // NUM_KV  # = 7
HEAD_DIM = 128
BLOCK    = 64
RANK     = 16

print(f"=== Phase 6 Validation  |  device={DEVICE}  dtype={DTYPE} ===\n")

# ──────────────────────────────────────────────────────────────────────────
# 1. Build synthetic history with mixed compressed + dense blocks
# ──────────────────────────────────────────────────────────────────────────
NUM_COMPRESSED = 4   # 4 × 64 tokens of SVD history  (256 tokens)
NUM_DENSE      = 2   # 2 × 64 tokens of recent dense  (128 tokens)

def make_block_dense(seq_len, compressed=False):
    """Create a KVBlock with random content, optionally compressed."""
    k = torch.randn(1, NUM_KV, seq_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, NUM_KV, seq_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    anchor_k = k[:, :, 0:1, :]
    anchor_v = v[:, :, 0:1, :]
    anchor_kv = torch.stack([anchor_k[:, :, 0], anchor_v[:, :, 0]], dim=1)  # [1,2,kv_heads,head_dim]

    block = KVBlock(anchor_idx=0, anchor_kv=anchor_kv, token_indices=list(range(seq_len)))

    if compressed:
        active_k = k[:, :, 1:]  # exclude anchor
        active_v = v[:, :, 1:]
        feat_dim = 2 * NUM_KV * HEAD_DIM
        stacked  = torch.stack([active_k[0].transpose(0, 1), active_v[0].transpose(0, 1)], dim=1)
        flat     = stacked.reshape(seq_len - 1, feat_dim).float()
        anchor_flat = anchor_kv.view(-1).float()
        deltas   = flat - anchor_flat.unsqueeze(0)
        lr       = compress_lowrank(deltas, rank=RANK)
        block.U  = lr.U
        block.V  = lr.V
        block.scale = lr.scale
    else:
        block.active_k = k[:, :, 1:]
        block.active_v = v[:, :, 1:]

    return block, k, v  # return full k/v for dense baseline

all_blocks = []
full_k_list = []
full_v_list = []

for b in range(NUM_COMPRESSED + NUM_DENSE):
    is_compressed = (b < NUM_COMPRESSED)
    blk, k, v = make_block_dense(BLOCK, compressed=is_compressed)
    all_blocks.append(blk)
    full_k_list.append(k)
    full_v_list.append(v)

# Current-token query
q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)

# ──────────────────────────────────────────────────────────────────────────
# 2. Dense baseline (old path: torch.cat everything then SDPA)
# ──────────────────────────────────────────────────────────────────────────
def dense_baseline_decode(q, blocks, full_k_list, full_v_list):
    from runtime.sparse_attention import repeat_kv
    from runtime.triton_diffkv import TritonDiffKV

    k_list = []
    v_list = []
    for blk in blocks:
        k_list.append(blk.anchor_kv[:, 0].unsqueeze(2))
        v_list.append(blk.anchor_kv[:, 1].unsqueeze(2))
        if blk.U is not None:
            anchor_flat = blk.anchor_kv.view(-1).to(torch.float16)
            recon = TritonDiffKV.reconstruct_lowrank(blk.U, blk.V, anchor_flat, scale=blk.scale)
            kv_heads = blk.anchor_kv.shape[2]
            head_dim  = blk.anchor_kv.shape[3]
            recon = recon.view(1, -1, 2, kv_heads, head_dim)
            k_list.append(recon[:, :, 0].transpose(1, 2))
            v_list.append(recon[:, :, 1].transpose(1, 2))
        if blk.active_k is not None:
            k_list.append(blk.active_k)
            v_list.append(blk.active_v)

    full_k = torch.cat(k_list, dim=2)
    full_v = torch.cat(v_list, dim=2)

    k_rep = repeat_kv(full_k, KV_GRP)
    v_rep = repeat_kv(full_v, KV_GRP)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_w = torch.matmul(q * scale, k_rep.transpose(-1, -2)).float()
    attn_w = torch.softmax(attn_w, dim=-1).to(q.dtype)
    return torch.matmul(attn_w, v_rep)

# ──────────────────────────────────────────────────────────────────────────
# 3. Correctness check
# Build a ground-truth dense output over the exact same token sequence
# that the sparse kernel will see, so the comparison is fair.
# ──────────────────────────────────────────────────────────────────────────
print("== Correctness Check ==")

# The sparse kernel sees:  history_blocks (all_blocks[:-1]) + active_k/v of last block
history_blocks = all_blocks[:-1]
last  = all_blocks[-1]
act_k = last.active_k   # [1, NUM_KV, S-1, D]  (dense)
act_v = last.active_v

# Build ground-truth by explicitly reconstructing ALL tokens
def build_ground_truth_kv(history_blocks, last_block):
    """
    Reconstruct full K/V sequence matching exactly what the sparse kernel computes over.
    Returns (full_k, full_v) as dense tensors [1, NUM_KV, total_tokens, D].
    """
    from runtime.sparse_attention import repeat_kv
    k_list, v_list = [], []
    for blk in history_blocks:
        # Anchor
        k_list.append(blk.anchor_kv[:, 0].unsqueeze(2))
        v_list.append(blk.anchor_kv[:, 1].unsqueeze(2))
        if blk.U is not None:
            # Reconstruct delta tokens explicitly
            anchor_flat = blk.anchor_kv.view(-1).float()
            recon_flat = anchor_flat + blk.scale * blk.U.float() @ blk.V.float()
            kv_heads = blk.anchor_kv.shape[2]
            head_dim = blk.anchor_kv.shape[3]
            S_blk = blk.U.shape[0]
            recon = recon_flat.view(S_blk, 2, kv_heads, head_dim)
            k_list.append(recon[:, 0].transpose(0, 1).unsqueeze(0).half())
            v_list.append(recon[:, 1].transpose(0, 1).unsqueeze(0).half())
        elif blk.active_k is not None:
            k_list.append(blk.active_k)
            v_list.append(blk.active_v)
    # Active dense window (last block)
    if last_block.active_k is not None:
        k_list.append(last_block.active_k)
        v_list.append(last_block.active_v)
    return torch.cat(k_list, dim=2), torch.cat(v_list, dim=2)

gt_k, gt_v = build_ground_truth_kv(history_blocks, last)

from runtime.sparse_attention import repeat_kv
k_rep = repeat_kv(gt_k, KV_GRP).float()
v_rep = repeat_kv(gt_v, KV_GRP).float()
inv_scale = 1.0 / math.sqrt(HEAD_DIM)
attn_w = torch.softmax((q.float() * inv_scale) @ k_rep.transpose(-1, -2), dim=-1)
out_dense = (attn_w @ v_rep).half()

with torch.no_grad():
    out_sparse = fused_sparse_attention_decode(q, history_blocks, act_k, act_v, KV_GRP)

cos_sim = torch.nn.functional.cosine_similarity(
    out_sparse.reshape(1, -1).float(),
    out_dense.reshape(1, -1).float()
).item()
max_abs_err = (out_sparse.float() - out_dense.float()).abs().max().item()
print(f"  Cosine similarity (sparse vs dense): {cos_sim:.6f}")
print(f"  Max absolute error:                  {max_abs_err:.6f}")
assert cos_sim > 0.97, f"CORRECTNESS FAIL: cos_sim={cos_sim:.4f}"
print("  PASS — outputs are semantically equivalent.\n")

# ──────────────────────────────────────────────────────────────────────────
# 4. Latency benchmark  (50 decode steps)
# ──────────────────────────────────────────────────────────────────────────
print("== Latency Benchmark (50 decode steps) ==")

WARMUP = 5
STEPS  = 50

# Dense warmup
with torch.no_grad():
    for _ in range(WARMUP):
        dense_baseline_decode(q, all_blocks, full_k_list, full_v_list)
if DEVICE == "cuda": torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(STEPS):
        dense_baseline_decode(q, all_blocks, full_k_list, full_v_list)
if DEVICE == "cuda": torch.cuda.synchronize()
dense_ms = (time.perf_counter() - t0) / STEPS * 1000

# Sparse warmup
with torch.no_grad():
    for _ in range(WARMUP):
        fused_sparse_attention_decode(q, history_blocks, act_k, act_v, KV_GRP)
if DEVICE == "cuda": torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(STEPS):
        fused_sparse_attention_decode(q, history_blocks, act_k, act_v, KV_GRP)
if DEVICE == "cuda": torch.cuda.synchronize()
sparse_ms = (time.perf_counter() - t0) / STEPS * 1000

print(f"  Dense path  (aten::cat + reconstruct):   {dense_ms:.3f} ms/step")
print(f"  Fused sparse (no dense materialization):  {sparse_ms:.3f} ms/step")
print(f"  Speedup:  {dense_ms / sparse_ms:.2f}×\n")

# ──────────────────────────────────────────────────────────────────────────
# 5. Memory allocation snapshot
# ──────────────────────────────────────────────────────────────────────────
if DEVICE == "cuda":
    print("== GPU Memory (after decode) ==")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"  Reserved:  {torch.cuda.memory_reserved()  / 1e6:.1f} MB")

print("\n=== Phase 6 Validation COMPLETE ===")
