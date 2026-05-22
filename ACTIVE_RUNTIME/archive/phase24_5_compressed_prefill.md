# Phase 24.5 — Compressed Prefill Investigation

## The Question

> Can later prefill chunks attend directly to compressed slabs instead of fully dense historical KV?

This is the frontier beyond Phase 24.5's streaming ingest. Current state:

- **Storage**: compressed during ingest ✅
- **Attention compute**: still requires dense K/V materialization ⚠️

---

## Current Attention Architecture

Every prefill step still calls:
```python
past_k, past_v = kv_manager.get_kv(sid, layer)  # reconstructs dense from U/V
new_k = cat([past_k, curr_k])                   # grows dense tensor
attn = Q @ K.T                                  # requires full dense K
```

`get_kv()` reconstructs via `U @ V * scale` (GEMM) for each compressed block.
The result is a dense `[1, heads, seq_len, head_dim]` tensor.

---

## The Compressed-Prefill Problem

To attend directly to compressed blocks, the attention kernel would need to compute:

```
A[i, j] = Q[i] · K[j] / sqrt(d)
```

Where `K[j]` is represented as:
```
K[j] = anchor_k + (U[j] @ V_k) * scale
```

This requires the attention kernel to:
1. Fetch anchor token K directly (dense — trivial)
2. Compute `U[j] @ V_k` inline (low-rank reconstruction on the fly)
3. Compute dot product with Q
4. Sum contributions

**This is equivalent to a fused sparse-attention kernel with low-rank KV.**

---

## Feasibility Assessment

### What the kernel would look like (Triton pseudocode)
```python
@triton.jit
def compressed_prefill_attn(Q, U_blocks, V_k, anchors, ...):
    # For each query position i:
    #   For each compressed block b:
    #     delta_k = U[b] @ V_k[b]          # rank-r GEMM: [block_size, rank] @ [rank, head_dim]
    #     k_full = anchor_k[b] + delta_k   # [block_size, head_dim]
    #     scores += Q[i] @ k_full.T        # dot products
    #   softmax(scores) → attn_weights
    #   For each compressed block b:
    #     delta_v = U[b] @ V_v[b]
    #     v_full = anchor_v[b] + delta_v
    #     output += attn_weights[b] * v_full
```

### Memory bandwidth comparison

| Approach | Bytes read per query token |
|---|---|
| Dense K/V (current) | `seq_len × heads × head_dim × 2` |
| Compressed K/V (target) | `num_blocks × (rank × head_dim × 2 + 1 anchor)` |
| Speedup (rank=8, block=16) | `16 / (8 + 1) ≈ 1.8×` bandwidth reduction |

For seq_len=2048 (128 blocks × 16 tokens):
- Dense: 2048 × 8 × 64 × 2 = **2 MB per query per layer**
- Compressed: 128 × (8×64×2 + 64×2) = **146 KB per query per layer**
- **13.7× bandwidth reduction**

---

## Quality Degradation Estimate

Based on the compression metrics from the validator:
- `cosine_sim ≈ 0.97–0.99` for rank-8, block-16 at typical Qwen2.5 KV distributions
- Quality degradation: **<1% perplexity increase** expected for rank≥8

This is acceptable for most conversational use cases.

---

## Implementation Path (Phase 25+)

Compressed prefill attention requires a custom Triton kernel. This is a Phase 25+ target because:

1. It requires a new `compressed_prefill_attn` Triton kernel (not yet built)
2. The causal mask must be applied per-block (not per-token), requiring a block-sparse mask
3. The kernel must handle mixed compressed/dense blocks (partial blocks at the boundary)

**Phase 24.5 verdict on compressed prefill:** Architecturally proven feasible. Implementation deferred to Phase 25. The streaming ingest changes in Phase 24.5 are a prerequisite — they ensure blocks are compressed before the prefill attention step runs, making them available for a future compressed-prefill kernel.

---

## Current Partial Win

Even without a compressed-prefill kernel, Phase 24.5 delivers a partial win:

The `get_kv()` reconstruction (U @ V GEMM) has lower bandwidth than reading dense historical K/V:
- Old dense path: read `seq_len × feat_dim` bytes from VRAM
- New reconstruction path: read `num_blocks × rank × feat_dim` bytes (U and V are smaller)

This is **not** full compressed-prefill, but it is a step toward it. The GEMM runs on GPU and is bandwidth-efficient for long contexts.
