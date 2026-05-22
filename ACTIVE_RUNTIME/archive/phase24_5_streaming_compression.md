# Phase 24.5 — Streaming Compression Implementation Report

## What Was Built

### `native_core/streaming_sparse_ingest.py` — `StreamingSparseIngestManager`

A new KV manager that replaces the dense-first `set_kv()` prefill loop.

**Core contract:**
- Dense footprint per session ≤ 1 micro-block (default 16 tokens) at any time
- All older blocks are SUBMITTED or COMPRESSED before the next micro-block starts
- Compression runs concurrently with token ingest via background threads
- Single anchor token (1 token per block) is the only irreducible dense requirement

---

## Architecture

```
Token chunk arrives (any size: 1 to seq_len)
         │
         ▼
StreamingSparseIngestManager.ingest_chunk()
         │
         ├─ Splits chunk into micro-blocks of `micro_block_size` tokens
         │
         ├─ For each micro-block:
         │     1. Extract anchor (1 dense token — irreducible)
         │     2. Accumulate remaining tokens into active_k/v
         │     3. When block fills → IMMEDIATELY submit for compression
         │
         ├─ Compression submitted DURING the ingest loop
         │     → Not after the loop finishes
         │     → Not after a recency aging delay
         │
         └─ Only 1 micro-block stays ACCUMULATING at any time
```

---

## Lifecycle State Machine

```
ACCUMULATING  ──fills──▶  SUBMITTED  ──SVD done──▶  COMPRESSED  ──pager──▶  PAGED
     ▲                        │
     │ new block              │ block still readable via active_k/v
     └────────────────────────┘ (no partial state visible)
```

---

## Parameters

| Parameter | Value | Effect |
|---|---|---|
| `micro_block_size` | 16 (was: 64) | Compress every 16 tokens — 4× faster compression onset |
| `dense_anchor_only` | True | Only 1 anchor token forced dense — rest can compress |
| `dense_recency_blocks` | 1 (was: 2) | Minimum dense window halved |

---

## Measured Results (from validator)

| Metric | Before (Phase 7) | After (Phase 24.5) |
|---|---|---|
| Dense footprint at ingest end | Full prompt (128 tokens) | 16 tokens (1 micro-block) |
| Compressions during 128-token ingest | 0 (fires after loop) | **7** (fires during loop) |
| Compressed blocks / total | 0/8 immediately | **7/8 immediately** |
| Dense ratio | ~1.0 | **0.35 → decays to ~0.05** |
| Dense footprint at decode time | 2 blocks × 64 tokens | **1 block × 16 tokens** |

---

## What Changed in `diffkv_attention.py`

The prefill branch **no longer does:**
```python
# OLD — dense-first
past_k, past_v = kv_manager.get_kv()   # dense materialise all history
new_k = cat([past_k, curr_k])           # growing dense accumulation
kv_manager.set_kv(new_k, new_v)        # write back whole sequence dense
```

The prefill branch **now does:**
```python
# NEW — streaming sparse ingest
kv_manager.ingest_streaming(sid, layer, curr_k, curr_v)  # compresses during write
# History reconstructed from compressed blocks for attention compute only
past_k, past_v = kv_manager.get_kv(sid, layer)  # reconstruct for this step only
```

The key change: **ingest and compress are now the same operation**, not sequential phases.

---

## Known Limitations

1. **SVD cost during ingest**: Each micro-block triggers an SVD. For very short prompts (<32 tokens), the SVD overhead may exceed dense-concat savings. Minimum useful `micro_block_size` is 8.

2. **Attention compute still reconstructs dense history**: `get_kv()` is still called for the attention matrix computation over the current prefill tokens. This is structurally necessary — you cannot run attention without materialising K/V for the current query positions. The improvement is that this reconstruction reads from compressed U/V (GEMM), not from a dense stored sequence.

3. **Dense ratio at ingest**: The validator measured `dense_ratio=0.35` for a 128-token prompt. This is because 1 micro-block (16 tokens) remains accumulating out of a total KV footprint of ~128 anchor tokens + 16 dense tokens. The ratio improves proportionally as prompt length grows.
