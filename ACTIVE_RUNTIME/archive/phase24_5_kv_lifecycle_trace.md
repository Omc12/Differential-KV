# Phase 24.5 — KV Lifecycle Trace

## Exact Block Lifecycle (Current State)

```
Token(s) arrive via attention forward
      │
      ▼
[SITE 1] dkv_attention.py:177 — set_kv(sid, layer, new_k, new_v)
  - On prefill: new_k = cat([past_dense_k, curr_k])  ← full dense concat FIRST
  - On decode:  new_k = key_states (single token)
      │
      ▼
[SITE 2] kv_runtime_manager.py:set_kv()
  - PREFILL PATH (len(blocks)==0):
      → Iterates [0 : seq_len : block_size]
      → Creates KVBlock with anchor + active_k/v (DENSE residency)
      → ALL blocks allocated dense in VRAM
      → THEN calls _submit_compression() for blocks outside dense_recency_blocks=2
      → Compression is async — blocks remain DENSE until worker finishes
  - DECODE PATH:
      → Appends to last block's active_k/v (cat in-place)
      → When block fills (active_k.shape[2] >= block_size-1): submit for compression
      │
      ▼
[SITE 3] AsyncCompressor.submit()
  - Copies k/v to CPU-pinned memory (k.to("cpu"))
  - Queues (block, k_cpu, v_cpu) for background SVD
  - Block stays DENSE in GPU until worker completes
      │
      ▼
[SITE 4] AsyncCompressor._worker_loop()
  - Moves k/v back to GPU
  - Calls compress_block_sync() → SVD → U/V assigned to block
  - Clears block.active_k = None (VRAM freed)
      │
      ▼
[SITE 5] pager.maybe_evict() — triggered after each decode step
  - Checks GPU VRAM budget
  - Evicts least-recently-used COMPRESSED blocks to CPU RAM
```

---

## Lifecycle Timing Table

| Stage | When triggered | Block state |
|---|---|---|
| Dense allocation | Immediately on set_kv() | `active_k/v` filled in GPU VRAM |
| Compression eligible | After `dense_recency_blocks=2` blocks accumulate | Still dense |
| Compression submitted | Only when `active_k.shape[2] >= block_size-1 (63 tokens)` | Still dense (queued) |
| Compression begins | When background worker dequeues | Moving CPU→GPU |
| Compression complete | Background thread finishes SVD | `U/V` set, `active_k=None` |
| Paging eligible | After compression completes | Compressed slab |
| Paged to RAM | When VRAM budget exceeded | CPU RAM |

---

## Dense Residency Measurement

For a 2048-token prefill:
- `block_size = 64` → 32 blocks created
- `dense_recency_blocks = 2` → 30 blocks submit for async compression
- **But those 30 blocks are ALL allocated dense first**
- Compression only starts after ALL blocks are created
- Peak VRAM = full 2048 tokens dense, for the duration of the SVD queue drain

For a 256-token prompt:
- 4 blocks total
- 2 blocks remain dense (recency window)
- 2 blocks queued for compression
- **Those 2 blocks still allocated dense first**
- No blocks are sparse during ingest; compression trails ingest by one pass

---

## Hardcoded Dense-First Assumptions Found

1. **`set_kv()` prefill path** (`kv_runtime_manager.py:237–266`):
   - Iterates and allocates ALL blocks dense unconditionally
   - Submits for compression AFTER the full loop completes
   - Dense allocation precedes compression eligibility by the entire loop duration

2. **`dkv_attention.py:171–175` (prefill branch)**:
   - `new_k = cat([past_k, curr_k])` — reconstructs full dense KV via `get_kv()` first
   - Then writes entire sequence back via `set_kv()`
   - This means every prefill token causes a growing dense accumulation

3. **`async_compressor.py:107–108`**:
   - `k_cpu = k.detach().to("cpu")` — copies dense tensors off GPU
   - Block occupies GPU VRAM throughout queuing phase

4. **`dense_recency_blocks=2`** hardcoded constant:
   - No mechanism to shrink this below 2
   - Even a 1-token context keeps 2 full dense blocks resident

5. **Compression only triggers on full blocks** (`active_k.shape[2] >= block_size-1`):
   - A partial block never triggers compression
   - During short decode runs, no compression occurs at all

---

## Summary

**Every prompt begins fully dense. No exceptions.**
The recency window and async compressor are post-hoc mechanisms — they reduce steady-state
VRAM but do NOT change the fact that all KV is allocated dense at ingest time.
