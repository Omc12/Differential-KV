# Phase 24.5 — Dense Dependency Map

## Classification of All Dense Requirements

### 1. Mathematically Required Dense

| Requirement | Why | Evidence |
|---|---|---|
| **Anchor KV per block** | Each block needs 1 dense reference point for delta computation | `KVBlock.anchor_kv` — the single token used as compression baseline |
| **Last active block during decode** | Attention must see the freshest uncompressed token just appended | `last_block.active_k` in `diffkv_attention.py:103` |
| **RoPE positional encoding** | Applied to raw Q/K before storage — cannot be compressed before application | `apply_rotary_pos_emb` in diffkv_attention.py:64 |
| **First token of any session** | No anchor exists yet — cannot compute delta from nothing | `len(blocks)==0` path in set_kv() |

These 4 requirements are **genuinely mathematically dense-critical**.
They cannot be eliminated without architectural reconstruction.

---

### 2. Replay-Capture Constrained Dense

| Requirement | Why | Can be relaxed? |
|---|---|---|
| **`dense_recency_blocks=2`** | Keeps recent history dense for fast replay without U@V reconstruction | YES — can be 1 or even 0 if recon cache is warm |
| **Prefill reconstruction via `get_kv()`** | Gets past dense KV to concatenate with current tokens | YES — can attend directly to compressed blocks |
| **Block stays dense until SVD completes** | Ensures block is readable during async compression | YES — with a read-copy mechanism |

These are correctness constraints under the **current replay model**, but can be relaxed with careful state machine management.

---

### 3. Scheduler Constrained Dense

| Requirement | Why | Can be relaxed? |
|---|---|---|
| **Full prefill dense concat** | `diffkv_attention.py:171`: `new_k = cat([past_k, curr_k])` requires full dense history | YES — can be replaced by compressed-prefill path |
| **Block full before compress** | `active_k.shape[2] >= block_size-1` gate | YES — can use partial compression with smaller micro-blocks |
| **Async copy to CPU then back** | CPU-pinned transfer adds latency and keeps GPU tensor live | YES — can compress in-place on GPU |

These are pure scheduler choices baked in. None are architectural requirements.

---

### 4. Artificially Dense (Legacy Assumptions)

| Assumption | Where | True cost |
|---|---|---|
| **Allocate ALL blocks dense at loop start** | `set_kv()` prefill loop, lines 239-256 | Peak VRAM = entire prompt dense simultaneously |
| **Compress only AFTER full loop** | Lines 260-266 after the allocation loop | No compression overlap with ingest |
| **dense_recency_blocks=2 hardcoded constant** | `kv_runtime_manager.py:104` | Keeps minimum 128 tokens dense always |
| **No micro-block streaming** | Block must fill to 63 tokens before eligible | Forces 63-token minimum dense residency per block |
| **Prefill attends to `get_kv()` dense output** | `diffkv_attention.py:158-163` | Forces dense materialization before every prefill chunk |

These are **entirely artificial constraints** with no mathematical basis.

---

## Separation Matrix

```
┌─────────────────────────────────┬────────────────────────────────┐
│      TRULY DENSE-CRITICAL       │       REPLAY-LIMITED           │
│                                 │                                │
│  • Anchor KV per block (1 tok)  │  • dense_recency_blocks=2      │
│  • Current-step active window   │  • Block readable during SVD   │
│  • RoPE application             │  • Full sequence for decode    │
│  • Session-first token          │    position_ids                │
└─────────────────────────────────┴────────────────────────────────┘
┌─────────────────────────────────┬────────────────────────────────┐
│      EASILY COMPRESSIBLE        │    ARTIFICIALLY DENSE          │
│                                 │                                │
│  • Historical prefill blocks    │  • All-blocks-dense loop       │
│  • Blocks beyond recency window │  • Compress-after-loop only    │
│  • Already-decoded history      │  • 63-token minimum block size │
│  • Previous session chunks      │  • dense_recency_blocks=2 const│
└─────────────────────────────────┴────────────────────────────────┘
```

---

## Elimination Priority

1. **HIGH IMPACT — Eliminate prefill full-dense-loop** → compress during ingest, not after
2. **HIGH IMPACT — Micro-block streaming** → reduce minimum dense residency from 63 to 8-16 tokens
3. **MEDIUM — Compressed-prefill attention** → attend to compressed history without dense materialization
4. **MEDIUM — Reduce dense_recency_blocks** → 1 is sufficient for most replay safety
5. **LOW — GPU-resident compression** → eliminate CPU bounce for fast blocks
