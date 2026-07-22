# Phase 24.5 — OpenWebUI Behavior Analysis

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct` (via DKV serving stack)
- Frontend: Open WebUI on Docker → `http://host.docker.internal:8080/v1`
- Backend: `launch_real_serving.py` with Phase 24.5 streaming ingest active
- `micro_block_size=16`, `streaming_ingest=True`

---

## Expected Behavior by Prompt Category

### Short Prompts (< 32 tokens)
```
Example: "What is 2 + 2?"
```
- Phase 24.5: 1–2 blocks created, 0–1 compressed during ingest
- Dense footprint: ≤ 16 tokens (1 micro-block)
- VRAM growth: negligible
- Dense ratio: ~0.5 (too short for streaming to make big difference)
- **Expected**: Fast TTFT, minimal VRAM, slight compression overhead from SVD on tiny block

### Medium Prompts (128–512 tokens)
```
Example: "Explain the difference between supervised and unsupervised learning in detail."
```
- Phase 24.5: 8–32 blocks, 7–31 compressed during ingest
- Dense footprint: exactly 16 tokens (1 accumulating micro-block)
- VRAM growth: ~15× less than old dense path
- Dense ratio: 0.12–0.35 (anchor tokens + 1 accumulating block)
- **Expected**: Visible VRAM savings, compression fires during prefill, streaming quality unchanged

### Giant Prompts (2048–8192 tokens)
```
Example: Pasting a long document for summarization
```
- Phase 24.5: 128–512 blocks, 127–511 compressed during ingest
- Dense footprint: 16 tokens regardless of prompt length
- VRAM growth: bounded (O(1) dense vs. O(seq_len) old)
- Dense ratio: < 0.01 at ingest end
- **Expected**: Dramatic VRAM improvement. Old path would OOM at ~4096+ tokens; new path stays stable.

### Long Multi-Turn Conversations
```
Session: 20+ turns, 2000+ total tokens
```
- Phase 24.5: Each decode token triggers micro-block accumulation
- When 16 decode tokens accumulate → compression fires automatically
- Decode dense footprint: 1 micro-block (16 tokens) even after 1000 decode tokens
- **Expected**: Session KV stays mostly compressed. VRAM per session grows logarithmically instead of linearly.

### Idle Resume
```
Returning to a session after inactivity
```
- Phase 24.5: Compressed blocks may be paged to CPU RAM by PagedKVStore
- On resume: pager reloads compressed blocks on access
- Dense footprint on resume: only the last accumulating block
- **Expected**: Resume latency dominated by pager reload (CPU→GPU transfer), not dense recompute

---

## VRAM Behavior Under Paging Pressure

| Sessions active | Old dense VRAM | Phase 24.5 VRAM |
|---|---|---|
| 1 session × 512 tokens | ~3.5 MB | ~0.1 MB |
| 4 sessions × 512 tokens | ~14 MB | ~0.4 MB |
| 8 sessions × 2048 tokens | ~112 MB | ~0.8 MB |
| 16 sessions × 2048 tokens | OOM risk | ~1.6 MB |

(Estimates for Qwen2.5-0.5B: 16 heads, 64 head_dim, 24 layers, fp16)

---

## Known Behavior Quirks in OpenWebUI Integration

### 1. Model Name Display
OpenWebUI will show the model as `dkv-Qwen/Qwen2.5-0.5B-Instruct` — the prefix is added by the gateway's `/v1/models` endpoint.

### 2. Session Continuity
Each OpenWebUI conversation has a stable `session_id` via `ProductionSessionManager`. KV history is preserved across follow-up messages in the same tab.

### 3. Streaming Chunk Behavior
The batch engine flushes at:
- Every 6 tokens, or
- On sentence-boundary punctuation (`.`, `!`, `?`, `\n`)

This means OpenWebUI will display text in natural phrase-length chunks rather than word-by-word.

### 4. Dense Residency During Active Streaming
While the model is generating, the active micro-block (last 16 tokens) remains dense. This is correct — these tokens are still being appended to.

---

## What Validates Phase 24.5 Success in OpenWebUI

1. **Short prompts**: No degradation in response quality or TTFT
2. **Medium prompts**: Response starts within normal latency; KV is mostly compressed by first decode token
3. **Giant prompts**: No OOM errors; VRAM stays bounded; response begins normally
4. **Long conversations**: VRAM does NOT grow linearly with turn count
5. **Concurrent sessions**: Multiple tabs can co-exist without VRAM explosion

All 5 behaviors are consequences of the streaming sparse ingest eliminating dense-first KV allocation.
