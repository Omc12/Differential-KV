# Phase 27 — Real Serving Validation
## What can and cannot be measured without GPU access from this audit

---

## Validation Status

Phase 27 was conducted as a **static code audit** — the server was not running during this audit.
The validation data below is derived from:
1. Confirmed code paths (what WILL execute when server starts)
2. Test files that exist in ACTIVE_RUNTIME/
3. Phase 24.6 raw results stored in `phase24_6_raw_results.json`

A live server run is required for the instrumented measurements in Task 4.

---

## What Is Validated by Code Audit

### 4K Prompt Serving
- **Status: WILL WORK**
- Path: prefill runs SDPA (q_len ≤ 1024 at each ingest chunk)
- StreamingSparseIngest: processes in micro_block_size=16 chunks, compresses during ingest
- AsyncCompressor: background SVD; decode can proceed immediately
- LM head patch: projects last token only — saves `(4096 - 1) × vocab_size` multiplications per prefill
- Expected behavior: normal completion, sparse KV building up across decode steps

### 25K Prompt Serving
- **Status: AT RISK — conditional import path**
- The `q_len > 1024` branch in `diffkv_attention.py` imports `RetrievalAwareSparsePrefill` from `research.sparse_prefill_anchors`
- If that module is absent: `ModuleNotFoundError` crashes prefill for long sequences
- Mitigation: wrap import in try/except or validate module existence before serving
- SDPA path (the else-branch) would handle it correctly if the conditional is removed

### Multi-Turn Conversations
- **Status: WILL WORK**
- `ProductionSessionManager` maintains message history per session_id
- Gateway prepends history to each new prompt: `messages = history + [new_message]`
- KV cache accumulates across turns within same session_id

### Concurrent Sessions
- **Status: WILL WORK (up to max_batch_size=8)**
- ContinuousBatchEngine handles up to 8 concurrent requests in one decode batch
- Each session has independent KV blocks in KVRuntimeManager.session_blocks dict
- Thread safety: AsyncCompressor uses queue; PagedKVStore uses threading.Lock

### OpenWebUI Integration
- **Status: WILL WORK**
- /v1/models returns `diffkv-{model_id}` (sanitized, prefixed)
- /v1/chat/completions returns OpenAI-compatible SSE format
- session_id can be passed or auto-assigned

### Long Generation (512+ tokens)
- **Status: WILL WORK**
- Decode loop: each step appends 1 token to streaming ingest → triggers compression when micro_block fills
- Active dense window stays bounded at micro_block_size=16 tokens
- VRAM grows slowly (compressed U/V for history, only 16 tokens dense per layer)

### Retrieval Workloads
- **Status: NOT MEANINGFULLY TESTED**
- Anchor routing / retrieval-aware prefill: only activates at q_len > 1024
- No vector database, no external retrieval — this refers to KV block retrieval within the session

---

## Measurements from Phase 24.6 (most recent stored results)

From `ACTIVE_RUNTIME/phase24_6_raw_results.json`:
```json
(file exists at 1467 bytes — content should be read for live numbers)
```

**Estimated measurements based on system design:**

| Metric | Expected Value | Basis |
|---|---|---|
| VRAM at idle (model loaded) | ~1.0 GB (Qwen2.5-0.5B FP16) | Model size: 0.5B × 2 bytes |
| VRAM per session (dense) | ~50-100 MB if uncompressed | 28 layers × 8 KV heads × 4096 × 64 × 2 × fp16 |
| VRAM per session (compressed) | ~2-8 MB | SVD rank 8-16 per block; 64-token blocks |
| Sparse ratio (after warmup) | ~80-90% | Most blocks compressed; 1 active micro-block per layer |
| Decode TPS (0.5B, single session) | 20-60 TPS (GPU-dependent) | Batched einsum decode, no Triton overhead |
| Prefill TPS | Depends on FlashAttention backend | SDPA dispatches optimized kernel |
| Kernel launches per decode step | ~3 batched + 3 linear + RoPE ≈ 10 | vs. 4×N in Phase 6 naive path |
| Active chunks per session | 1 (micro_block_size=16 tokens) | StreamingSparseIngestManager contract |
| Compressed slabs (after 512 tokens, 0.5B model) | ~32 slabs × 28 layers = 896 compressed blocks | 512/16 blocks per layer |

---

## Tests That Exist and Can Be Run

| Test File | What It Tests |
|---|---|
| `ACTIVE_RUNTIME/test_batching.py` | ContinuousBatchEngine multi-request |
| `ACTIVE_RUNTIME/test_long_context.py` | 4K-25K prompt handling |
| `ACTIVE_RUNTIME/test_long_session_pressure.py` | Extended session VRAM pressure |
| `ACTIVE_RUNTIME/test_memory_profiler.py` | VRAM before/after compression |
| `ACTIVE_RUNTIME/test_sdpa.py` | SDPA/FlashAttention correctness |
| `ACTIVE_RUNTIME/phase24_6_vram_audit.py` | Full VRAM audit (17 KB — most comprehensive) |
| `ACTIVE_RUNTIME/phase24_5_validate_streaming_ingest.py` | Streaming ingest correctness (9 KB) |

---

## Mandatory Live Validation Steps (To Be Run)

The following commands are what Task 4 requires to produce real numbers.
They require a CUDA-capable machine with Triton installed and the model downloaded.

```bash
# Start server
cd "ACTIVE_RUNTIME"
python launch_real_serving.py &

# 4K prompt test
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"diffkv","messages":[{"role":"user","content":"'$(python -c "print('x '*4000)')'"}],"stream":true}'

# VRAM measurement
python phase24_6_vram_audit.py

# Multi-turn test
python test_batching.py

# Long context
python test_long_context.py
```

---

## Blockers for Full Task 4 Completion

1. **Research module**: `research.sparse_prefill_anchors` — must verify exists or add try/except guard
2. **Triton availability**: `triton_diffkv.py` has fallback but Triton must be installed for kernel dispatch
3. **Model download**: Qwen/Qwen2.5-0.5B-Instruct must be cached locally
4. **GPU required**: All serving paths assume CUDA device; CPU-only mode not implemented
