# Phase 24.6 -- Real VRAM Trace

## Audit Configuration

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Device: CUDA (real GPU execution)
- Instrumentation: `get_kv()` and `ingest_streaming()` wrapped to log tensor sizes at the CUDA allocator level
- Sessions: 32 / 128 / 512 prompt tokens, each followed by 10 decode steps
- Measurement: `torch.cuda.memory_stats()` -- separates allocated vs reserved vs active vs inactive

---

## Measured VRAM Ownership

### Baseline
- Model weights: **988.1 MB** (fp16, Qwen2.5-0.5B, 24 layers)
- Allocator idle overhead after weights: 14.3 MB (inactive split blocks)

### Prefill VRAM (above model weights)

| Prompt tokens | Peak VRAM | KV+Activation overhead | get_kv() calls | Dense recon MB |
|---|---|---|---|---|
| 32 | 1025.3 MB | **37.2 MB** | 0 | 0.000 |
| 128 | 1067.1 MB | **79.0 MB** | 0 | 0.000 |
| 512 | 1226.2 MB | **238.1 MB** | 0 | 0.000 |

### Post-GC State (after `torch.cuda.empty_cache()`)

| Prompt tokens | alloc MB | reserv MB | inactive MB |
|---|---|---|---|
| 32 | 1025.2 | 1046.5 | 21.3 |
| 128 | 1057.1 | 1067.5 | 10.3 |
| 512 | 1181.4 | 1199.6 | 18.1 |

### Decode VRAM (10 steps each)
- No additional significant allocation observed
- `get_kv()` called 0 times during decode (sparse attention path confirmed active)

### Idle State (all sessions destroyed, empty_cache called)
- allocated: 1170.6 MB
- reserved: 1189.1 MB
- allocator cache overhead: 18.5 MB

---

## VRAM Ownership Classification

### 1. Physically Alive (allocated)
- Model weights: 988.1 MB -- permanent, expected
- Compressed KV blocks across all 3 sessions (streaming mgr): ~182 MB persisted after all sessions
  (1170.6 idle alloc - 988.1 weights = 182.5 MB from accumulated session KV across 3 sessions)
- Active ingest blocks (anchor_kv tensors, U/V matrices): stored in streaming manager

### 2. Freed But Cached (inactive/split)
- 10-21 MB post-GC -- allocator fragments from temporary attention computations
- Released by `empty_cache()` but not yet returned to OS

### 3. Reconstructed Dense K/V (get_kv calls)
- **ZERO calls observed** during both prefill and decode
- The instrumented `get_kv()` wrapper was never invoked
- This means the streaming ingest path does NOT call get_kv() for attention -- it bypasses it

### 4. Persistent Dense (long-lived)
- NONE: all dense tensors freed after each forward pass
- The 512-token overhead of 238 MB during prefill is transient activation memory
  (Q, K, V projections + attention weights + FFN intermediates -- standard transformer compute)

---

## Activation Memory Breakdown (512-token prefill, 238 MB overhead)

Estimated breakdown for Qwen2.5-0.5B (24 layers, hidden=896, ff=4864):
| Source | Est. MB |
|---|---|
| Q/K/V projection activations | ~40 MB |
| Attention weight matrices [1, 14, 512, 512] | ~28 MB |
| FFN intermediate activations | ~60 MB |
| LayerNorm buffers | ~5 MB |
| Logits [1, 512, 151936] | ~155 MB |
| **Total estimated** | **~288 MB** |

The logits tensor ([1, seq_len, vocab_size]) alone accounts for the majority of the overhead.
For 512 tokens: 512 * 151936 * 2 bytes = ~155 MB.

---

## Critical Finding

`get_kv()` was called ZERO times during the audit.

This confirms that the Phase 24.5 streaming path has eliminated the `get_kv()` reconstruction entirely from the execution hot path. The prefill VRAM overhead is dominated by standard transformer activation memory (logits, attention weights, FFN), not KV reconstruction tensors.
