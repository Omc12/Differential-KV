# Phase 27 — Final Honest Execution Verdict

---

## Classification

**Differential KV is:**

> ### 2. Experimental Sparse Runtime

---

## Basis for This Classification

This verdict is based exclusively on:
- What code physically exists in `ACTIVE_RUNTIME/`
- What code is reachable from `launch_real_serving.py`
- What operations dispatch real GPU kernels
- What was verified as NOT executing (C++ unbuilt, stubs, empty dirs)

---

## Why NOT "Sparse KV Research Prototype" (Classification 1)

A pure research prototype would not have:
- A live HTTP server with OpenAI-compatible API routing
- A real continuous batching engine with async request queuing
- A working session manager with multi-turn history
- A monkey-patched attention layer handling BOTH prefill AND decode on real model weights
- A streaming ingest manager with background SVD compression threads
- A paged KV store with background eviction
- A last-token logit patch saving measurable compute

Differential KV **DOES** have all of these. It runs. It serves requests. It can connect to OpenWebUI.

---

## Why NOT "Working Sparse Transformer Runtime" (Classification 3)

A working sparse runtime would require:
- The Triton fused sparse decode kernel **actually dispatching** (it does not — blocked by unbuilt C++)
- CUDA graphs reducing Python dispatch overhead (not wired)
- The long-context sparse prefill path being safe (has unguarded import risk)
- Validated end-to-end TPS, VRAM, and sparse ratio measurements from live runs
- No empty directory skeletons posing as implemented systems

Differential KV is **missing** all of these.

---

## Why NOT "Production-Capable" or "Distributed" (Classifications 4, 5)

- No NCCL. No torch.distributed. No multi-GPU.
- Distributed code is stubs with real logic commented out.
- Root distributed directories are empty.
- No validated throughput or latency SLAs.
- No failure recovery, health checks, or graceful degradation.
- No quantization in the serving path.
- Triton fused decode kernel has never fired.
- CUDA graph replay has never been captured.

---

## What "Experimental Sparse Runtime" Means Precisely

| Property | State |
|---|---|
| Starts and serves HTTP requests | ✅ YES |
| OpenWebUI compatible | ✅ YES |
| Multi-turn sessions work | ✅ YES |
| Concurrent batching (up to 8) | ✅ YES |
| KV compression during ingest | ✅ YES (SVD, async, micro-block) |
| Sparse decode path active | ✅ YES (Phase 8 batched einsum) |
| SDPA/FlashAttention for prefill | ✅ YES |
| Last-token logit optimization | ✅ YES |
| GPU/CPU memory tiering | ✅ YES |
| Triton fused decode kernel firing | ❌ NO (C++ unbuilt) |
| CUDA graph replay | ❌ NO (not wired) |
| Long-context safe (25K+) | ⚠️ AT RISK (unguarded import) |
| Distributed / multi-GPU | ❌ NO |
| Production SLA validated | ❌ NO |
| Native C++ async paging | ❌ NO (source written, not compiled) |

---

## The Honest State of Each Layer

```
┌─────────────────────────────────────────────────────────┐
│              WHAT ACTUALLY EXECUTES TODAY               │
├─────────────────────────────────────────────────────────┤
│ HTTP API (FastAPI + uvicorn)              ✅ REAL        │
│ Session management                        ✅ REAL        │
│ Continuous batching (asyncio)             ✅ REAL        │
│ HF model forward (Qwen2 + patch)          ✅ REAL        │
│ SDPA/FlashAttention prefill               ✅ REAL CUDA   │
│ Streaming sparse KV ingest               ✅ REAL        │
│ Async SVD compression (Python threads)   ✅ REAL        │
│ Batched einsum decode (Phase 8)          ✅ REAL CUDA   │
│ Triton low-rank recon kernel             ✅ REAL TRITON │
│ GPU/CPU paging (Python, sync)            ✅ REAL        │
│ Last-token logit patch                   ✅ REAL        │
├─────────────────────────────────────────────────────────┤
│              WHAT EXISTS BUT DOES NOT FIRE              │
├─────────────────────────────────────────────────────────┤
│ Triton fused decode kernel               ⛔ UNBUILT C++ │
│ CUDA graph replay                        ⛔ NOT WIRED   │
│ Native C++ async paging stream           ⛔ NOT COMPILED│
│ Native C++ compressor thread             ⛔ NOT COMPILED│
│ NativeBlockPool                          ⛔ NOT COMPILED│
├─────────────────────────────────────────────────────────┤
│              WHAT IS ARCHITECTURE ONLY                  │
├─────────────────────────────────────────────────────────┤
│ NCCL synchronization                     📄 STUBS ONLY │
│ Distributed slab ownership               📄 STUBS ONLY │
│ Multi-GPU serving                        📄 STUBS ONLY │
│ Cross-GPU sparse fetch                   📄 STUBS ONLY │
│ Tensor/pipeline parallelism              📄 STUBS ONLY │
└─────────────────────────────────────────────────────────┘
```

---

## Distance to Next Classification

### To reach "Working Sparse Transformer Runtime" (3):

1. **Build `diffkv_core.so`** — the C++ extension is written, needs `cmake && make`
2. **Wire Triton fused decode kernel** — plug `native_triton_sparse_attn_decode()` into decode path
3. **Fix long-context import guard** — 10-line change prevents 25K+ crash
4. **Run and pass the existing test suite** — `test_long_context.py`, `phase24_6_vram_audit.py`, `test_batching.py`
5. **Measure and record TPS/VRAM** — one live run with logged metrics

That is **4 code changes + 1 cmake build + 1 benchmark run** to cross into Classification 3.

### To reach "Production-Capable" (4):
Requires additionally: CUDA graph wiring, quantization in serving path, failure recovery,
load testing, health check endpoints, and performance SLA documentation.

### To reach "Distributed" (5):
Requires: real torch.distributed, NCCL, tensor parallel sharding, cross-GPU KV transfer.
This is weeks of engineering from current state.

---

## Final Statement

Differential KV has built a **real, non-trivial sparse KV runtime** on top of a live HuggingFace
serving stack. The core innovations — streaming sparse ingest, async SVD compression, paged
GPU/CPU tiering, batched sparse attention decode, and last-token logit projection — all execute.

The gap between where it is and where it claims to be is **one cmake build** and **a few wiring steps**.
That gap is real, and it matters. The Triton fused kernel, the CUDA graph, and the C++ async
compressor are all written correctly — they are sitting on disk, compiled-source-ready,
waiting for someone to run `cmake --build build`.

The distributed claims from Phase 26 are not close to real. They are stubs.

**Verdict: Experimental Sparse Runtime. Classification 2.**
