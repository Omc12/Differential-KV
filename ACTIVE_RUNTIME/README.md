# Differential KV Cache — Research Prototype

> **Status:** Phase 28 — Fused Sparse Execution & Production Throughput Convergence COMPLETED  
> **Goal:** Production-grade sparse KV inference runtime with Open WebUI integration.

---

## What This Is

A **sparse KV-cache inference runtime** for transformer models. Instead of storing
the full dense KV history, DiffKV compresses each block to an anchor token + a low-rank
delta (U @ V.T), enabling long-context inference with a fraction of the VRAM.

---

## Quick Start — Serve with Open WebUI

```powershell
# 1. From the ACTIVE_RUNTIME/ directory, start the server:
python -m serving.openai_compatible_api_gateway `
    --model Qwen/Qwen2.5-1.5B-Instruct `
    --host 0.0.0.0 `
    --port 8000 `
    --serving-mode balanced

# 2. In Open WebUI → Settings → Connections → OpenAI API:
#    URL: http://localhost:8000/v1   (or http://host.docker.internal:8000/v1 from Docker)
#    Key: none
```

See **[docs/open_webui_integration.md](docs/open_webui_integration.md)** for the full guide
including all serving modes, Docker tips, and troubleshooting.

---

## Architecture (Phase 28)

```
ACTIVE_RUNTIME/
├── serving/
│   ├── openai_compatible_api_gateway.py  ← FastAPI OpenAI-compatible server entry point
│   ├── batch_engine.py                   ← Continuous-batching decode loop
│   ├── hf_diffkv_wrapper.py              ← HuggingFace model + KVRuntimeManager wiring
│   └── production_session_manager.py     ← Multi-session lifecycle + LRU VRAM residency
│
├── runtime/
│   ├── diffkv_attention.py               ← HF attention monkey-patch (prefill + decode routing)
│   └── native_block_pool.py             ← Contiguous GPU memory pool (vLLM-style block table)
│
├── native_core/
│   ├── kv_runtime_manager.py            ← Master session/block/compression orchestrator
│   ├── streaming_sparse_ingest.py       ← Low-latency streaming block ingest
│   ├── recon_cache.py                   ← LRU reconstruction cache + GPU-resident pool
│   ├── compression/
│   │   ├── lowrank.py                   ← SVD low-rank delta compression
│   │   └── async_compressor.py         ← Background compression thread pool
│   ├── paging/
│   │   └── paged_kv_store.py           ← GPU→CPU spillover under memory pressure
│   └── sparse_decode/
│       ├── triton_sparse_attn.py       ← FlashDecoding Triton kernel (SRAM-resident)
│       └── triton_diffkv.py            ← Fused low-rank reconstruction kernel
│
└── docs/
    └── open_webui_integration.md        ← Open WebUI setup guide
```

---

## Serving Modes

| Mode           | Context       | VRAM    | Recommended for               |
|----------------|---------------|---------|-------------------------------|
| `lightweight`  | Short         | ~0.5 GB | CPU / low VRAM / testing      |
| `balanced`     | ~4K tokens    | ~2 GB   | Default interactive chat      |
| `performance`  | ~16K tokens   | ~8 GB   | GPU batch serving             |
| `long-context` | 32K+ tokens   | ~12 GB  | Document/code analysis        |
| `fused-sparse` | Long          | ~8 GB   | Maximum decode throughput     |

---

## Running Tests

```powershell
# From ACTIVE_RUNTIME/ directory
python -m pytest tests/ -v
```
