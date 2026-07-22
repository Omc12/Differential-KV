# Phase 19 Unified Execution Trace

This trace documents the single authoritative runtime path for Differential KV following architectural convergence. Experimental branches have been stripped, and the execution is strictly defined by the `native_core`.

## 1. Request Ingestion & Batching
**Location:** `ACTIVE_RUNTIME/serving/batch_engine.py`
- `ContinuousBatchEngine` receives inference requests via async `submit()`.
- Requests are placed into `active_requests`.
- Differentiates between requests needing PREFILL (dense) and DECODE (sparse).

## 2. Prefill Ingestion (Dense)
**Location:** `ACTIVE_RUNTIME/serving/hf_dkv_wrapper.py`
- Executes standard PyTorch SDPA for token ingestion.
- Passes uncompressed Key/Value tensors directly to the `KVRuntimeManager`.

## 3. Asynchronous Compression & Memory Paging
**Location:** `ACTIVE_RUNTIME/native_core/kv_runtime_manager.py`
- Uses `Dense Recency Window` to hold the latest 128 tokens in standard VRAM.
- Pushes older blocks (64 tokens) to `ACTIVE_RUNTIME/native_core/compression/async_compressor.py`.
- Background thread executes SVD and drops rank via Adaptive Compression.
- Compressed $U$ and $V$ matrices are written directly into `ACTIVE_RUNTIME/native_core/metadata_pool/metadata_pool.py`.
- `ACTIVE_RUNTIME/native_core/paging/paged_kv_store.py` tracks VRAM capacity and pages LRU compressed blocks out to pinned CPU RAM if the budget is exceeded.

## 4. Static Sparse Decode
**Location:** `ACTIVE_RUNTIME/native_core/graph_runtime/static_decode_graph.py`
- On decode step, Python orchestration is bypassed.
- `StaticSparseDecodeGraph` uses `torch.cuda.CUDAGraph.replay()` to fire the decode kernel.
- **Location:** `ACTIVE_RUNTIME/native_core/sparse_decode/triton_dkv.py`
- The Triton Sparse Decode kernel reads from the pre-allocated metadata pool.
- $O(1)$ block-sparse generation computes the token.

## Trace Purity
This execution trace contains **zero** research systems. It is an unbroken, deterministic pipeline optimized solely for memory footprint reduction and decode throughput.
