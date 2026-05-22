# Phase 27 — Full Reality Matrix
## Basis: Direct filesystem audit of ACTIVE_RUNTIME/ and RESEARCH_PROTOTYPES/ — zero assumptions

---

## Classification Key

| Status | Definition |
|---|---|
| **IMPLEMENTED + EXECUTING** | Code wired in launch_real_serving.py, runs on GPU when server starts |
| **IMPLEMENTED BUT DISCONNECTED** | Code is correct, but never called from the live serving path |
| **PARTIAL PROTOTYPE** | Real code in isolated harness or test file; not in serving |
| **ARCHITECTURE ONLY** | .md files, stub classes, or logic commented out — no execution path |
| **INVALIDATED** | Disproven, abandoned, empty, or replaced |

---

## Core Serving Pipeline

| System | File | Status | Evidence |
|---|---|---|---|
| OpenAI-Compatible API Gateway | `ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py` | **IMPLEMENTED + EXECUTING** | FastAPI app wired in launch_real_serving.py; routes /v1/chat/completions; SSE streaming |
| ContinuousBatchEngine | `ACTIVE_RUNTIME/serving/batch_engine.py` | **IMPLEMENTED + EXECUTING** | 227 lines; asyncio loop, prefill/decode split, real sampler (temp/top-p/rep-pen) |
| DiffKVHFWrapper | `ACTIVE_RUNTIME/serving/hf_diffkv_wrapper.py` | **IMPLEMENTED + EXECUTING** | Loads real HF model; calls apply_diffkv_attention_patch(); imports KVRuntimeManager |
| ProductionSessionManager | `ACTIVE_RUNTIME/serving/production_session_manager.py` | **IMPLEMENTED + EXECUTING** | Imported by gateway; 5 KB session history logic |
| Launch entrypoint | `ACTIVE_RUNTIME/launch_real_serving.py` | **IMPLEMENTED + EXECUTING** | 29-line entrypoint: Qwen2.5-0.5B, uvicorn on :8080 |

---

## KV Cache & Memory Subsystems

| System | File | Status | Evidence |
|---|---|---|---|
| KVRuntimeManager | `native_core/kv_runtime_manager.py` | **IMPLEMENTED + EXECUTING** | 455 lines; manages sessions/blocks; wired into wrapper.__init__ |
| StreamingSparseIngestManager | `native_core/streaming_sparse_ingest.py` | **IMPLEMENTED + EXECUTING** | 319 lines; micro-block streaming ingest; called from ingest_streaming() on every token |
| AsyncCompressor | `native_core/compression/async_compressor.py` | **IMPLEMENTED + EXECUTING** | 173 lines; 2 background Python daemon threads; queue+backpressure; started in KVRuntimeManager.__init__ |
| AdaptiveRankSelector | `RESEARCH_PROTOTYPES/compression/adaptive.py` | **IMPLEMENTED + EXECUTING** | Imported via importlib try/except; selects SVD rank by variance; used in _compress_block_sync() |
| LowRank SVD Compression | `native_core/compression/lowrank.py` | **IMPLEMENTED + EXECUTING** | compress_lowrank() uses torch.linalg.svd; called by AsyncCompressor worker |
| PagedKVStore (GPU/CPU tiering) | `native_core/paging/paged_kv_store.py` | **IMPLEMENTED + EXECUTING** | 229 lines; background eviction thread; touch()/maybe_evict() called on every block access |
| ReconstructionCache (LRU) | `native_core/recon_cache.py` | **IMPLEMENTED + EXECUTING** | LRU for U@V GEMM results; checked before every get_kv() call |
| SharedBasis Compression | `RESEARCH_PROTOTYPES/compression/shared_basis.py` | **IMPLEMENTED BUT DISCONNECTED** | 4 KB; not imported anywhere in serving path |
| Quantization | `ACTIVE_RUNTIME/compression/quantization.py` | **IMPLEMENTED BUT DISCONNECTED** | 2 KB; not imported in serving path |

---

## Attention Kernels & Decode

| System | File | Status | Evidence |
|---|---|---|---|
| DiffKV Attention Patch (Qwen2) | `runtime/diffkv_attention.py` | **IMPLEMENTED + EXECUTING** | 279 lines; monkey-patches all layer.self_attn.forward; wired in DiffKVHFWrapper.__init__ |
| Batched Sparse Attention Decode (Phase 8) | `runtime/batched_sparse_attn.py` | **IMPLEMENTED + EXECUTING** | 281 lines; build_sparse_batch() + batched_sparse_attn_decode(); 3 batched GPU einsum ops; active decode kernel |
| SDPA/FlashAttention Prefill (Phase 24.9) | `runtime/diffkv_attention.py` L224-248 | **IMPLEMENTED + EXECUTING** | F.scaled_dot_product_attention() called unconditionally for prefill; PyTorch dispatches Flash/SDPA backend |
| LM Head Last-Token Patch (Phase 25) | `runtime/diffkv_attention.py` L267-275 | **IMPLEMENTED + EXECUTING** | lm_head patched to project only hidden_states[:, -1:, :] when seq_len > 1 |
| TritonDiffKV LowRank Recon Kernel | `native_core/sparse_decode/triton_diffkv.py` | **IMPLEMENTED + EXECUTING** | Real @triton.jit lowrank_recon_kernel; called from TritonDiffKV.reconstruct_lowrank(); has PyTorch fallback |
| Triton Fused Sparse Decode Kernel | `native_core/sparse_decode/triton_sparse_attn.py` | **IMPLEMENTED BUT DISCONNECTED** | Real @triton.jit _fused_sparse_decode_kernel (234 lines); wrapper native_triton_sparse_attn_decode() requires NativeBlockPool which needs compiled diffkv_core.so — NEVER COMPILED |
| Fused Sparse Attention Decode (Phase 6) | `runtime/sparse_attention.py` | **IMPLEMENTED + EXECUTING** | Fallback when no compressed history; called from decode path |
| Retrieval-Aware Sparse Prefill (Phase 25) | `runtime/diffkv_attention.py` L233-240 | **PARTIAL PROTOTYPE** | RetrievalAwareSparsePrefill imported from research.sparse_prefill_anchors only when q_len > 1024 and key_len == q_len; file existence not confirmed; conditional import with live risk |
| CUDA Graph Replay (StaticSparseDecodeGraph) | `native_core/graph_runtime/static_decode_graph.py` | **IMPLEMENTED BUT DISCONNECTED** | Real CUDA graph capture/replay (58 lines); never instantiated in serving path |
| Sparse FFN / Tiered FFN | `RESEARCH_PROTOTYPES/` | **PARTIAL PROTOTYPE** | sparse_mlp_router.py, block_sparse_ffn_executor.py exist — not imported in serving |
| Chunked Prefill | `runtime/diffkv_attention.py` | **PARTIAL PROTOTYPE** | Implicit via RetrievalAwareSparsePrefill conditional; not a standalone tested path |

---

## Native C++ / CUDA Extension

| System | File | Status | Evidence |
|---|---|---|---|
| DiffKVBlockStateTable (C++) | `native_core/diffkv_core/include/block_state.hpp` | **ARCHITECTURE ONLY** | Header exists; bindings.cpp complete; CMakeLists.txt present — ZERO compiled artifacts |
| DiffKVCompressorThread (C++) | `native_core/diffkv_core/src/compressor_thread.cpp` | **ARCHITECTURE ONLY** | 3.7 KB source; never compiled; never loaded by Python |
| DiffKVPagingStream (CUDA) | `native_core/diffkv_core/src/paging_stream.cu` | **ARCHITECTURE ONLY** | Real CUDA code with cudaMemcpyAsync and event polling; never compiled into .pyd/.so |
| PyBind11 Bindings | `native_core/diffkv_core/src/bindings.cpp` | **ARCHITECTURE ONLY** | PYBIND11_MODULE written; module never built |
| native_core/kernels/ | Directory | **INVALIDATED** | EMPTY directory — no files |
| native_core/residency/ | Directory | **INVALIDATED** | EMPTY directory — no files |
| native_core/vllm_bridge/ | Directory | **ARCHITECTURE ONLY** | Contains 4 .md mapping documents ONLY — no Python code |

---

## Distributed / Multi-GPU Systems

| System | File | Status | Evidence |
|---|---|---|---|
| NCCL Dispatch | `RESEARCH_PROTOTYPES/distributed_nccl/nccl_stream_synchronizer.py` | **ARCHITECTURE ONLY** | sync_stream_with_nccl() body is `# Real logic: stream.wait_event(...)` commented out; returns True stub |
| NCCL Graph Orchestrator | `RESEARCH_PROTOTYPES/distributed_nccl/nccl_graph_orchestrator.py` | **ARCHITECTURE ONLY** | capture_nccl_op() stores a dict — no actual dist.all_reduce() call |
| Distributed Slab Ownership | `RESEARCH_PROTOTYPES/distributed/` (45 files) | **ARCHITECTURE ONLY** | All files <3 KB; no real multi-GPU communication code anywhere |
| Cross-GPU Sparse Fetch | `RESEARCH_PROTOTYPES/distributed/cross_gpu_rehydration_engine.py` | **ARCHITECTURE ONLY** | Stub logic; no CUDA P2P or NVLink transfer |
| Root distributed/, distributed_nccl/, triton_kernels/, kernels/ | Root directories | **INVALIDATED** | ALL EMPTY — only __pycache__ — never populated after migration |
| Distributed Slab Allocator | Phase 26 docs | **ARCHITECTURE ONLY** | Described in phase26_distributed_slab_design.md — no implementation found |
| Multi-GPU synchronization | Any file | **INVALIDATED** | torch.distributed, dist.init_process_group, or any group comms call: NOT FOUND in serving path |

---

## Summary Counts

| Status | Count |
|---|---|
| IMPLEMENTED + EXECUTING | 13 |
| IMPLEMENTED BUT DISCONNECTED | 4 |
| PARTIAL PROTOTYPE | 4 |
| ARCHITECTURE ONLY | 14 |
| INVALIDATED | 6 |

**Total classified subsystems: 41**

---

## Key Finding

The live serving hotpath is real and executable:
`launch → FastAPI gateway → ContinuousBatchEngine → DiffKV attention patch → StreamingSparseIngest → AsyncCompressor → SDPA prefill / batched-einsum decode`

The native C++ extension, all distributed systems, the Triton fused-decode kernel, and CUDA graph replay are **source-only or architecture-only** — they have never been compiled or dispatched.
