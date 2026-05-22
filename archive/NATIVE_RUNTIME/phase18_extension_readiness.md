# Phase 18 Native Extension Readiness

This document outlines precisely which systems from `NATIVE_RUNTIME` are ready to be packaged as a C++/CUDA PyBind extension or integrated directly into a production serving engine like vLLM.

## 1. Ready for PyBind/C++ Extraction
- **Triton Sparse Decode (`triton_diffkv.py`):** The kernel is stable and mathematically verified. It can be compiled ahead-of-time (AOT) and exposed as a raw `torch.ops.diffkv.sparse_decode` C++ operation, completely bypassing Triton's JIT overhead during deployment.
- **Persistent Metadata Pool (`metadata_pool.py`):** The pre-allocated tensor logic maps perfectly to a C++ Struct/Class holding raw device pointers. Moving this to C++ prevents the Python garbage collector from ever touching critical runtime memory arrays.

## 2. Ready for vLLM Integration
- **Paged KV Runtime (`paged_kv_store.py`):** Differential KV’s concept of "LRU paging to CPU" maps cleanly onto vLLM's `BlockSpaceManager`. Rather than writing our own custom allocator, we should integrate Differential KV as a custom model backend in vLLM, instructing its allocator to page compressed blocks directly.
- **Async KV Compression (`async_compressor.py`):** The Python threading model works, but in vLLM, this should be written as a custom Ray worker or a dedicated C++ background thread operating entirely outside the Python Global Interpreter Lock (GIL).

## 3. Not Ready (Python Eager Retained)
- **Continuous Batch Engine:** Our custom `batch_engine.py` was highly effective for finding orchestration limits, but rebuilding a full serving engine (handling HTTP requests, tokenization, paged attention) is redundant. We should abandon `batch_engine.py` and graft the Differential KV memory modules into an existing native engine.
