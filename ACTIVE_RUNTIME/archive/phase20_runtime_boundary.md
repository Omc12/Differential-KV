# Phase 20 Runtime Boundary

This document explicitly defines the boundaries between the Differential KV native core and a production serving engine (like vLLM) via the `vllm_bridge`. 

## 1. Allocator Boundaries
- **vLLM Owns:** The `BlockSpaceManager` and all raw physical GPU block allocations. vLLM dictates the maximum capacity and handles the paging of physical blocks between CPU and GPU.
- **Differential KV Owns:** The **Block Data Format**. When vLLM allocates a block, Differential KV writes $U$ and $V$ matrices into it instead of dense $[K, V]$ matrices. 

## 2. Scheduler Boundaries
- **vLLM Owns:** Request ingestion, batch size decisions, continuous batching iteration, tokenization, and EOS handling.
- **Differential KV Owns:** Nothing here. Differential KV submits to whatever sequence lengths and batch topologies vLLM schedules.

## 3. KV Ownership Boundaries
- **vLLM Owns:** The pointer arrays (block tables) mapping logical tokens to physical block IDs.
- **Differential KV Owns:** The `PersistentMetadataPool`. Differential KV maintains a shadow mapping of which physical blocks are currently compressed and what their ranks are, exposing this to the custom attention kernel.

## 4. Graph Replay Ownership
- **vLLM Owns:** The `CUDAGraph` execution. vLLM already manages graph capture for varying batch sizes (e.g., padding to fixed bucket sizes like 16, 32, 64).
- **Differential KV Owns:** Ensuring the Triton Sparse Decode kernel operates safely inside vLLM's graph boundaries without dynamic allocations or host-device synchronizations.

## 5. Custom Attention Injection Points
- **vLLM Owns:** The model forward pass and the dispatch call to `vllm.attention.ops.paged_attention`.
- **Differential KV Owns:** A drop-in replacement backend (`vllm.attention.backends.diffkv`) that substitutes standard PagedAttention with our `TritonSparseDecode` kernel when operating on compressed blocks, while falling back to standard PagedAttention for the Dense Recency Window.
