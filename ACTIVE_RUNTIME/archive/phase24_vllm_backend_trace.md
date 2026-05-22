# Phase 24 vLLM Backend Trace

This document maps the exact execution trace of the Differential KV backend integrated natively into vLLM's `v1` architecture.

## Integration Architecture (`vllm.attention.backends.diffkv`)

Differential KV hooks into the vLLM `v1` attention backend registry. The backend implements `AttentionBackend` and `AttentionImpl`.

### 1. Engine Initialization
- **Action:** OpenWebUI (or API client) starts vLLM with `--model d-ffkv-qwen2.5-0.5b-Instruct --kv-cache-dtype diffkv`.
- **vLLM Core:** Detects `diffkv` dtype. Instantiates `DiffKVAttentionBackend`.
- **Allocation:** The vLLM `BlockSpaceManager` is configured to allocate 3 separate physical memory pools corresponding to Slab-8, Slab-16, and Slab-32, in addition to the standard Dense Recency pool.
- **Worker Init:** `DiffKVCompressorThread` and `DiffKVPagingStream` native C++ objects are instantiated per GPU worker.

### 2. Request Ingestion & Prefill
- **Action:** User sends a 10K token prompt.
- **vLLM Scheduler:** Allocates dense blocks.
- **Forward Pass:** `DiffKVAttentionImpl.forward_prefill()` delegates to standard FlashAttention/XFormers for the dense prefill phase. KV cache is written to the Dense pool.

### 3. Continuous Decode & Async Compression
- **Action:** The model begins generating tokens autoregressively.
- **Trigger:** As dense blocks age out of the Recency Window, the vLLM scheduler (customized to track block age) triggers `compressor.submit(job)`.
- **Background SVD:** The C++ `DiffKVCompressorThread` executes cuSOLVER SVD and writes to the appropriate Slab pool (8, 16, or 32).
- **State Transition:** The native `DiffKVBlockStateTable` atomically transitions the block from `Compressing` to `CompressedResident`.

### 4. Sparse Decode Forward Pass
- **Action:** vLLM executes a decode step.
- **Graph Replay:** `are_replay_safe()` is checked. If true, the CUDA graph replays.
- **Kernel Dispatch:** `DiffKVAttentionImpl.forward_decode()` executes a split dispatch:
  1. Standard `paged_attention` runs over the Dense Recency Window.
  2. `TritonSparseDecode` runs over the `CompressedResident` blocks in the Slab pools, reading $U$ and $V$.
  3. Logsumexp reduction combines the results.

### 5. Paging Eviction/Reload
- **Action:** VRAM pressure triggers eviction.
- **Eviction:** `pager.issue_eviction(job)` asynchronously moves compressed slab blocks to CPU.
- **Reload:** When an evicted block is needed, `pager.issue_reload(job)` asynchronously loads it back. `sync_to_compute_stream()` ensures the compute stream waits safely without blocking the CPU.

## Elimination of Python Hotpaths
The Python `ContinuousBatchEngine` and all custom Python loops are GONE. vLLM's `v1` C++ core and scheduler completely own the request lifecycle, batching, and tensor parallelism. Differential KV acts purely as a memory compression and sparse attention backend.
