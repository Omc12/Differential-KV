# Phase 23 Native Runtime Trace

Following the extraction of `libdiffkv_core`, the runtime trace is now definitively split between native and Python responsibilities.

## Authoritative Execution Trace (Post Phase 23)

```
Request Ingestion
    │ [Python] ContinuousBatchEngine.submit() / vLLM LLMEngine
    ▼
Dense Prefill
    │ [Python→GPU] HuggingFace SDPA forward pass — writes to Dense Slab Pool
    ▼
Block Compression Trigger
    │ [Python] KVRuntimeManager detects block is full
    │ [Python→C++] Constructs CompressJob, calls compressor.submit(job)
    │ [C++] SPSC ring buffer push — zero GIL contact
    ▼
Async Compression Worker Loop  ← ENTIRELY NATIVE
    │ [C++] Pops CompressJob from lock-free SPSC queue
    │ [C++] Checks session_alive() atomically
    │ [C++] Calls cuSOLVER gesvda (on GPU, no Python)
    │ [C++] Writes U,V to Slab Pool (GPU-direct)
    │ [C++] CAS: Compressing → CompressedResident
    ▼
Decode Step
    │ [Python] CUDA Graph replay (StaticSparseDecodeGraph)
    │   → Pre-check: diffkv_core.are_replay_safe(block_ids) ← C++ atomic read
    │ [GPU] TritonSparseDecode reads U,V from stable MetadataPool addresses
    ▼
Paging Eviction (when VRAM pressure detected)
    │ [Python] Calls pager.issue_eviction(job)
    │ [C++] CAS: CompressedResident → PagingOut
    │ [C++] cudaMemcpyAsync D2H on paging_stream (non-blocking)
    │ [C++] Records cudaEvent_t
    ▼
Paging Reload (when evicted block needed)
    │ [Python] Calls pager.issue_reload(job)
    │ [C++] cudaMemcpyAsync H2D on paging_stream (non-blocking)
    │ [C++] pager.sync_to_compute_stream(block_id, compute_stream)
    │         → compute stream WAITS ON GPU for transfer — CPU never blocks
    ▼
Poll Completions (between batch steps, not in decode path)
    │ [Python] pager.poll_completions()
    │ [C++] cudaEventQuery — advances state machine for completed transfers
    ▼
Streaming Response
    │ [Python] Token decoded, flushed to frontend
```

## What Python Still Owns
- Request routing and scheduling decisions
- Triton kernel dispatch call (kernel itself is GPU-native)
- High-level VRAM budget monitoring

## What C++ Now Owns
- All compression threading and synchronization
- All paging stream management
- All block state transitions
- Graph replay safety checks
