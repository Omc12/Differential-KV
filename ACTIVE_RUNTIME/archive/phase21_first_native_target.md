# Phase 21 First Native Target

## The Single Highest-Leverage Native Extraction Point

**Target: `AsyncCompressor` (C++ Background Thread)**

## The Justification

### Why Not TritonSparseDecode?
The Triton kernel is already effectively native. Triton compiles directly to PTX/SASS and executes on the GPU without Python involvement once launched. It is already the fastest component in the system.

### Why Not MetadataPool?
The `PersistentMetadataPool` already operates on pre-allocated GPU tensors with stable addresses. Python only writes to it during block transitions, not during decode. It is already orchestration-safe.

### Why the AsyncCompressor?
The `AsyncCompressor` is the **single most dangerous Python component** in the entire serving path:

1. **GIL Contention Under Load:** Background Python threads call `torch.linalg.svd`. Even though PyTorch releases the GIL during the underlying LAPACK computation, the Python thread still must acquire the GIL to pull items from the queue, update internal dictionaries, and trigger the metadata pool write. Under heavy serving load (100+ concurrent sessions), this GIL contention measurably stalls the main generation thread.

2. **Queue Backpressure:** Python `queue.Queue` introduces hidden latency. When the queue fills, it creates implicit Python-level blocking—invisible in profilers but catastrophic in latency distributions (P99 spikes).

3. **Maximum Systemic Impact:** Extracting this single component to C++ directly unlocks the remaining two "NO" entries in the `native_readiness_matrix.md`. A C++ SVD worker running on a dedicated CPU thread with a lockfree queue makes both Async Compression and Paging Reload natively safe without touching the GPU kernels at all.

## The Native Implementation Plan
1. Write a `C++` extension (`dkv_compressor.cpp`) that spawns a dedicated OS thread.
2. Use `std::queue` and `std::mutex` (or a lockfree SPSC queue) to receive block compression requests.
3. Call `cuBLAS` or `cuSOLVER` directly for batched SVD without Python overhead.
4. Write the result directly to the pre-allocated `MetadataPool` GPU tensor via a raw CUDA pointer.

This is the extraction target for Phase 22.
