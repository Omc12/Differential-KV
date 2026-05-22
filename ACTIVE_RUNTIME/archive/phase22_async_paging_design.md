# Phase 22 Async Paging Design

## The Fatal Bottleneck Eliminated
`tensor.to(device)` in Python blocks the **main CUDA stream**. The entire GPU decode pipeline stalls while PCIe transfers data. At 32 GB/s PCIe bandwidth, reloading a single Rank-16 slab block (e.g., 4MB) takes ~125µs. At 100 concurrent sessions with rapid eviction, this creates cascading stalls.

## The Native Async Reload Architecture

### 1. Dedicated Paging CUDA Stream
A separate, persistent CUDA stream (`paging_stream`) is created at engine startup. It is entirely disjoint from the main compute stream.

```cpp
cudaStream_t paging_stream;
cudaStreamCreateWithPriority(&paging_stream, cudaStreamNonBlocking, -1);
```

### 2. Reload Event Signaling
When a block marked `CPUResident` is needed for the next decode step:
1. The `BlockSpaceManager` detects the miss and issues `cudaMemcpyAsync()` on `paging_stream`.
2. A `cudaEvent_t reload_event` is recorded on `paging_stream`.

### 3. Compute Overlap
The main compute stream continues executing attention for **all other blocks** that are already resident. It only waits for the reload via `cudaStreamWaitEvent(compute_stream, reload_event)` immediately **before** the Triton kernel needs to access the reloaded block.

If the PCIe transfer completes before the compute stream reaches that point: **zero stall**.

### 4. Graph Replay Safety
CUDA Graphs cannot contain `cudaMemcpyAsync` calls to dynamic addresses. The graph captures only the **compute portion**. The paging reload is issued **outside** the graph, before replay. The `cudaStreamWaitEvent` call is graph-safe (static event pointer).

### 5. Cancellation Safety
If a session disconnects while a reload is in flight:
- The paging stream's transfer continues to completion (CUDA cancellation is unsafe mid-transfer).
- The completion callback marks the block `Invalid` instead of `CompressedResident`.
- The slab pool slot is immediately freed.
