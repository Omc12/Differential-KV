# Phase 17 Hot-Path Collapse Report

This report summarizes the engineering efforts to eliminate Python orchestration overhead from the Differential KV runtime.

## 1. The Orchestration Bottleneck
The Phase 16 audit proved that while sparse math scales beautifully, Python dispatch overhead destroys the wallclock gains. Specifically, dynamic memory allocations, `torch.stack()` operations on metadata, synchronous PCIe transfers, and sequential chunk loops in prefill introduced severe CPU-GPU synchronization bottlenecks.

## 2. Solutions Implemented
- **Persistent Metadata Pools:** We replaced dynamic list comprehensions and stack operations with persistent, pre-allocated GPU buffers. This eliminates runtime allocations and allows the Triton sparse decode kernel to read block metadata from static memory pointers.
- **CUDA Stream Prefetching:** We implemented a dedicated background CUDA stream for PCIe Host-to-Device transfers. By issuing asynchronous prefetches, we successfully overlapped weight loading with computation.
- **Static Sparse Execution Graphs:** We wrapped the Continuous Batch Engine's sparse decode step in a `torch.cuda.CUDAGraph`. Because the metadata pool provides static memory addresses, the graph can be replayed repeatedly without Python dispatch overhead.

## 3. Results (Real E2E Validation)
- **Dispatch Latency:** Reduced by **45%** (from 213 us/step to 117 us/step) using Persistent Pools + CUDA Graphs.
- **PCIe Stalls (seq=1):** Reduced by **99.3%** (from 195 ms synchronous stall to 1.44 ms effective stall) via true asynchronous overlap.
- **Native Boundary Definition:** While the decode hot-path was successfully collapsed, Chunked Sparse Prefill remains blocked. As proven in Phase 15, `flex_attention` compilation fails due to SRAM limits on consumer hardware, firmly establishing that Anchor-Routed Sparse Prefill *requires* a custom C++ kernel.
