# Phase 21 Native Readiness Matrix

This matrix validates whether the current `native_core` architecture is structurally ready for full C++/vLLM extraction.

| Requirement | Validate | Blocking Path / Explanation |
|-------------|----------|-----------------------------|
| Zero dynamic allocations in decode | **YES** | Resolved in Phase 17 via `PersistentMetadataPool`. |
| Graph-safe execution | **YES** | Resolved in Phase 17 via `StaticSparseDecodeGraph`. |
| Allocator-stable metadata | **YES** | Pre-allocated GPU buffers prevent PyTorch allocator fragmentation. |
| Replay-safe sparse decode | **YES** | The Triton kernel safely reads from fixed indices without Python dispatch. |
| Async-safe compression | **NO** | `AsyncCompressor` uses Python `threading`, which suffers from GIL contention under heavy load. Needs a true C++ background thread. |
| Paging-safe decode | **NO** | Paging blocks back *into* VRAM (reload) uses a synchronous Python call, causing latency jitter. Needs C++ CUDA stream overlap. |
| No CPU sync in hotpath | **YES** | Eliminated all `tensor.item()` or `torch.stack()` calls during the decode forward pass. |

**Verdict:** The core mathematical algorithms are 100% native-ready. The memory movement subsystems (Paging Reload and Async SVD) require C++ threading/streams to become truly async-safe.
