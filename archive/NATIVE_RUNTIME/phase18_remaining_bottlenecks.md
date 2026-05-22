# Phase 18 Remaining Bottlenecks

Despite successfully extracting a minimal, stable native runtime and eliminating hot-path Python loops, several absolute boundaries remain that block enterprise-level production scaling. 

## 1. CUDA Graph Invalidation Jitter
- **The Issue:** `StaticSparseDecodeGraph` provides incredible speedups by bypassing Python dispatch. However, CUDA graphs demand a fixed topology (batch size and tensor addresses). When a session finishes and is removed from the batch, or a new session joins, the batch size changes.
- **The Bottleneck:** Re-capturing the CUDA graph takes 10-20ms. In a high-churn serving environment, frequent re-captures cause unacceptable latency jitter across all active sessions. 

## 2. The Python GIL (Global Interpreter Lock)
- **The Issue:** `AsyncCompressor` runs on a background Python thread. 
- **The Bottleneck:** Even with `torch.linalg.svd` releasing the GIL during the actual math execution, the Python thread still must acquire the GIL to update dictionaries, handle queues, and manage `PagedKVStore` state. Under extreme load (many concurrent users), GIL contention between the decode loop and the background compression thread causes microscopic but accumulating stalls.

## 3. Allocator Fragmentation in VRAM
- **The Issue:** PyTorch’s caching allocator expects short-lived activation tensors. Differential KV holds long-lived persistent metadata pools and compressed blocks of varying ranks (Rank 4, 16, 32).
- **The Bottleneck:** Mixing long-lived, variably-sized persistent buffers with short-lived eager allocations (like logits or temporary intermediate states) eventually causes severe memory fragmentation. PyTorch will eventually trigger an expensive `cudaMalloc` or `empty_cache()`, causing massive latency spikes. 

## Conclusion
The engineering inside `NATIVE_RUNTIME` is mathematically correct and fast, but it is living on borrowed time inside the PyTorch eager ecosystem. To achieve 100% stability, memory management and graph execution must descend to native C++.
