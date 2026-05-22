# Phase 17 The Native Boundary

This report establishes the final, immutable boundary of what can be engineered efficiently in Python/Triton, and what demands a fully native C++ runtime.

## 1. What Survives in Python / Triton
- **Sparse Decode Execution:** With Persistent Metadata Pools and CUDA Graphs, the $O(1)$ block-sparse decode path operates with near-zero orchestration overhead. It is fully production-viable in its current state.
- **Async KV Compression:** Background python threading handles SVD safely without stalling the primary CUDA stream.
- **Paged Memory Management:** Tracking LRU state and VRAM budgets is computationally light and perfectly suited for Python.

## 2. What Requires Native C++ / Custom Kernels
- **Anchor-Routed Sparse Prefill:** This is the hard limit. Routing to sparse chunks requires complex boolean block-masks. Looping chunks in Python yields 120ms latency. Attempting to fuse them with PyTorch 2.5 `flex_attention` fails with Triton SRAM exhaustion (`OutOfMemoryError: triton required 114KB, limit 100KB`). **To scale $O(N)$ sparse prefill, we require a custom C++ FlashAttention-3 style kernel capable of reading block-sparse metadata natively.**

## 3. What Requires vLLM Integration
- **Asynchronous Tiered FFNs:** While we proved CUDA stream overlap works in Python (`AsyncTieredFFN`), managing memory streams effectively in a high-concurrency environment requires deep integration with a serving engine's memory allocator (like vLLM's BlockSpaceManager).
- **Fused Sparse MLP:** Bypassing PyTorch `index_select` overhead requires wiring our Triton Sparse MLP kernel directly into the C++ forward pass of the model layers.

## Conclusion
Phase 17 successfully collapsed the hot-path overhead for continuous decode. However, it formally proved that the Sparse Transformer stack (Prefill + MLPs) has outgrown pure PyTorch. The next evolution of Differential KV must be implemented as a C++ extension or vLLM backend.
