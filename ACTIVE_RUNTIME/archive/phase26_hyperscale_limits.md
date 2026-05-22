# Phase 26 — The True Hyperscale Limits

## Context
With the system now operating as a Distributed Sparse Transformer, the traditional physical limits of VRAM, PCIe bandwidth, and $O(N^2)$ compute scaling have been bypassed.

## Evaluating the Final Bottlenecks
We must evaluate what breaks when the context approaches tens of millions of tokens or thousands of concurrent users.

1. **Cross-GPU Bandwidth:** Solved. Compressing blocks to 10% of their original size and fetching only 2% of the blocks drops bandwidth requirements by over 99.8%.
2. **Anchor Lookup Latency:** Solved. Matrix multiplication of $Q_{pool}$ against a 3.5MB anchor array takes mere microseconds in SRAM.
3. **FFN FLOPs:** Bounded. Even without sparse MLP kernels, FFN scales linearly $O(N)$.
4. **Metadata Scaling:** Bounded. The global registry of anchors requires $\sim3.5$ MB per million tokens. This is trivial even for 100 million tokens (350 MB).

### The True Dominant Blocker: Distributed Allocator Fragmentation
As thousands of conversational sessions stream tokens continuously across 8 or 64 GPUs, millions of highly compressed slab tensors (of varying sizes and shapes depending on adaptive rank and sequence length) are constantly being created, fetched, cached, and evicted.

**The Limit:** GPU VRAM allocator fragmentation.
PyTorch's native CUDA caching allocator is heavily optimized for massive, contiguous, static matrices (e.g., standard dense KV caches). It is NOT designed to allocate and free millions of tiny `[16, 64]` shape low-rank $U/V$ tensors asynchronously. Over time, the memory pool will severely fragment, leading to a synthetic out-of-memory error even when 50% of the VRAM is technically free.

We must build a specialized **Page-Aligned PagedSparseAllocator** that allocates VRAM in fixed hardware pages and packs low-rank tensors into them, similar to vLLM's PagedAttention, but adapted explicitly for multi-tensor sparse slabs.
