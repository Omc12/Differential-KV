# Phase 20 Post-Python Bottlenecks

With Python orchestration loops collapsed via CUDA Graphs and Metadata Pools, we have exposed the true physical boundaries of the Differential KV architecture.

## 1. The True Bottleneck: Memory Bandwidth vs Compute
Standard dense attention (PagedAttention) is fundamentally memory-bandwidth bound. Differential KV successfully reduces the memory footprint of the KV cache by compressing blocks from $[N, d]$ into $U [N, r]$ and $V [r, d]$.
- **The Catch:** While the *size* of the memory is smaller, executing the sparse decode requires reading $U$, reading $V$, and performing two matrix multiplications instead of one. 
- **The Reality:** For small batch sizes (e.g., `< 16`), the time saved by loading less memory from HBM is offset by the additional FLOPs required to decompress the block on the fly. The true bottleneck is balancing the memory bandwidth savings against the arithmetic intensity of the SVD decompression step inside the Triton kernel.

## 2. Secondary Bottleneck: Compression Queue Pressure
Under a real serving load, a massive prefill request (e.g., 100K tokens) generates 1,500+ blocks instantly. The background `AsyncCompressor` must execute `torch.linalg.svd` on all of them. If the queue backs up, the Dense Recency Window overflows, and the engine must page uncompressed blocks to RAM, temporarily destroying the VRAM savings.

## 3. Paging Jitter
When a sequence is evicted to CPU RAM, reloading it for a resumed generation step is slow. Compressing the block helps (smaller transfer size), but PCIe bandwidth remains a hard physical limit.
