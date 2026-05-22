# Phase 16 Synergy Report

This document analyzes how the verified components of the Differential KV runtime interact with one another, specifically determining if their benefits compound or cancel each other out during live execution.

## 1. High-Synergy Combinations

**Async KV Compression + Paged KV Store**
- **Synergy:** Excellent. By offloading SVD to a background thread, the dense fallback system safely serves tokens from the uncompressed window. Once compression completes, the `PagedKVStore` immediately recognizes the block's reduced footprint and safely evicts/pages it if VRAM pressure requires. This completely decouples compression latency from decoding throughput.

**Adaptive Rank Selection + Continuous Batching**
- **Synergy:** High. `AdaptiveRankSelector` allocates higher rank (e.g., 32) to complex semantic blocks and rank 4 to simple uniform blocks. The Continuous Batch Engine processes sequences of varying lengths simultaneously; the adaptive rank ensures that even with highly mixed batch states, the Triton sparse decode kernel maintains high throughput because it only computes over the strict information capacity required per block.

**Global Memory Anchors + Chunked Prefill**
- **Synergy:** Mathematically perfect, practically broken. Anchors (K/Q centroids) allow chunked prefill to route to distant historical blocks without full $O(N^2)$ memory materialization. The mathematical synergy is undeniable (restoring cosine similarity from 0.01 to 0.99), but Python orchestration ruins the physical execution synergy.

## 2. Anti-Synergy / Negative Interference

**Hierarchical FFN Residency + Continuous Batching (seq=1)**
- **Anti-Synergy:** Severe. At `seq=1`, the Continuous Batch Engine requires ultra-low latency execution to maintain human-readable TPS. When the `TieredFFNWeights` predicts incorrectly, it must fetch a weight block over PCIe. This synchronous stall halts the entire batch matrix multiplication. Continuous batching amplifies this penalty, as one stalled sequence forces all other concurrent sequences to wait.

**Python Chunked Prefill + Native Sparse Decode**
- **Anti-Synergy:** High. The runtime aims to reduce TTFT (Time To First Token). While the Triton Sparse Decode is incredibly fast for generation, executing Chunked Sparse Prefill via a Python `for` loop (or via a failed `torch.compile` FlexAttention graph) causes massive TTFT regression (120ms vs 13ms). A blazing fast decode engine is useless if the user waits 10x longer for the first token.

## 3. Conclusions
True synergy in a sparse runtime is gated entirely by orchestration. Components that operate asynchronously or run in a single fused Triton kernel synergize beautifully. Components that require CPU-GPU synchronization, dynamic PCIe transfers during the forward pass, or Python `for` loops create catastrophic negative interference.
