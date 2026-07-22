# Phase 22 Slab Allocator Design

## Overview
The Adaptive Rank Selector (dynamic rank 4–64) is officially deprecated. It is replaced by a **Fixed-Bucket Slab Allocator** with exactly three slab tiers.

## The Three Slabs

| Slab | Rank | Block Memory Size | Use Case |
|------|------|------------------|----------|
| Slab-8 | 8 | `2 * rank * head_dim * heads * sizeof(fp16)` | Highly compressible historical context (flat, repetitive) |
| Slab-16 | 16 | Same formula, rank=16 | General long-context history (default) |
| Slab-32 | 32 | Same formula, rank=32 | Complex, high-entropy context (recent but evicted) |

## Allocation Rules
1. **At compression time**, the `DKVCompressionWorker` executes truncated SVD, retains the top-K singular values by a fixed threshold, then **rounds up** to the nearest slab tier.
2. **All blocks in a slab are identical in size.** No heterogeneous allocation within a slab.
3. **Pre-allocated Slab Pools:** At engine startup, three separate GPU memory pools are allocated, each sized for a fixed number of blocks.

## Code Paths Touched
- `native_core/compression/async_compressor.py`: `rank` is now selected as `[8, 16, 32][bisect_right([8, 16], computed_rank)]` instead of the previous dynamic calculation.
- `native_core/metadata_pool/metadata_pool.py`: `U_pool` and `V_pool` are now split into three fixed-size tensors per slab tier.
- `native_core/paging/paged_kv_store.py`: Each slab has its own independent LRU eviction list. Eviction never crosses slab boundaries.

## vLLM Compatibility
vLLM's `BlockSpaceManager` can manage three separate physical memory pools (one per slab). Each slab pool is internally homogeneous — satisfying the core vLLM assumption.
