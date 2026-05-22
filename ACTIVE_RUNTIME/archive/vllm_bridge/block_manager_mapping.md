# Block Manager Mapping

This defines the exact data structure mapping between Differential KV and vLLM's `BlockSpaceManager`.

## Ownership
vLLM's `BlockSpaceManager` retains absolute ownership over logical-to-physical block mapping, ref counts, and the physical memory pool (both GPU and CPU).

## The Extension: Rank-Aware Physical Blocks
Standard vLLM allocates physical blocks as fixed-size dense tensors: `[num_blocks, block_size, num_heads, head_size]`.

**Differential KV introduces a heterogeneous block pool:**
1. **Dense Pool:** Identical to standard vLLM (used for the Recency Window).
2. **Compressed Pool:** A parallel memory pool storing $U$ and $V$ matrices.

When a block ages out of the Recency Window, the `Compression Worker` reads from the Dense Pool, runs SVD, writes to the Compressed Pool, and updates the vLLM `BlockTable`.

## The Handshake
```python
# vLLM BlockTable entry extension
class PhysicalTokenBlock:
    ...
    is_compressed: bool = False
    compressed_rank: int = 0
    compressed_u_ptr: int = -1
    compressed_v_ptr: int = -1
```
vLLM's scheduler remains completely unaware of the internal math. It simply passes the extended `BlockTable` to the `diffkv` attention backend.
