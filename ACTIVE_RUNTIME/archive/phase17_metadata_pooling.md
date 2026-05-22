# Phase 17 Metadata Pooling

## The Problem
During the continuous batching sparse decode step, the engine originally gathered compressed $U$ and $V$ matrices for each active block by iterating through python lists and calling `torch.stack()`. This forced a CPU-GPU synchronization, caused memory fragmentation, and introduced severe python overhead (upwards of 200us per step).

## The Solution
We implemented `PersistentMetadataPool`.
- **Pre-allocation:** $U$, $V$, and `session_block_indices` are pre-allocated as static, fixed-size tensors on the GPU during engine initialization.
- **In-place updates:** When a new block is compressed, the background thread writes the resulting matrices directly into the pre-allocated pool using `copy_(non_blocking=True)`.
- **Zero-allocation dispatch:** The Triton sparse decode kernel now accepts the static pool tensors and the session's pre-allocated integer index array.

## Result
Metadata dispatch latency dropped from **213.58 us** down to **117.39 us** per step, effectively cutting the Python orchestration tax for batch assembly in half.
