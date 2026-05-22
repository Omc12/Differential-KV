# Phase 26 — Distributed Slab Ownership Design

## Core Architecture

To scale beyond a single device, we introduce **Multi-GPU Slab Ownership**. This completely decouples logical session context from physical GPU residency.

### 1. The Chunk Sharding Strategy
We use **Context-Parallel Round-Robin Sharding**.
If a cluster has 4 GPUs:
- Chunks 0, 4, 8 → reside on GPU 0
- Chunks 1, 5, 9 → reside on GPU 1
- Chunks 2, 6, 10 → reside on GPU 2
- Chunks 3, 7, 11 → reside on GPU 3

### 2. The Distributed Ownership Table
The `KVRuntimeManager` is elevated to a `DistributedKVManager`. It maintains a lightweight metadata table:
```json
{
  "block_idx": 45,
  "node_id": "gpu_2",
  "compression_state": "COMPRESSED",
  "anchor_vector": [0.12, -0.45, ...] // Kept locally on all GPUs!
}
```

### 3. Globally Replicated Anchors
While the massive $U$ and $V$ tensors (which represent 90% of the data) are sharded across the GPUs, the tiny `anchor_kv` vectors (1 token per 512 tokens) are **all-gathered and replicated** across all GPUs.
This means every GPU has a full semantic map of the entire 1,000,000 token document, consuming only a few megabytes.

### 4. Dynamic Execution
When GPU 0 processes a new query:
1. It queries its *local* replica of the global anchor map.
2. It identifies that Block 45 is highly relevant.
3. It checks the ownership table: Block 45 belongs to GPU 2.
4. GPU 0 issues an asynchronous Point-to-Point (P2P) NVLink/NCCL request to GPU 2: `fetch_slab(45)`.
5. GPU 2 sends the highly compressed $U$ and $V$ tensors over the bus.

## Why this works
Standard models cannot do this because sending a dense 512-token block of KV cache over NVLink is extremely bandwidth-heavy. Because Differential KV compresses blocks by a factor of 10-16x, the cross-GPU communication is slashed by the same 10-16x margin.
