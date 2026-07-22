# Phase 26 — Cross-GPU Sparse Execution

## The Core Frontier Question
> Can attention execution occur directly against remote compressed slabs without full replication or dense all-gather?

## Exploration of Cross-GPU Methods

### 1. Dense All-Gather (The Baseline to Avoid)
In standard Ring-Attention or Megatron Context-Parallelism, GPUs must all-gather the dense $K$ and $V$ tensors from all other GPUs. This requires massive PCIe/NVLink bandwidth. For a 1M token context, transferring dense KV caches saturates the bus entirely, halting execution.

### 2. Compressed Transfer (The DKV Method)
Instead of executing attention across the network, Differential KV operates on a **Pull-Based Sparse Fetch** model.
- The compute GPU identifies exactly which 2 chunks it needs from the remote GPU.
- It initiates a transfer of ONLY the compressed $U$ and $V$ tensors for those 2 chunks.
- The transfer size is miniscule (e.g., $180$ KB vs $1.8$ MB).
- Once the $U$ and $V$ tensors arrive in the local GPU's memory, the standard `RetrievalAwareSparsePrefill` logic operates on them EXACTLY as if they were local.

### 3. Remote Low-Rank Execution
Alternatively, we evaluated executing $S = Q \times K_{remote}^T$ using RPC. GPU 0 sends its query $Q$ to GPU 1. GPU 1 computes the attention scores against its local blocks and returns the weighted values. 
**Conclusion:** This is strictly worse. Sending $Q$ and receiving $V_{out}$ scales with $O(N)$ active query size and forces synchronization barriers. Pulling the $U$ and $V$ tensors is fundamentally superior because $U$ and $V$ size is constant $O(1)$ relative to the block, allowing purely asynchronous fetching.

## Verdict
Cross-GPU sparse execution is highly optimal using **Anchor-Directed Slab Fetch**. We do NOT all-gather. We do NOT replicate. We asynchronously fetch compressed slabs based on $O(1)$ semantic routing.
