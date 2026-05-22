# Phase 21 Memory Ownership Trace

This trace defines the exact lifecycle and ownership of physical memory blocks throughout the Differential KV pipeline, explicitly identifying integration hazards.

## The Lifecycle Trace

1. **Request Ingestion (vLLM)**
   - *Ownership:* vLLM `BlockSpaceManager`
   - *Action:* Allocates logical tokens and reserves physical Dense blocks.

2. **Block Allocation & Prefill (Dense)**
   - *Ownership:* vLLM PagedAttention
   - *Action:* Writes uncompressed $[K, V]$ matrices into the Dense physical block pool.

3. **Compression Trigger**
   - *Ownership:* Hand-off from vLLM to Differential KV Worker.
   - *Hazard:* **Synchronization Hazard.** The block must be marked as "Compressing" to prevent vLLM from evicting it or modifying it while SVD runs.

4. **Async Compression**
   - *Ownership:* Differential KV AsyncCompressor (C++ Thread)
   - *Action:* Reads Dense Block $\rightarrow$ Computes SVD $\rightarrow$ Writes $U, V$ to Compressed Pool.

5. **Decode (Sparse)**
   - *Ownership:* Differential KV TritonSparseDecode
   - *Action:* Reads $U, V$ from the Compressed Pool.
   - *Hazard:* **Tensor Lifetime Hazard.** If a user suddenly disconnects, vLLM must signal the metadata pool to free the $U, V$ pointers before freeing the physical Dense block tracking them.

6. **Paging (Eviction/Reload)**
   - *Ownership:* vLLM Pager
   - *Action:* Moves the Compressed $U, V$ block to CPU RAM.
   - *Hazard:* **Allocator Hazard.** vLLM standard paging expects homogeneous block sizes. Compressed blocks have varying ranks (Rank 8, 16, 32). The vLLM pager must be modified to accept heterogeneously sized block transfers.

## Conclusion
Memory ownership is logically sound, but integrating it into vLLM requires extending vLLM's `BlockSpaceManager` to understand two critical concepts: **Heterogeneous Block Sizes** (for adaptive rank) and **Compression States** (to prevent race conditions during async SVD).
