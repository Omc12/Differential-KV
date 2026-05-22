# Phase 21 Python Hotpath Audit

This audit identifies every remaining Python-controlled operation that executes per token, layer, or block in the current runtime, classifying its readiness for native C++ extraction.

## Operations Audit

| Operation | Frequency | Classification | Justification |
|-----------|-----------|----------------|---------------|
| **Metadata Updates** | Per Token | *Native-Candidate* | Currently updates Python dicts/lists. Easily migratable to C++ structs updating GPU pointers directly. |
| **Block Residency Updates** | Per Block | *Native-Candidate* | Tracking LRU status is lightweight in Python, but natively moving it to C++ (via vLLM's BlockSpaceManager) avoids GIL contention. |
| **Compression Scheduling** | Per Block | *Native-Candidate* | Triggering the compression queue from Python works, but the SVD worker *must* be C++ to release the GIL completely. |
| **Rank Metadata Propagation** | Per Block | *Native-Candidate* | Assigning rank 8, 16, or 32 is a simple scalar operation. C++ extraction is trivial. |
| **Paging Reload Logic** | Occasional | **Fatal Python Bottleneck** | `tensor.to(device)` in Python blocks the main thread during execution. Must be moved to native CUDA streams. |
| **Graph Invalidation Logic** | Per Session | **Fatal Python Bottleneck** | Checking if batch topologies changed requires a Python conditional tree. In vLLM, this is solved by padded C++ graph capturing, completely bypassing Python logic. |

**Conclusion:** The logic inside the Differential KV core is fundamentally sound and mathematically verified. The Python bottlenecks that remain are entirely due to the limitations of PyTorch's eager orchestration, all of which are cleanly resolved by standard C++ integration techniques.
