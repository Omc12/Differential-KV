# Phase 26 — Distributed Retrieval Routing

## The Cross-Device Routing Mechanism

Under the retrieval-aware sparse prefill engine, queries must find historical needles even if those needles live on a different GPU.

### 1. Global Anchor Registry
Because anchors are extremely lightweight (1 token per chunk), we maintain a `GlobalAnchorRegistry` on every device via `torch.distributed.all_gather`.
For a 1M token context (chunked at 512):
- Total chunks = ~1953.
- Anchor memory = 1953 tokens $\times 14$ heads $\times 64$ dim $\times 2$ bytes = **~3.5 MB**.
Every GPU can easily store the entire semantic map.

### 2. GPU-Local Hot Windows
When a query chunk (e.g., tokens 50,000 to 50,512) is evaluated on GPU 0, GPU 0 computes relevance against the 3.5 MB global anchor map locally. 
It selects the top-K chunks.

### 3. Sparse Remote Fetch
If the selected top-K chunks belong to GPU 0, they are loaded instantly.
If they belong to GPU 1:
- GPU 0 initiates an asynchronous `nccl.recv()`.
- GPU 1 initiates a corresponding `nccl.send()`.
Because the slabs are compressed to Rank 16, a chunk that would normally require ~1.8 MB of bandwidth now only requires **~180 KB**.

### 4. Locality Heuristics
To further optimize routing, the chunking mechanism can employ locality-aware assignment. When a session starts, if the query is heavily semantic to a specific domain, the ingest manager can group related chunks onto the same physical GPU, minimizing future cross-bus transfers.
