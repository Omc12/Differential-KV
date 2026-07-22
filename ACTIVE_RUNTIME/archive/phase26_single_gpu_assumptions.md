# Phase 26 — Single-GPU Assumptions Trace

## Audit of Memory Ownership

Currently, Differential KV is fundamentally bound to a single GPU. Our audit of the core engine (`KVRuntimeManager`, `StreamingSparseIngestManager`, and `RetrievalAwareSparsePrefill`) reveals several hardcoded locality assumptions:

### 1. Pointer Locality in `StreamingKVBlock`
Every `StreamingKVBlock` holds `U`, `V`, and `anchor_kv` as direct `torch.Tensor` objects residing on `self.device`. There is no abstraction for a "RemoteBlock" or an RPC handle. If the memory footprint of these blocks exceeds the VRAM of `cuda:0`, the system will OOM (or spill to slow CPU RAM via `PagedKVStore`).

### 2. Global Anchor Pooling
The `RetrievalAwareSparsePrefill` engine routes attention by executing:
```python
keys_anchors = torch.stack([b.anchor_kv for b in blocks])
```
This forces all semantic anchors for a session to be materialized into a single dense tensor on the local GPU for the relevance matmul. If blocks were distributed across GPUs, this operation would fail or require a dense all-gather.

### 3. Sparse Attention Kernels
The `TritonDKV` and PyTorch SDPA paths expect $Q$, $K$, and $V$ (or $U$ and $V$) to reside in local SRAM/HBM. They cannot read pointers from a neighboring GPU over NVLink unless specifically mapped via Unified Memory (which is incredibly slow for unstructured access) or manually exchanged.

### 4. Graph Capture Boundaries
CUDA Graph replay for decoding assumes that memory addresses for the compressed slabs are static and local. Cross-GPU memory transfers cannot be easily captured in a static local graph without explicit communication synchronization primitives.

## Conclusion
The runtime successfully limits memory footprint to $O(blocks)$, but all of those blocks MUST fit on one GPU. To hyperscale to millions of tokens, we must break the assumption that `StreamingKVBlock.U` is a local tensor.
