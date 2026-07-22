# vLLM Backend Layout

This document maps the structural integration of Differential KV into the vLLM architecture.

## Integration Path
Differential KV integrates natively as an **Attention Backend** and a **Custom KV Allocator Variant** inside vLLM.

```text
vllm/
├── attention/
│   ├── backends/
│   │   ├── dkv/                   <-- NEW: Differential KV Backend
│   │   │   ├── impl.py               <-- Maps to TritonSparseDecode
│   │   │   └── metadata.py           <-- Maps to PersistentMetadataPool
├── core/
│   ├── block_manager.py              <-- MODIFIED: Supports Rank-Aware Blocks
│   └── dkv_compression_worker.py  <-- NEW: AsyncCompressor Ray Actor
├── worker/
│   └── model_runner.py               <-- Calls dkv backend
```

## Hook Points
1. **Decode Forward Pass:** `vllm.attention.backends.dkv.impl.forward()` replaces PagedAttention for sequences utilizing compressed history.
2. **Block Eviction / Swapping:** When vLLM's `BlockSpaceManager` detects memory pressure, it flags blocks for eviction. `dkv_compression_worker` intercepts this, compressing the block via SVD before the swap occurs, drastically reducing the PCIe bandwidth required to page it out.
3. **Graph Capture:** `ModelRunner.capture_model()` must capture the `PersistentMetadataPool` state pointers natively, avoiding dynamic graph rebuilds when ranks update.
