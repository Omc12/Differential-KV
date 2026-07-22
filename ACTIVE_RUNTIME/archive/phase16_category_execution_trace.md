# Phase 16 Category Execution Trace

This document maps the exact, physical execution traces for all ACTIVE_EXECUTING and PARTIAL systems within the current Differential KV runtime.

## Trace 1: The Sparse KV Runtime Core (Categories 1)

This is the only fully integrated and end-to-end active path in the repository.

**1. Inference Request Entry:**
`run_real_sparse_stress_test.py` or `batch_engine.py` receives a text prompt.

**2. Prefill Phase (Dense):**
- Hits `hf_dkv_wrapper.py:DKVAttention.forward()`
- Executes standard Dense SDPA for initial token ingestion.
- Emits uncompressed Key/Value tensors.

**3. Memory Ingestion & Adaptive Compression:**
- `KVRuntimeManager.append_tokens()` is called.
- Appends to the Dense Recency Window (typically 128 tokens).
- Once a 64-token block falls out of the dense window, it triggers `_compress_block_sync()`.
- **Adaptive Rank Selection**: Calculates the 95% variance bucket (e.g., rank 8, 16, or 32).
- **Async Compression**: Offloads the `torch.linalg.svd()` work to a background thread (`AsyncCompressor`), preventing the GPU from blocking.

**4. Paging & Residency:**
- `PagedKVStore` continually monitors the VRAM budget (e.g., 2.0 GB).
- If VRAM is exceeded, compressed blocks (LRU) are evicted to pinned CPU RAM via `.to(cpu)`.

**5. Continuous Decode Phase:**
- `batch_engine.py:ContinuousBatchEngine.step()` loops over active sessions.
- `hf_dkv_wrapper.py` routes decode tokens to `TritonDKV.forward()`.
- **Triton Sparse Decode Kernel**: Executes $O(1)$ block-sparse attention directly from the compressed SVD matrices ($U \Sigma V^T$) WITHOUT reconstructing the dense KV cache in VRAM.

---

## Trace 2: Disconnected Execution Paths (Categories 2, 3, 4, 7)

These traces exist in isolated test harnesses, but have not been merged into the `batch_engine.py` hot path due to orchestration limits.

**Sparse Transformer Execution (Category 2)**
- `test_phase11_fused.py` → `TritonSparseMLP.forward()` → Fused Triton Kernel for Block-Sparse Gate/Up projections.

**Hierarchical Residency (Category 3)**
- `test_phase12_tiered_ffn.py` → `TieredFFNWeights.forward()` → Detects cache miss → Synchronous PCIe transfer of FFN block from RAM to VRAM → Dense PyTorch F.linear.

**Sparse Prefill & Anchors (Category 4 & 7)**
- `test_phase15_fused_prefill.py` → `FusedSparsePrefill.execute()` → Calculates Global K/Q Centroids (Anchors) → Builds boolean retrieval mask → Attempts `flex_attention` compile → **(FAILS due to SRAM exhaustion)**.
