# Phase 27 — System Verification Report
## Each claimed system: proven to execute, proven not to, or proven disconnected

---

## Verification Method

For each system:
1. Located the implementation file
2. Traced every import that reaches it from `launch_real_serving.py`
3. Identified the exact call site in the live path OR the exact break in the chain

---

## 1. StreamingSparseIngestManager

**Verdict: EXECUTES**

- File: `ACTIVE_RUNTIME/native_core/streaming_sparse_ingest.py` (319 lines)
- Import chain: `launch_real_serving.py` → `DKVHFWrapper` → `KVRuntimeManager.__init__` → `StreamingSparseIngestManager()` instantiated
- Call site: `kv_manager.ingest_streaming(sid, layer_idx, k, v)` — called in `dkv_attention.py` L86 (decode) and L159 (prefill)
- Behavior: micro-block accumulation → compression trigger on fill → delegate to AsyncCompressor
- **CONFIRMED EXECUTING**

---

## 2. AsyncCompressor

**Verdict: EXECUTES (background threads)**

- File: `ACTIVE_RUNTIME/native_core/compression/async_compressor.py` (173 lines)
- Import chain: `KVRuntimeManager.__init__` → `AsyncCompressor(compress_fn=self._compress_block_sync)` → `.start()` called immediately
- Thread start: `threading.Thread(target=self._worker_loop, daemon=True).start()` × 2 workers
- Call site: `AsyncCompressor.submit(block, k, v)` → `queue.put_nowait()` → worker wakes → `torch.linalg.svd` → `block.U/V = lr_delta` → `block.active_k = None`
- Backpressure: `queue.Full` → synchronous fallback inline
- **CONFIRMED EXECUTING** (background threads active as long as server runs)

---

## 3. Adaptive Rank

**Verdict: EXECUTES (when RESEARCH_PROTOTYPES/compression/adaptive.py is importable)**

- File: `RESEARCH_PROTOTYPES/compression/adaptive.py`
- Import chain: `kv_runtime_manager.py` L36-47 — dynamic import via `importlib.util.spec_from_file_location`
- Guard: `try/except` — if import fails, falls back to fixed `self.rank = 8`
- Call site: `self._rank_selector.select_rank(deltas)` in `_compress_block_sync()`
- **EXECUTES IF FILE LOADABLE — silently falls back otherwise**

---

## 4. Shared Basis

**Verdict: DISCONNECTED**

- File: `RESEARCH_PROTOTYPES/compression/shared_basis.py` (4 KB)
- No import chain from `launch_real_serving.py` or any file in `ACTIVE_RUNTIME/serving/`
- Not referenced in `kv_runtime_manager.py`, `dkv_attention.py`, or `batch_engine.py`
- Code exists and may be correct; it is simply never called
- **CONFIRMED DISCONNECTED**

---

## 5. Sparse Triton Decode (_fused_sparse_decode_kernel)

**Verdict: CODE EXISTS, NEVER DISPATCHES**

- File: `ACTIVE_RUNTIME/native_core/sparse_decode/triton_sparse_attn.py` (234 lines)
- The Triton kernel `_fused_sparse_decode_kernel` with `@triton.jit` is real and complete
- Wrapper `native_triton_sparse_attn_decode()` requires argument `pool: NativeBlockPool`
- `NativeBlockPool` requires `dkv_core` Python extension to be compiled (bindings.cpp → .pyd/.so)
- `dkv_core/` directory: CMakeLists.txt present, source files present, **zero compiled artifacts**
- The actual decode path calls `batched_sparse_attn_decode()` (Phase 8 PyTorch batched einsum) — not this kernel
- **CONFIRMED: TRITON FUSED DECODE KERNEL NEVER DISPATCHES IN PRODUCTION**

---

## 6. FlashAttention / SDPA Prefill

**Verdict: EXECUTES**

- Call site: `runtime/dkv_attention.py` L224-248
- `F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=(q_len > 1))`
- PyTorch 2.x SDPA dispatcher: selects FlashAttention2 if available, else efficient attention, else math
- This is called on every prefill token batch regardless of context length
- **CONFIRMED EXECUTING**

---

## 7. Chunked Sparse Prefill / Anchor Routing

**Verdict: PARTIAL PROTOTYPE — conditional import with live risk**

- Call site: `runtime/dkv_attention.py` L232-240
- Condition: `elif q_len > 1024 and key_len == q_len:` — only for long same-length prefill
- Import: `from research.sparse_prefill_anchors import RetrievalAwareSparsePrefill`
- The `research/` module directory in `ACTIVE_RUNTIME/` was not directly audited for this file
- If the import fails at runtime: `AttributeError` or `ModuleNotFoundError` will crash the prefill for long sequences
- For q_len ≤ 1024 (typical chat): this branch is never taken; serving works fine
- **STATUS: CONDITIONALLY PARTIAL — does not affect normal chat serving**

---

## 8. Sparse FFN Execution

**Verdict: NOT IN SERVING PATH**

- Files: `RESEARCH_PROTOTYPES/sparse_mlp_router.py`, `block_sparse_ffn_executor.py`
- Neither is imported from any file in `ACTIVE_RUNTIME/serving/` or `ACTIVE_RUNTIME/runtime/`
- FFN layers execute via standard Qwen2 MLP (not patched) — full dense FFN on every token
- **CONFIRMED: SPARSE FFN NEVER EXECUTES**

---

## 9. Tiered FFN Paging

**Verdict: ARCHITECTURE ONLY**

- Referenced in Phase 12 markdown documents in `ACTIVE_RUNTIME/`
- `native_core/residency/` directory: **EMPTY**
- No tiered FFN paging code in the live serving path
- **CONFIRMED: DOES NOT EXIST AS EXECUTABLE CODE**

---

## 10. CUDA Graph Replay (StaticSparseDecodeGraph)

**Verdict: IMPLEMENTED BUT DISCONNECTED**

- File: `ACTIVE_RUNTIME/native_core/graph_runtime/static_decode_graph.py` (58 lines)
- Code is real: `torch.cuda.CUDAGraph()`, `with torch.cuda.graph(self.graph):`, `self.graph.replay()`
- Never instantiated in `launch_real_serving.py`, `batch_engine.py`, or `dkv_attention.py`
- Depends on a `decode_fn` and `NativeBlockPool` which are not wired in
- **CONFIRMED: CUDA GRAPH NEVER CAPTURED OR REPLAYED**

---

## 11. Distributed Slab Logic

**Verdict: ARCHITECTURE ONLY — stubs with commented-out real logic**

- `RESEARCH_PROTOTYPES/distributed/` — 45 files, all <3 KB
- `distributed_kv_fabric.py`: routing logic with no actual network calls
- `distributed_sparse_cache.py`: in-process dict only, no inter-process communication
- All rely on Python dicts for "distributed" state — not actual distributed memory
- **CONFIRMED: NO DISTRIBUTED SLAB LOGIC EXECUTES**

---

## 12. Cross-GPU Sparse Fetch

**Verdict: ARCHITECTURE ONLY**

- `RESEARCH_PROTOTYPES/distributed/cross_gpu_rehydration_engine.py`: stub
- No `cudaMemcpyPeer`, no `torch.distributed.send/recv`, no RDMA
- Root `distributed/` directory: **EMPTY** (only `__pycache__`)
- **CONFIRMED: CROSS-GPU FETCH NEVER HAPPENS**

---

## 13. Last-Token Logits Patch

**Verdict: EXECUTES**

- File: `ACTIVE_RUNTIME/runtime/dkv_attention.py` L267-275
- Applied in `apply_dkv_attention_patch()` called at wrapper init:
  ```python
  def last_token_lm_head_forward(hidden_states):
      if hidden_states.shape[1] > 1:
          return original_lm_head_forward(hidden_states[:, -1:, :])
      return original_lm_head_forward(hidden_states)
  model.lm_head.forward = last_token_lm_head_forward
  ```
- Effect: projects `[B, S, D]` → `[B, 1, D]` through vocab head for prefill — eliminates `S × vocab_size` logit matrix
- **CONFIRMED EXECUTING** on every prefill call

---

## Overall Verdict Table

| System | Executes? | Dispatches GPU? |
|---|---|---|
| StreamingSparseIngestManager | YES | No (Python tensor ops) |
| AsyncCompressor | YES (background thread) | YES (SVD on GPU) |
| Adaptive Rank | YES (if importable) | YES (within SVD) |
| Shared Basis | NO — disconnected | NO |
| Triton Fused Sparse Decode | NO — needs compiled C++ | N/A |
| FlashAttention/SDPA Prefill | YES | YES (CUDA Flash kernel) |
| Chunked Sparse Prefill | CONDITIONAL (q_len > 1024) | UNKNOWN |
| Sparse FFN | NO | NO |
| Tiered FFN Paging | NO | NO |
| CUDA Graph Replay | NO — disconnected | NO |
| Distributed Slab Logic | NO — stubs | NO |
| Cross-GPU Sparse Fetch | NO — stubs | NO |
| Last-Token Logits Patch | YES | YES (cuBLAS lm_head) |
