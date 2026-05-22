# Phase 27 — True Runtime Roadmap
## Only real engineering steps. No fantasy milestones.

---

## Basis

This roadmap is derived exclusively from the Phase 27 code audit findings.
Every item maps to a specific gap between what currently executes and what would produce
a measurable improvement.

---

## Already Working (Do Not Touch)

These systems execute correctly in the live serving path.
No action needed unless a specific bug is discovered.

| System | Evidence |
|---|---|
| ContinuousBatchEngine (prefill + decode loop) | batch_engine.py — real asyncio batching |
| DiffKV Attention Patch (Qwen2 monkey-patch) | diffkv_attention.py — all layers patched |
| StreamingSparseIngestManager (micro-block ingest) | streaming_sparse_ingest.py — called every token |
| AsyncCompressor (background SVD pipeline) | async_compressor.py — 2 daemon threads |
| AdaptiveRankSelector | adaptive.py — variance-based rank selection |
| PagedKVStore (GPU/CPU tiering) | paged_kv_store.py — background eviction thread |
| ReconstructionCache (LRU for U@V GEMMs) | recon_cache.py — called on every get_kv() |
| Batched Sparse Attention Decode (Phase 8) | batched_sparse_attn.py — 3 batched einsum ops |
| SDPA/FlashAttention Prefill (Phase 24.9) | F.scaled_dot_product_attention() in prefill branch |
| LM Head Last-Token Patch (Phase 25) | lm_head.forward replaced in apply_diffkv_attention_patch() |
| OpenAI API Gateway + session management | openai_compatible_api_gateway.py + session manager |
| TritonDiffKV LowRank Reconstruction | triton_diffkv.py — @triton.jit with PyTorch fallback |

---

## Needs Wiring (Code Exists, Not Connected)

These require integration work only — no new code.

### W1. Guard the RetrievalAwareSparsePrefill Import
**Priority: HIGH (blocks 25K+ prompt serving)**

- Location: `ACTIVE_RUNTIME/runtime/diffkv_attention.py` L232-240
- Problem: `from research.sparse_prefill_anchors import RetrievalAwareSparsePrefill` has no try/except guard
- If the module is missing, ALL prefill with q_len > 1024 crashes with ModuleNotFoundError
- Fix: Wrap import in try/except; fall through to SDPA if unavailable

```python
# Replace lines 232-240 in diffkv_attention.py:
try:
    from research.sparse_prefill_anchors import RetrievalAwareSparsePrefill
    if not hasattr(layer.self_attn, "sparse_prefill_engine"):
        layer.self_attn.sparse_prefill_engine = RetrievalAwareSparsePrefill(
            sink_tokens=64, chunk_size=512, local_window_chunks=1, top_k_retrieval_chunks=2
        )
    attn_output = layer.self_attn.sparse_prefill_engine.execute_sparse_attention(
        query_states, key_states, value_states
    )
except (ImportError, Exception):
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=None, dropout_p=0.0, is_causal=True
    )
```

**Effort: 10 lines. Impact: eliminates crash risk for all long-context serving.**

---

### W2. Wire StaticSparseDecodeGraph into Decode Path
**Priority: MEDIUM (reduces Python dispatch overhead)**

- Location: `ACTIVE_RUNTIME/native_core/graph_runtime/static_decode_graph.py`
- Problem: CUDA graph is never captured or replayed — `_batch_loop()` in `batch_engine.py` always calls `_step()` directly
- Blocker: Requires NativeBlockPool (needs C++ build) OR refactor to work with Python block structure
- Fix Option A: Refactor `StaticSparseDecodeGraph` to use `batched_sparse_attn_decode` as the `decode_fn` without NativeBlockPool
- Fix Option B: Build diffkv_core.so first (see N1 below)

**Effort: Medium. Impact: eliminates Python overhead per decode step once captured.**

---

### W3. Wire SharedBasis into Compression Pipeline
**Priority: LOW**

- Location: `RESEARCH_PROTOTYPES/compression/shared_basis.py`
- Currently not imported by KVRuntimeManager
- Would allow cross-session basis sharing — reduces compression rank needed per block
- Fix: Add optional `shared_basis` parameter to `KVRuntimeManager.__init__` and plug into `_compress_block_sync`

**Effort: Small. Impact: lower VRAM for multi-session workloads.**

---

### W4. Move Phase Markdown Reports Out of ACTIVE_RUNTIME
**Priority: LOW (housekeeping)**

- ACTIVE_RUNTIME/ contains ~110 phase*.md files mixed with runtime code
- Makes it impossible to see what's real vs. documentation at a glance
- Fix: `mv ACTIVE_RUNTIME/phase*.md ACTIVE_RUNTIME/archive/`

**Effort: 1 command. Impact: clarity.**

---

## Needs Native Code (C++ Build Required)

### N1. Build diffkv_core Python Extension
**Priority: HIGH — unlocks Triton fused decode kernel**

- Location: `ACTIVE_RUNTIME/native_core/diffkv_core/`
- Status: Full C++ source written (bindings.cpp, compressor_thread.cpp, paging_stream.cu, 4 headers)
- What's missing: `cmake -B build && cmake --build build` has never been run
- What it unlocks:
  - `NativeBlockPool` → enables `native_triton_sparse_attn_decode()` (real Triton fused decode)
  - `DiffKVBlockStateTable` → atomic CAS state machine for block lifecycle
  - `DiffKVCompressorThread` → C++ SPSC queue compressor (replaces Python threading.Queue)
  - `DiffKVPagingStream` → async CUDA stream D2H/H2D (replaces synchronous `.to("cpu")`)

**Steps:**
```bash
cd ACTIVE_RUNTIME/native_core/diffkv_core
pip install pybind11 cmake
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python)
cmake --build build --target diffkv_core
cp build/diffkv_core*.so ../
```

**Effort: 1–4 hours (environment-dependent). Impact: unlocks Triton fused decode + C++ async paging.**

---

### N2. Wire Triton Fused Decode Kernel After N1
**Priority: HIGH (after N1 is done)**

- Location: `ACTIVE_RUNTIME/native_core/sparse_decode/triton_sparse_attn.py`
- The `@triton.jit _fused_sparse_decode_kernel` is complete and correct
- Once NativeBlockPool is importable, replace `batched_sparse_attn_decode()` call in `diffkv_attention.py` with `native_triton_sparse_attn_decode()`
- Expected impact: eliminates the N-step Python accumulation loop; full FlashAttention accumulation in SRAM

**Effort: Wire-up after N1. Impact: largest remaining decode speedup.**

---

### N3. Wire CUDA Graph Replay After N1+N2
**Priority: MEDIUM (after N1, N2)**

- Once NativeBlockPool and Triton decode are live, capture static decode graph
- `StaticSparseDecodeGraph` code already exists and is correct
- Wire into `ContinuousBatchEngine._step()` decode branch

**Effort: Small integration after N1+N2. Impact: eliminates all Python overhead from steady-state decode.**

---

## Needs Distributed Runtime (Do Not Attempt Without N1 First)

### D1. Single-Machine Multi-GPU Serving
**Current state: NOT IMPLEMENTED — single GPU only**

Required steps (in order):
1. Complete N1 (native build)
2. Implement real tensor-parallel attention: split heads across GPUs via `device_map="auto"` or manual shard
3. Wire `torch.distributed.init_process_group()` in launch script
4. Replace Python-dict stubs in `RESEARCH_PROTOTYPES/distributed/` with real `dist.send/recv`
5. Implement real NCCL sync in `nccl_stream_synchronizer.py`

**Current stub code is unusable — must be rewritten from scratch.**

---

### D2. Cross-GPU KV Slab Ownership
**Current state: Architecture only in RESEARCH_PROTOTYPES/distributed/**

- Design exists in phase26 markdown and distributed/ Python scaffolding
- No real implementation
- Requires D1 + NVLink or PCIe P2P to be viable
- Do not implement until single-GPU runtime is fully native (N1+N2+N3 done)

---

## Pure Research (Not Engineering Steps)

These are research questions, not blocked engineering tasks:

| Topic | Status |
|---|---|
| Sparse FFN execution (block-sparse MLP) | Research prototype exists; integration unclear; not in roadmap |
| Shared basis across sessions | Prototype exists; uncertain VRAM savings vs. overhead |
| Retrieval-aware sparse prefill for 100K+ context | Partial prototype; needs research validation first |
| Adaptive micro_block_size tuning | Research question; current default (16) works |
| Quantized anchor tokens (INT8/FP8) | Research; quantization.py exists but disconnected |

---

## Dead / Rejected (No Action)

| Item | Reason |
|---|---|
| All content in `RESEARCH_PROTOTYPES/distributed_nccl/` | Stubs; rewrite required, not fix |
| Root-level empty directories (distributed/, triton_kernels/, etc.) | Delete |
| 49 MB phase4_reconstruction_trace.json | Archive |
| 100+ phase*.md files in ACTIVE_RUNTIME/ | Archive to ACTIVE_RUNTIME/archive/ |
| Real-multiuser orchestrators in RESEARCH_PROTOTYPES root | Superseded by batch_engine.py |
| patch_hf_decode_bypass.py in RESEARCH_PROTOTYPES | Superseded by diffkv_attention.py |

---

## Roadmap Priority Order

```
1. [W1] Fix long-context import guard       ← 10 lines, TODAY
2. [N1] Build diffkv_core.so               ← 1-4 hours, HIGH IMPACT
3. [N2] Wire Triton fused decode            ← after N1, HIGH IMPACT
4. [W4] Archive phase*.md from ACTIVE_RUNTIME ← housekeeping, any time
5. [N3] Wire CUDA graph replay              ← after N1+N2
6. [W2] Wire StaticSparseDecodeGraph        ← after N3
7. [W3] Wire SharedBasis                    ← low priority
8. [D1] Multi-GPU serving                  ← after all N items done
```
