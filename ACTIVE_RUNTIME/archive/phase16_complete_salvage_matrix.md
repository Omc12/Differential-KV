# Phase 16 Complete Salvage Matrix

This matrix classifies all recovered systems across the 7 salvage categories, defining their actual physical integration status within the Differential KV ecosystem.

## Status Definitions
- **ACTIVE_EXECUTING**: Wired directly into `batch_engine.py` or `KVRuntimeManager`. Modifies live execution.
- **PARTIAL**: Executable but not fully wired into the end-to-end hot path.
- **RESEARCH_ONLY**: Valid mathematically but too slow or complex for production.
- **FUTURE_NATIVE_ONLY**: Validated conceptually, but fundamentally blocked by Python/Triton limits (requires C++ backend).
- **DISCONNECTED**: Exists in the codebase but bypassed.
- **REJECTED**: Architecture theater, fake telemetry, or mathematically unsound.

---

### CATEGORY 1 — Sparse KV Runtime
| System | Source Location | Execution Path | Status |
|--------|----------------|----------------|--------|
| Async KV Compressor | `runtime/async_compressor.py` | `KVRuntimeManager._compress_block_sync` | **ACTIVE_EXECUTING** |
| Paged KV Store | `runtime/paged_kv_store.py` | `KVRuntimeManager` | **ACTIVE_EXECUTING** |
| Adaptive Rank Selection | `compression/adaptive.py` | `KVRuntimeManager._compress_block_sync` | **ACTIVE_EXECUTING** |
| Shared Basis Compression | `compression/shared_basis.py` | None | **DISCONNECTED** |
| Triton Sparse Decode | `runtime/triton_sparse_attn.py` | `batch_engine.py` / `hf_diffkv_wrapper.py` | **ACTIVE_EXECUTING** |

### CATEGORY 2 — Sparse Transformer Execution
| System | Source Location | Execution Path | Status |
|--------|----------------|----------------|--------|
| Block-Sparse MLP (Python) | `runtime/sparse_mlp.py` | Phase 11 Test Harness | **RESEARCH_ONLY** |
| Fused Triton Sparse MLP | `runtime/sparse_mlp_fused.py` | Phase 11 Test Harness | **FUTURE_NATIVE_ONLY** |
| Layer Compressibility Analyzer | `runtime/layer_compressibility.py` | Phase 12 Test Harness | **RESEARCH_ONLY** |

### CATEGORY 3 — Hierarchical Residency
| System | Source Location | Execution Path | Status |
|--------|----------------|----------------|--------|
| Tiered FFN Weights | `runtime/tiered_ffn.py` | Phase 12 Test Harness | **FUTURE_NATIVE_ONLY** |
| Predictive Loading | `RESEARCH_PROTOTYPES/` | None | **REJECTED** |

### CATEGORY 4 — Sparse Prefill Systems
| System | Source Location | Execution Path | Status |
|--------|----------------|----------------|--------|
| Chunked Sparse Prefill | `runtime/sparse_prefill.py` | Phase 13 Test Harness | **RESEARCH_ONLY** |
| Prefill Attention Pruner | `runtime/prefill_attention_pruner.py` | None | **DISCONNECTED** |

### CATEGORY 5 — Runtime Fusion & Native Execution
| System | Source Location | Execution Path | Status |
|--------|----------------|----------------|--------|
| Fused FlexAttention Prefill | `runtime/fused_sparse_prefill.py` | Phase 15 Test Harness | **FUTURE_NATIVE_ONLY** (SRAM limits) |
| CUDA Graph Schedulers | `EXPERIMENTAL_RUNTIME/` | None | **REJECTED** |

### CATEGORY 6 — Distributed Sparse Runtime
| System | Source Location | Execution Path | Status |
|--------|----------------|----------------|--------|
| Multi-GPU Paging | `ARCHIVED_SYNTHETIC_SYSTEMS/distributed`| None | **REJECTED** |
| Distributed KV | `RESEARCH_PROTOTYPES/distributed` | None | **REJECTED** |

### CATEGORY 7 — Locality / Geometry / Anchor Systems
| System | Source Location | Execution Path | Status |
|--------|----------------|----------------|--------|
| Global Memory Anchors | `runtime/sparse_prefill_anchors.py` | Phase 14 Test Harness | **FUTURE_NATIVE_ONLY** |
| Narrative Semantic Routing | `ARCHIVED_SYNTHETIC_SYSTEMS/` | None | **REJECTED** |
| Geometry Execution | `ARCHIVED_SYNTHETIC_SYSTEMS/` | None | **REJECTED** |
