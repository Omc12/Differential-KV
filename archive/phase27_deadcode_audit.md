# Phase 27 — Dead Code Audit
## Separating REAL RUNTIME from RESEARCH RESIDUE

---

## Audit Scope

Every directory in the project root was classified. The goal: identify code that
cannot be reached from `launch_real_serving.py` under any code path.

---

## REAL RUNTIME — Files That Actually Execute

These files are reachable from `launch_real_serving.py` through direct or indirect imports:

```
ACTIVE_RUNTIME/
├── launch_real_serving.py                          ← entrypoint
├── serving/
│   ├── batch_engine.py                             ← LIVE
│   ├── hf_diffkv_wrapper.py                        ← LIVE
│   ├── openai_compatible_api_gateway.py             ← LIVE
│   └── production_session_manager.py               ← LIVE
├── runtime/
│   ├── diffkv_attention.py                         ← LIVE (patched into model)
│   ├── batched_sparse_attn.py                      ← LIVE (decode kernel)
│   ├── sparse_attention.py                         ← LIVE (fallback path)
│   └── lgs_resolver.py                             ← CHECK (may be unused)
├── native_core/
│   ├── kv_runtime_manager.py                       ← LIVE
│   ├── streaming_sparse_ingest.py                  ← LIVE
│   ├── recon_cache.py                              ← LIVE
│   ├── compression/
│   │   ├── async_compressor.py                     ← LIVE
│   │   └── lowrank.py                              ← LIVE
│   ├── paging/
│   │   └── paged_kv_store.py                       ← LIVE
│   └── sparse_decode/
│       ├── triton_diffkv.py                        ← LIVE (with fallback)
│       └── triton_sparse_attn.py                   ← DEAD (needs unbuilt C++)
└── RESEARCH_PROTOTYPES/compression/
    └── adaptive.py                                 ← LIVE (via try/except import)
```

---

## DEAD CODE — Confirmed Unreachable

### Category 1: Empty Directories (No Code At All)

These were created but never populated — pure migration artifacts:

| Path | Reason Dead |
|---|---|
| `distributed/` (root) | Empty — only `__pycache__` |
| `distributed_nccl/` (root) | Empty — only `__pycache__` |
| `distributed_execution/` (root) | Empty |
| `distributed_inference/` (root) | Empty |
| `distributed_optimization/` (root) | Empty |
| `distributed_orchestration/` (root) | Empty |
| `triton_kernels/` (root) | Empty |
| `kernels/` (root) | Empty |
| `ACTIVE_RUNTIME/native_core/kernels/` | Empty |
| `ACTIVE_RUNTIME/native_core/residency/` | Empty |

**Action: Safe to delete all 10 directories.**

---

### Category 2: C++ Source Never Compiled

| Path | Status |
|---|---|
| `ACTIVE_RUNTIME/native_core/diffkv_core/src/bindings.cpp` | Source only — no .pyd/.so |
| `ACTIVE_RUNTIME/native_core/diffkv_core/src/compressor_thread.cpp` | Source only |
| `ACTIVE_RUNTIME/native_core/diffkv_core/src/paging_stream.cu` | Source only — never compiled |
| `ACTIVE_RUNTIME/native_core/diffkv_core/include/*.hpp` (4 headers) | Source only |
| `ACTIVE_RUNTIME/native_core/diffkv_core/CMakeLists.txt` | Build config — never run |

**Action: Either build or move to `RESEARCH_PROTOTYPES/native_core_src/`. Do not delete — building this is the primary next engineering step.**

---

### Category 3: Disconnected Python Code in ACTIVE_RUNTIME

| File | Why Dead |
|---|---|
| `ACTIVE_RUNTIME/native_core/graph_runtime/static_decode_graph.py` | Not imported; depends on unbuilt NativeBlockPool |
| `ACTIVE_RUNTIME/native_core/vllm_bridge/*.md` (4 files) | Documentation, not code |
| `ACTIVE_RUNTIME/native_core/sparse_decode/triton_sparse_attn.py` | Not reached — needs NativeBlockPool |
| `ACTIVE_RUNTIME/compression/quantization.py` | Not imported in serving path |
| `ACTIVE_RUNTIME/compression/quantization_advanced.py` | Not imported in serving path |
| `ACTIVE_RUNTIME/compression/shared_basis.py` | Not imported in serving path |
| `ACTIVE_RUNTIME/compression/sparse_repair.py` | Not imported in serving path |
| `ACTIVE_RUNTIME/runtime/lgs_resolver.py` | Needs import verification — may be unused |
| `ACTIVE_RUNTIME/runtime/prefill_attention_pruner.py` | Not imported in main path |
| `ACTIVE_RUNTIME/runtime/sparse_prefill_mlp.py` | Not imported in main path |
| `ACTIVE_RUNTIME/runtime/triton_sparse_mlp.py` | Not imported in main path |

---

### Category 4: Research Prototypes — Stubs and Scaffolding

The `RESEARCH_PROTOTYPES/` directory contains 109 subdirectories and ~70 files.
The vast majority is dead research residue:

**Confirmed stubs (real logic commented out):**
- `RESEARCH_PROTOTYPES/distributed_nccl/nccl_stream_synchronizer.py`
- `RESEARCH_PROTOTYPES/distributed_nccl/nccl_graph_orchestrator.py`
- `RESEARCH_PROTOTYPES/distributed_nccl/distributed_replay_stabilizer.py`

**Confirmed empty scaffolding classes:**
- `RESEARCH_PROTOTYPES/distributed/global_sparse_router.py` (207 B)
- `RESEARCH_PROTOTYPES/distributed/retrieval_locality_mesh.py` (194 B)
- `RESEARCH_PROTOTYPES/distributed/distributed_hotset_tracker.py` (187 B)
- `RESEARCH_PROTOTYPES/distributed/node_affinity_scheduler.py` (220 B)
- `RESEARCH_PROTOTYPES/distributed/remote_anchor_prefetch.py` (209 B)

**Files with real logic but not connected to serving:**
- `RESEARCH_PROTOTYPES/compression/shared_basis.py` — real SVD basis code, disconnected
- `RESEARCH_PROTOTYPES/compression/delta_encoder.py` — real delta encoder, disconnected
- `RESEARCH_PROTOTYPES/compression/adaptive_scheduler.py` — real scheduler, disconnected

**Dead orchestration files at RESEARCH_PROTOTYPES root (sample):**
- `enable_execution_audit.py` (7.2 KB) — meta-audit script, not serving code
- `deployment_reproducibility_manager.py` (4 KB) — deployment tooling
- `real_multiuser_serving_orchestrator.py` (4.5 KB) — duplicate serving path
- `patch_hf_decode_bypass.py` (5 KB) — alternative patch, superseded by diffkv_attention.py
- `legacy_system_classifier.py` (6 KB) — classification tool, not runtime
- `differential_kv_cli.py` (7 KB) — CLI tool
- Dozens of single-purpose controller files (`sparse_qos_stabilizer.py`, `memory_pressure_safety_system.py`, etc.) — none imported in serving

---

### Category 5: Duplicate Phase Report Markdown Files (145+ files)

`ACTIVE_RUNTIME/` contains 145 items. ~110 of them are markdown phase reports:
`phase15_runtime_reality_report.md`, `phase16_category_execution_trace.md`, ..., `phase26_hyperscale_limits.md`

**These are documentation artifacts, not code.** They do not execute.
They accumulate indefinitely and clutter the ACTIVE_RUNTIME directory.

**Action: Move all `phase*.md` files to `ACTIVE_RUNTIME/archive/`.**

---

### Category 6: Oversized Trace JSON Files

| File | Size |
|---|---|
| `ACTIVE_RUNTIME/phase4_reconstruction_trace.json` | **49.3 MB** |
| `ACTIVE_RUNTIME/phase3_triton_trace.json` | **5.8 MB** |

These are runtime traces from early phases. They are never read by serving code.
**Action: Archive or delete. They bloat the repo significantly.**

---

## Summary: Real Runtime vs Research Residue

| Category | Count | Disposition |
|---|---|---|
| **Live serving files** | ~12 Python files | KEEP — these are the runtime |
| **Live research import (adaptive.py)** | 1 file | KEEP |
| **Disconnected but valuable** (C++ src, static_decode_graph, triton_sparse_attn) | ~10 files | KEEP — next engineering targets |
| **Empty directories** | 10 directories | DELETE |
| **Phase markdown reports in ACTIVE_RUNTIME** | ~110 files | MOVE to archive/ |
| **Oversized trace JSON** | 2 files | ARCHIVE |
| **RESEARCH_PROTOTYPES stubs** | ~60 files | ARCHIVE or DELETE |
| **RESEARCH_PROTOTYPES empty scaffolding** | ~20 files | DELETE |
| **Duplicate serving orchestrators in RP** | ~5 files | DELETE |

---

## Real Runtime Surface (Minimal)

If stripped to only what executes, Differential KV's live runtime is:

```
12 Python files
~2,200 lines of code (excluding tests)
1 C++ module (unbuilt but critical for next phase)
2 Triton kernel files (1 live, 1 pending native build)
```

Everything else is research history.
