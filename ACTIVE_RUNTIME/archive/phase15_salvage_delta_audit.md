# Phase 15 Salvage Delta Audit

## Overview
This audit traces the actual integration points of all systems revived from salvage or developed in Phases 11-14 into `ACTIVE_RUNTIME`. The harsh reality is that most recent systems were validated in isolated microbenchmarks but remain disconnected from the live `batch_engine.py` and `hf_dkv_wrapper.py` execution paths.

## Revived Systems & Integration Status

| System | File Location | Integration Point | Status |
|---|---|---|---|
| **Adaptive Rank Selection** | `kv_runtime_manager.py` (lines 30-45, via importlib) | Wired into `KVRuntimeManager._compress_block_sync` | **ACTIVE_EXECUTING** |
| **Shared Basis Compression** | `compression/shared_basis.py` | None | **DISCONNECTED** |
| **Block Sparse MLP (PyTorch)** | `runtime/sparse_mlp.py` | Tested in Phase 11 harness | **DISCONNECTED** |
| **Fused Triton Sparse MLP** | `runtime/sparse_mlp_fused.py` | Tested in Phase 11 harness | **DISCONNECTED** |
| **Prefill Attention Pruner** | `runtime/prefill_attention_pruner.py` | None | **DISCONNECTED** |
| **Tiered FFN Residency** | `runtime/tiered_ffn.py` | Tested in Phase 12 harness | **DISCONNECTED** |
| **Layer Compressibility Analyzer** | `runtime/layer_compressibility.py` | None | **DISCONNECTED** |
| **Chunked Sparse Prefill** | `runtime/sparse_prefill.py` | Tested in Phase 13 harness | **DISCONNECTED** |
| **Prefill-Aware Sparse MLP** | `runtime/sparse_prefill_mlp.py` | Tested in Phase 13 harness | **DISCONNECTED** |
| **Global Memory Anchors (Retrieval)** | `runtime/sparse_prefill_anchors.py` | Tested in Phase 14 harness | **DISCONNECTED** |

## Remaining Disconnected Systems (To be Fused)
The entire Sparse Transformer stack (MLP routing, tiered weights, sparse prefill attention, anchor routing) is mathematically verified but un-orchestrated. To become real, they must be fused into the live transformer forward pass (via `hf_dkv_wrapper.py` and `batch_engine.py`).

## Architecture Theater (Rejected)
- Fake semantic cognition networks
- Simulated "effective VRAM" reporting loops
- Random-dropping token pruners
- Narrative-based "geometric routing"

## Next Step
Task 2 focuses on Orchestration Collapse to physically wire these disconnected execution kernels into the continuous serving runtime, minimizing Python loop overheads before benching.
