# Phase 22 Next Frontier

## Phase 23 Target: Full C++ AsyncCompressor + Paging Reload + Block State Machine

## Why This Specific Combination

Phase 22 designed three tightly coupled native components:
1. **C++ AsyncCompressor** (lock-free SPSC queue + cuSOLVER worker)
2. **Async Paging Reload** (dedicated CUDA stream + event synchronization)
3. **Block State Machine** (atomic state with legal transition enforcement)

These three are not independent. The State Machine is the **synchronization backbone** that makes the Compressor and Pager safe to run concurrently. Extracting the Compressor without the State Machine creates race conditions. Extracting the Pager without event-safe State Machine transitions corrupts graph replay.

**They must be implemented together in Phase 23 as a single C++ extension module: `libdkv_core.so`.**

## Justification: Why Not vLLM Integration First?
Full vLLM backend integration requires these three components to already be native — vLLM will not accept Python GIL-bound threads as production workers. Phase 23 builds the native building blocks. Phase 24 wires them into vLLM.

## Justification: Why Not Sparse Prefill First?
Sparse prefill is blocked by SRAM limits on consumer hardware, not by orchestration. It requires hardware support (H100 SRAM characteristics) or a custom FlashAttention-3 kernel rewrite. This is a multi-month kernel engineering effort with uncertain ROI on the primary bottleneck (long-context memory capacity).

## Justification: Why Not Hierarchical Retrieval Runtime?
Retrieval routing is an architectural expansion, explicitly banned by the Phase 22 rules. We are not in a research phase.

## Phase 23 Deliverable
A single PyBind11-compiled C++ extension module `dkv_core`:
```
dkv_core.DKVCompressorThread   — lock-free compression worker
dkv_core.DKVPagingStream       — CUDA-stream-based async reload
dkv_core.DKVBlockStateTable    — atomic state machine table
```
This module becomes the new physical backbone of `native_core/`, replacing all Python threading in the compression and paging paths.
