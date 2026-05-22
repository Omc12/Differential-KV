# Phase 22 Native Boundary

This is the final extraction roadmap. Every runtime component is classified by what it requires to become production-safe.

## Classification Table

| Component | Classification | Justification |
|-----------|---------------|---------------|
| **Triton Sparse Decode** | ✅ Already Safe in Python | Triton compiles to PTX natively. Python only dispatches the kernel launch — overhead is <5µs. No extraction needed. |
| **Persistent Metadata Pool** | ✅ Already Safe in Python | Pre-allocated fixed tensors at stable GPU addresses. Python only writes during block transitions, never during decode. |
| **Slab Pool Manager** | ✅ Already Safe in Python | Allocating from a fixed pre-allocated tensor pool is a few integer operations. Overhead is negligible. |
| **CUDA Graph Replay** | ✅ Already Safe in Python | vLLM's existing padded graph capture eliminates all invalidation jitter. Python invokes `graph.replay()` — overhead is negligible. |
| **Dense Recency Window** | ✅ Already Safe in Python | Simple tensor slice tracking. No hot-path overhead. |
| **Block State Machine** | 🔧 Needs Native Threading | State transitions happen on multiple threads (main + compressor + pager). Python's GIL makes atomic state transitions unreliable under load. Needs `std::atomic<BlockState>`. |
| **Async Compressor** | 🔧 Needs Native Threading | Lock-free SPSC queue + cuSOLVER worker. Python thread GIL contention under load causes measurable main-thread stalls. |
| **Paging Reload** | ⚡ Needs CUDA Runtime | `cudaMemcpyAsync` on a dedicated stream requires direct CUDA API access. Not expressible safely in Python without C++ extension. |
| **Slab Pool Eviction** | 🔧 Needs Native Threading | LRU eviction decisions under concurrent access require atomic ordering. |
| **Full End-to-End Serving** | 🏗️ Needs Full vLLM Integration | Scheduler, tokenizer, graph bucketing, multi-GPU tensor parallelism — all native in vLLM. Rebuilding this is out of scope. |

## The Extraction Priority Order
1. **Async Compressor** (Phase 23) — Highest systemic impact.
2. **Paging Reload** (Phase 23 co-target) — Small CUDA extension, eliminates final PCIe stall.
3. **Block State Machine** (Phase 23 co-target) — Prerequisite for thread-safe compressor integration.
4. **Full vLLM Backend** (Phase 24+) — Final deployment target.
