# Phase 23 Race Condition Report

This report documents every race condition identified during Phase 23 stress testing and confirms that the native C++ design eliminates each one.

## Race Conditions Eliminated

### RC-1: Compression vs. Eviction (Previously Fatal)
**Scenario:** The Python LRU eviction thread detects VRAM pressure and calls `paged_kv_store.evict(block_id)` while the Python compressor thread is mid-SVD on the same block.
**Previous failure:** Both operations write to the block. The eviction copies stale half-computed data to CPU RAM. The eviction completes; the compressor then writes the compressed result to a GPU slot that has already been freed. **Silent data corruption.**
**Phase 23 fix:** `DiffKVBlockStateTable.transition()` uses `compare_exchange_strong`. The eviction calls `transition(block_id, CompressedResident, PagingOut)`. This CAS **fails** because the block is currently `Compressing`. Eviction is rejected. The block remains in `Compressing` state until the SVD completes and commits to `CompressedResident`. Only then is it eligible for eviction.
**Result: ELIMINATED.**

### RC-2: Stale Metadata Read During Graph Replay
**Scenario:** The Triton Sparse Decode kernel is launched via CUDA Graph replay. Between graph capture and replay, a background thread updates the MetadataPool U/V pointers for a block being simultaneously reloaded.
**Previous failure:** The Triton kernel reads stale GPU pointers from the pre-captured metadata, producing numerically corrupted attention output.
**Phase 23 fix:** `are_replay_safe(block_ids)` performs lock-free atomic reads of all blocks' states before graph replay is triggered. Any block in `Reloading` state blocks the replay via `cudaStreamWaitEvent` on the paging stream's completion event — inserted at the GPU scheduler level, not the CPU level.
**Result: ELIMINATED.**

### RC-3: Session Disconnect During Reload
**Scenario:** A session disconnects while its block is being transferred from CPU to GPU via `cudaMemcpyAsync`. The CPU-side pinned buffer is freed. The GPU-side transfer continues writing to an in-use slab slot.
**Previous failure:** The freed CPU buffer triggers undefined behavior. The slab slot is corrupted with partial data.
**Phase 23 fix:** CUDA cannot cancel an in-flight `cudaMemcpyAsync` safely. The transfer completes to the pre-allocated destination. `poll_completions()` then checks `session_alive()`. On disconnect detection, `force_invalidate(block_id)` is called before the slab slot is marked available. The slab pool's cleanup pass confirms `Invalid` state before freeing the slot.
**Result: SAFELY HANDLED (transfer completes to a safe buffer; data is discarded, not corrupted).**

### RC-4: Queue Overflow Double-Free
**Scenario:** The SPSC ring buffer overflows. The rejected `CompressJob` is discarded. Meanwhile the block remains `DenseResident`. A second compression trigger fires for the same block, creating a duplicate job.
**Phase 23 fix:** `submit()` transitions the block `DenseResident → Compressing` as a CAS **before** pushing to the queue. If the queue is full, the CAS is reversed: `Compressing → DenseResident`. The second trigger finds the block `DenseResident` and retries safely.
**Result: ELIMINATED.**

## Summary
All four critical race conditions are resolved at the C++ atomic layer. Zero Python-level locking is involved in the hot path.
