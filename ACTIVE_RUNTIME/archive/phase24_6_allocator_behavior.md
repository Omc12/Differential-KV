# Phase 24.6 -- CUDA Allocator Behavior Analysis

## Objective
Determine if VRAM savings are being hidden by PyTorch's CUDA allocator caching behavior.

---

## The Allocator Illusion

PyTorch uses a caching memory allocator to speed up memory allocations. This means that when a tensor is deleted (freed), the memory is **not** immediately returned to the GPU OS. It remains "reserved" by PyTorch for future use.

### The Metrics
- `allocated`: Memory physically occupied by active, live tensors.
- `reserved`: Total memory held by PyTorch, including both active tensors and freed but cached blocks.
- `inactive_split_bytes`: The portion of `reserved` memory that is currently unused (cached).

## Measurements from Phase 24.6 Audit

| Stage | Allocated | Reserved | Inactive (Cached) |
|---|---|---|---|
| Post-prefill (512 tok) | 1181.4 MB | 1199.6 MB | 18.2 MB |
| Post-GC (512 tok) | 1181.4 MB | 1199.6 MB | 18.2 MB |
| Idle (sessions cleared) | 1170.6 MB | 1189.1 MB | 18.5 MB |

### Analysis

1. **Low Allocator Overhead**: The allocator cache (inactive bytes) is surprisingly small, hovering around 18 MB. This means PyTorch is not hoarding massive amounts of unused VRAM.
2. **True Allocation Dominates**: The vast majority of `reserved` memory is actually `allocated` (physically in use by live tensors).
3. **No Massive GC Recoveries**: Calling `gc.collect()` and `torch.cuda.empty_cache()` does not result in massive VRAM drops. The memory being held is genuinely alive, not waiting for garbage collection.

## What is occupying the "Idle" 1170.6 MB?

Since the model weights are 988.1 MB, there is 182.5 MB of unaccounted live memory when the system is supposedly idle.

As discovered in the live residency analysis, this 182.5 MB is the **persistent compressed KV state** from the three sessions (32, 128, and 512 tokens).

Although `kv_manager.clear_session(sid)` is called, it seems the `StreamingSparseIngestManager` is not fully releasing the Python objects holding the U, V, and anchor_kv tensors. This is a memory leak in the session teardown logic, not a CUDA allocator issue.

## Verdict

The CUDA allocator is **NOT** masking massive VRAM savings. The VRAM savings from sparse ingest are real and accurately reflected in the `allocated` metric. The VRAM growth is bounded.

However, a session teardown leak is preventing memory from being fully released back to the baseline.
