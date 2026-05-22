# Phase 22 Memory State Machine

This is the authoritative state machine governing every KV block in the Differential KV runtime.

## States

| State | Description |
|-------|-------------|
| `DenseResident` | Block holds uncompressed `[K, V]` tensors in GPU VRAM. |
| `Compressing` | Block is being SVD-compressed by the C++ worker thread. |
| `CompressedResident` | Block holds `[U, V]` slab tensors in GPU VRAM. |
| `PagingOut` | Block is being asynchronously transferred from GPU → CPU. |
| `CPUResident` | Block holds `[U, V]` slab tensors in pinned CPU RAM. |
| `Reloading` | Block is being asynchronously transferred from CPU → GPU. |
| `Invalid` | Block was freed during an in-progress operation. |
| `Freed` | Block slot is available for reuse. |

## Legal State Transitions

```
DenseResident     → Compressing       (compression triggered)
Compressing       → CompressedResident (SVD success)
Compressing       → Invalid            (session disconnected mid-compression)
CompressedResident → PagingOut         (VRAM pressure, LRU eviction)
PagingOut         → CPUResident        (transfer complete)
PagingOut         → Invalid            (session disconnected mid-eviction)
CPUResident       → Reloading          (block needed for decode)
Reloading         → CompressedResident (transfer complete)
Reloading         → Invalid            (session disconnected mid-reload)
CompressedResident → Freed             (session ended cleanly)
Invalid           → Freed             (cleanup pass)
```

## Illegal Transitions (Must Panic/Assert)
- `DenseResident → CPUResident` (skipping compression is forbidden)
- `Compressing → PagingOut` (evicting a block while compressing causes data loss)
- `Reloading → PagingOut` (evicting a block being reloaded causes corruption)
- Any state → `DenseResident` (decompression is not supported; reload always returns Compressed)

## Synchronization Rules
1. State transitions are **atomic** (C++ `std::atomic<BlockState>`).
2. The Triton Sparse Decode kernel may ONLY access a block in `CompressedResident` state.
3. The C++ Compressor worker may ONLY write to a block in `Compressing` state.
4. The Paging system may ONLY initiate eviction on a block in `CompressedResident` state.

## Graph Replay Safety
CUDA Graph replay assumes all block addresses are stable. Before replay, the runtime must verify: `ALL blocks accessed by the upcoming step ∈ {CompressedResident, DenseResident}`. Any block in a transient state (`Reloading`, `PagingOut`) must be awaited before graph execution proceeds.
