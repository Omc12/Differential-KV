# Phase 23 Next Frontier

## Phase 24 Target: Full vLLM Compressed Backend Integration

## Justification

Phase 23 proved that the three native components (`DiffKVBlockStateTable`, `DiffKVCompressorThread`, `DiffKVPagingStream`) are production-safe, race-condition-free, and mechanically stable. The runtime is no longer a Python threading experiment — it is a compiled native subsystem.

The next highest-leverage move is to stop operating as a **standalone runtime** and instead integrate directly into **vLLM** as a proper backend.

### Why vLLM Integration Now?
1. **Slab Pool Cooperates with vLLM's Allocator:** Now that slabs are fixed-size (Rank-8/16/32), vLLM's `BlockSpaceManager` can manage the three pools natively. This removes our custom VRAM budget tracking entirely.
2. **Graph Capture Unification:** vLLM already handles padded CUDA Graph capture for varying batch sizes. Plugging Differential KV decode into vLLM's graph capture eliminates our remaining graph invalidation problem.
3. **`are_replay_safe()` Moves Into vLLM Scheduler:** The 1,024 atomic reads per step (newly discovered bottleneck in Phase 23) disappear when the vLLM scheduler is aware of block states natively.
4. **Multi-Request Scheduling:** vLLM's token scheduler is battle-tested at scale. Replacing our `ContinuousBatchEngine` with vLLM eliminates the last significant Python orchestration surface.

### Why Not Multi-GPU First?
Multi-GPU coherence requires tensor parallelism, which requires vLLM integration first. You cannot parallelize across GPUs without a production serving backbone.

### Why Not Sparse Prefill?
Still blocked by hardware SRAM limits. Not an orchestration problem.

## Phase 24 Deliverable
A functional `vllm.attention.backends.diffkv` module, registered via vLLM's backend registry, that:
1. Uses `DiffKVBlockStateTable` for block state tracking.
2. Dispatches to `TritonSparseDecode` for compressed blocks.
3. Registers `DiffKVCompressorThread` as a background vLLM worker.
4. Uses `DiffKVPagingStream` for all block evictions/reloads.
