# Phase 17 Static Sparse Graphs

## The Problem
Eager PyTorch execution issues hundreds of tiny kernel launches during the decode step. Even with persistent metadata pools, invoking the Triton sparse kernel requires Python-side shape checking, tensor bounds validation, and dynamic launch grid calculation. At massive scale continuous batching, this fragments the GPU schedule.

## The Solution
We implemented `StaticSparseDecodeGraph` using `torch.cuda.CUDAGraph`.
- **Static Topology:** Because the `PersistentMetadataPool` guarantees that block indices and U/V matrices reside at fixed memory addresses, we can capture the Triton kernel execution.
- **Graph Replay:** During `ContinuousBatchEngine.step()`, instead of dispatching the Triton kernel, we simply `copy_` the incoming query vector into the static input buffer and invoke `graph.replay()`.

## Result
Coupled with the Metadata Pool, the static execution graph collapsed Python orchestration overhead completely for the decode hot path. We avoided dynamic graph rebuilds because the topology (the maximum batch size and block pointers) remains static, shifting the dynamism entirely into the static index array read by the Triton kernel.
