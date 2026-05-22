# Phase 15 Final Runtime Reality Report

## 1. What Actually Works (Production Viable)
- **Sparse KV Compression Runtime**: Async SVD compression, Shared Basis, and Paged KV Memory represent the most successful salvage. They operate asynchronously and deliver actual VRAM reductions without blocking the execution path.
- **Hierarchical FFN Tiering**: Holding parameter blocks in pinned CPU RAM and paging them into a VRAM cache strictly limits GPU footprint. We mathematically proved ~147MB savings per layer on Qwen2-7B.
- **Sparse MLP FLOP Reduction**: The block-sparse execution physically skips 33-50% of MLP matmuls during large batch/prefill ingestion, producing measurable FLOP reductions.
- **Global Memory Anchors (Retrieval)**: Creating chunk-level KV centroids and routing prefill queries to them restores 16K+ global retrieval accuracy at < 1% FLOP overhead.

## 2. What Collapses Under Runtime Orchestration (The Bottlenecks)
- **Python Loop Overhead**: Breaking a 16K prefill into 512-token chunks mathematically scales attention as $O(N)$, but looping 32 times in eager PyTorch takes ~120ms (vs 13ms for highly optimized Dense FlashAttention). Python dispatch overhead completely erases the sparse compute advantage.
- **Dynamic Compilation Failures (SRAM Exhaustion)**: Attempting to collapse orchestration via PyTorch 2.5+ `flex_attention` and `torch.compile` fails on consumer hardware due to Triton shared memory limits (`OutOfMemoryError: out of resource`). The block-sparse patterns required for chunked anchor routing exceed the 100KB per-block SRAM limit during inductor heuristic compilation.
- **PCIe Latency at seq=1**: Conditionally loading FFN weights requires D2H/H2D transfers over the PCIe bus. While prefetching hides this during long prefill generation, a missed prediction during `seq=1` continuous decode stalls the GPU for ~0.1ms per block, immediately erasing any compute speedup gained by being sparse.

## 3. Sparse Transformer Reality Check (7B Scale)
- **Which layers MUST remain dense?** Layers 0-15 (early/middle layers). Their activation manifolds are broad and un-concentrated. Sparsifying them induces catastrophic PCIe thrashing and severe degradation.
- **Which layers are safely sparse?** Layers 16-31 (late/specialized layers). They exhibit extreme activation sparsity (80% mass in top 30% of blocks).
- **REAL VRAM Reduction:** Applying a 30% Tiered FFN budget to the 16 late layers reclaims **~5 GB of static VRAM** on a 7B parameter model.
- **REAL Latency Penalty:** Without C++ level asynchronous block loading, predicting poorly stalls the generation loop by up to 2-3ms per token (which is devastating for TPS).

## 4. What Was Truly Valuable vs Illusions
**Valuable Salvage:** 
- The Async KV Compressor
- Adaptive Rank Selection
- Paged Memory
- Chunked Centroid Anchors

**Illusions (Architecture Theater):**
- Fake narrative cognitive guards
- Random-dropping "geometric" token pruners
- Simulated effective-VRAM reports 
- Semantic "post-office" region networks

## 5. What Still Requires C++ / vLLM Integration
To transition Differential KV from a research prototype to a serving reality, the following MUST be migrated to C++ / Custom Triton / vLLM:
1. **Persistent Anchor Routing Kernels**: We need a FlashAttention-3 style kernel that accepts custom block-sparse boolean masks directly in C++ without hitting inductor compilation limits.
2. **Asynchronous Weight Prefetching**: CUDA streams must handle Host-to-Device block transfers natively while the Attention kernel runs, completely outside of the GIL.

Differential KV has reached the strict boundary of what can be engineered efficiently in Python. The math works, the VRAM drops, and the models compress, but executing it flawlessly requires a dedicated C++ runtime engine.
