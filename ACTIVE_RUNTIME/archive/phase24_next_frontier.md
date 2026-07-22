# Phase 24 Next Frontier

## The True Next Systems Bottleneck

Now that Differential KV is a fully functional, deployment-ready vLLM backend for single-GPU serving, the most significant remaining barrier to widespread adoption is:

**Target: Multi-GPU Compressed Serving (Tensor Parallelism)**

## Justification

1. **Model Size Constraint:** Single-GPU serving is fundamentally limited by VRAM. Even with aggressive 4x compression, an 80GB GPU can only hold so much. Modern production models (70B+) require multiple GPUs (TP=2, TP=4, TP=8).
2. **The Distributed SVD Challenge:** Currently, the `AsyncCompressor` runs cuSOLVER on a single GPU. In a TP setup, the KV cache is sharded across GPUs. Performing SVD on sharded tensors requires either gathering the tensors (huge bandwidth cost) or implementing distributed SVD (complex math).
3. **Slab Coherence:** The native `DKVBlockStateTable` must be synchronized across multiple GPU workers.

## Why Not Sparse Prefill?
While sparse prefill (FlashAttention integration) would speed up the initial prompt ingestion, the core value proposition of Differential KV is **memory capacity** for long contexts. Multi-GPU support unlocks massive models, which is a far higher leverage capability for production serving than a faster prefill phase on a small model.

## Phase 25 Focus
Phase 25 must tackle the integration of Differential KV into vLLM's Tensor Parallelism architecture, specifically designing the distributed compression strategy and cross-GPU metadata synchronization.
