# Phase 20 Serving Mapping

This document provides a strict, physical mapping between Differential KV concepts and Native Serving Equivalents (vLLM).

| Differential KV Component | Native Serving Equivalent |
|---------------------------|--------------------------|
| `PagedKVStore` | `vllm.core.block_manager.BlockSpaceManager` |
| `MetadataPool` | `vllm.attention.backends.abstract.AttentionMetadata` |
| `AsyncCompressor` | Custom Ray Actor / Native C++ background thread |
| `Triton Sparse Decode` | `vllm.attention.ops.paged_attention` (Custom Backend) |
| `Dense Recency Window` | Standard uncompressed vLLM KV blocks |
| `Adaptive Rank Selection` | Custom metadata field in `BlockSpaceManager` tracking block compression state |
| `ContinuousBatchEngine` | `vllm.engine.llm_engine.LLMEngine` |

**Conclusion:** We do not need to invent new allocators or schedulers. Differential KV fits perfectly as a custom attention backend and a specialized background compression task within the existing vLLM architecture.
