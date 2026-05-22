# Phase 20 Batch Engine Survival

The `ContinuousBatchEngine` was written specifically to test Differential KV's orchestration overhead. Now that we are migrating to a real serving backend (vLLM), we must aggressively abandon overlapping features.

## 1. What Becomes Obsolete
- **`ContinuousBatchEngine` entirely:** vLLM's `LLMEngine` and asyncio loops are vastly superior, highly optimized C++ bound schedulers. Our Python-based batching loop is dead code.
- **`PagedKVStore` (Partial):** We no longer need to manually execute `tensor.to('cpu')` based on custom PyTorch memory tracking. vLLM's `BlockSpaceManager` handles physical block allocation natively.

## 2. What Remains Essential
- **`AsyncCompressor`:** vLLM does not natively compress blocks in the background. This logic (SVD and rank selection) must survive and be wrapped into a custom vLLM background worker.
- **`Triton Sparse Decode`:** This is the heart of the project. It survives as a custom vLLM Attention Backend.
- **`PersistentMetadataPool`:** Necessary to translate vLLM's dense block tables into sparse rank configurations.

## Conclusion
Differential KV is not an engine. It is a **Memory Virtualization Layer**. The code in `ACTIVE_RUNTIME/serving` should be deleted entirely, and the project should exist purely as an extension package that hooks into vLLM.
