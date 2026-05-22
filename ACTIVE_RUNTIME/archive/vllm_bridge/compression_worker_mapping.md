# Compression Worker Mapping

This defines how the asynchronous SVD compression engine integrates into vLLM without triggering Python GIL locks or GPU pipeline stalls.

## The vLLM Worker Model
vLLM relies on a central `LLMEngine` that orchestrates `Worker` processes (often via Ray for multi-GPU setups).

## The `DiffKVCompressionWorker`
Differential KV requires a dedicated background worker operating concurrently with the Model Runner.

### 1. Trigger Mechanism
When a sequence generates enough tokens to fill a dense block (e.g., 64 tokens), the `LLMEngine` queues a "Compress Event".

### 2. Physical Execution
The `DiffKVCompressionWorker` pulls the event.
- It accesses the Dense Memory Pool (read-only).
- It executes a custom C++ extension wrapper for batched `torch.linalg.svd`. (Must be C++ to release the GIL entirely, as Python `threading` proved insufficient under extreme load).
- It determines the rank via Adaptive Rank Selection.
- It writes the result to the Compressed Memory Pool.

### 3. Synchronization
Once the write completes, the worker sends an atomic update to the `BlockSpaceManager`, swapping the logical block's pointer from the Dense Pool to the Compressed Pool. The next decode step seamlessly routes to the sparse kernel.
