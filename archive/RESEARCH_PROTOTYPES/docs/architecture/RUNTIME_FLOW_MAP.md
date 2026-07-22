# Differential KV: Runtime Flow Map

## Execution Hotpath

1. **Request Ingestion**: `OpenAICompatibleAPIGateway` receives incoming HTTP/vLLM requests.
2. **Scheduling**: `SparseRequestScheduler` groups requests into occupancy-aware batches.
3. **Model Wrapper**: `DKVHFWrapper` intercepts standard HuggingFace calls.
4. **Residency Check**: `ActiveGPUResidencyController` ensures KV caches are GPU-resident.
5. **Fused Decode**: `DecodePipelineFusionEngine` executes Triton kernels for:
    - Sparse Attention calculation.
    - KV Virtualization lookup.
    - Token-Collapse pruning.
6. **Streaming**: Tokens are streamed back to the user via the API Gateway.

## Control Loops

- **ATC Loop**: Continuously prunes non-essential KV pairs based on attention scores.
- **LGS Loop**: Monitors latency constraints and adjusts batch sizes in real-time.
- **PDM Loop**: Periodically checkpoints session state for crash recovery.
- **Safety Loop**: Monitors VRAM pressure and triggers emergency offloading if needed.