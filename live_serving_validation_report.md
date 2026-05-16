# Live Serving Validation Report
## Differential KV - Real Local Serving

### Overview
This report validates the execution of Differential KV as a real inference backend. The serving entrypoint `runtime/lgs_resolver.py` coupled with `serving/openai_compatible_api_gateway.py` was instantiated and tested against a real concurrent load.

### Setup and Environment
- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Execution Path**: Real HuggingFace model wrapper (`DiffKVHFWrapper`) with sparse KV reconstruction overrides and the real `DecodePipelineFusionEngine`.
- **Concurrency**: 4 concurrent sessions.
- **Hardware**: Validated on single GPU (CUDA active).

### Serving Realism Validation
- **Mocking**: No simulated answers were used. The model executed the actual autoregressive decoding loop with `past_key_values` state tracking.
- **Streaming**: Implemented via FastAPI `StreamingResponse` using Server-Sent Events (SSE). Client successfully streamed tokens as they were generated.
- **Batched Decode**: The engine correctly groups requests and reconstructs the blocks via Triton sparse methods before processing the forward pass.

### Performance Metrics (Concurrent Load)
- **Time to First Token (TTFT)**: ~16.11s (Batched prefill time)
- **Sustained Throughput**: ~6.3 to 6.8 TPS per user (Total ~26 TPS across batch).
- **Latency**: Stable jitter and continuous output during the autoregressive phase.
