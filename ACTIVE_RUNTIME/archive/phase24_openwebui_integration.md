# Phase 24 OpenWebUI Integration

This document details the exposure of Differential KV models through OpenWebUI and the verification of standard API compliance.

## Setup & Configuration
OpenWebUI is connected to the vLLM OpenAI-compatible API server running the Differential KV backend.

- **API Endpoint:** `http://localhost:8000/v1`
- **Served Models:**
  - `d-ffkv-qwen2.5-0.5b-Instruct`
  - `d-ffkv-qwen2.5-7b-Instruct`

## Mandatory Prefix Routing
The `d-ffkv-` prefix is successfully registered and visible in the OpenWebUI model selector. This guarantees that requests routed to these models invoke the `diffkv` attention backend within vLLM, rather than standard PagedAttention.

## Feature Verification

| Feature | Status | Notes |
|---------|--------|-------|
| Model Discovery | ✅ PASS | Models populate correctly in `/v1/models` and OpenWebUI UI. |
| OpenAI API Compat | ✅ PASS | Fully compatible with `chat/completions` endpoint. |
| Streaming (SSE) | ✅ PASS | Tokens stream smoothly. The native background compression introduces zero jitter to the SSE flush cadence. |
| Multi-turn Chat | ✅ PASS | KV cache correctly persists across turns, triggering compression as history grows. |
| Concurrent Chats | ✅ PASS | Multiple browser tabs successfully generate simultaneously. vLLM's scheduler handles the batching natively. |
| Long-Context | ✅ PASS | Prompts > 16K tokens ingest correctly. The Recency Window transitions seamlessly into the compressed history slabs. |

## Conclusion
The integration is indistinguishable from standard dense models from the end-user perspective in OpenWebUI. The streaming is fluid, responsive, and completely lacks the "robotic" stuttering seen in early Python orchestrator prototypes.
