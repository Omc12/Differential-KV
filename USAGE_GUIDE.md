# Differential KV: Unified Usage Guide (Stages 1, 2, & 3)

Differential KV has reached **Operational Maturity**. The "packaging" phase is complete, and the system is fully usable with standard LLM WebUIs (Open WebUI, LibreChat, etc.) via its OpenAI-compatible serving layer.

## Current System Status
- **Stage 1 (Semantic Governance)**: Stabilized and active.
- **Stage 2 (CDBE Engine)**: Native sparse acceleration and continuous batching are active.
- **Stage 3A (Operational Serving)**: Production-style packaging, session recovery, and browser stability are integrated.

---

## 1. How to Run the Server

We have consolidated all stages into a single entry point: `run_diffkv_webui_server.py`. This script boots the unified runtime, initializes hardware acceleration, and starts the API gateway.

### Start Command
```bash
python run_diffkv_webui_server.py
```

### Server Details
- **Endpoint**: `http://localhost:8000/v1`
- **Model ID**: `diffkv-qwen2.5-7b`
- **Format**: OpenAI Chat Completions compatible.

---

## 2. Connecting a WebUI

Most WebUIs (like [Open WebUI](https://openwebui.com/)) support OpenAI-compatible backends.

### Open WebUI Configuration
1. Go to **Settings > Connections > OpenAI API**.
2. **OpenAI API Base URL**: `http://localhost:8000/v1`
3. **OpenAI API Key**: `sk-diffkv` (or any string, as authentication is local).
4. Save and select the `diffkv-qwen2.5-7b` model from the dropdown.

---

## 3. Key Recent Features for WebUI Users
- **Browser Resilience**: If you refresh the page or your tab is suspended, the `BrowserFailureRecoveryLayer` preserves your session context.
- **Token Smoothness**: The `InteractiveUXStabilityEvaluator` ensures tokens are delivered with minimal jitter for a premium chat feel.
- **WebSocket Streaming**: For high-concurrency environments, the server supports stable WebSocket-based token delivery.
- **Long Conversations**: Stage 3A.1 continuity monitors ensure that long-context sessions (up to 128k+ tokens) remain semantically grounded and stable.

---

## 4. Troubleshooting
- **No GPU Found**: The runtime will fallback to CPU (slow), but will still function for verification.
- **Connection Refused**: Ensure no other service is using port `8000`.
- **Model Loading**: First run may take a moment to initialize the `DiffKVHFWrapper` and Triton kernels.

---

### Is Packaging Left?
**No.** The `UnifiedRuntimePackagingLayer` implemented in Stage 3A serves as the final "glue" that allows Differential KV to be treated as a single, stable application rather than a collection of research modules. It is ready for interactive human usage.
