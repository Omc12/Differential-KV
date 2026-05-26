# DiffKV — Open WebUI Integration Guide

DiffKV exposes a fully OpenAI-compatible REST API. Connecting it to **Open WebUI** requires
two steps: start the DiffKV server, then add it as an external API connection in Open WebUI.

---

## Step 1 — Start the DiffKV Gateway

Open a terminal in the `ACTIVE_RUNTIME/` directory and run:

```powershell
# Minimum viable (CPU-only / lightweight)
python -m serving.openai_compatible_api_gateway `
    --model Qwen/Qwen2.5-1.5B-Instruct `
    --host 0.0.0.0 `
    --port 8000 `
    --serving-mode balanced

# GPU recommended (balanced / long-context)
python -m serving.openai_compatible_api_gateway `
    --model Qwen/Qwen2.5-1.5B-Instruct `
    --host 0.0.0.0 `
    --port 8000 `
    --serving-mode long-context `
    --rank 16 `
    --batch-size 4
```

### Serving Mode Reference

| Mode           | Context       | VRAM budget | Use case                          |
|----------------|---------------|-------------|-----------------------------------|
| `lightweight`  | Short         | ~0.5 GB     | CPU, testing, low VRAM            |
| `balanced`     | Medium (~4K)  | ~2 GB       | Default interactive chat          |
| `performance`  | Long (~16K)   | ~8 GB       | GPU-rich inference                |
| `long-context` | Very long (32K+)| ~12 GB    | Document/code tasks               |
| `fused-sparse` | Long          | ~8 GB       | Max decode throughput (GPU only)  |

### Optional Environment Variables

```powershell
$env:DIFFKV_TELEMETRY   = "1"          # Enable per-step timing logs
$env:DIFFKV_USE_TORCH_COMPILE = "1"   # torch.compile fusion (requires PyTorch 2.1+)
$env:TOKENIZERS_PARALLELISM = "false"  # Suppress HF warning (set automatically by gateway)
```

### Verify the Server is Running

```powershell
# Should return {"status": "ok"}
Invoke-WebRequest http://localhost:8000/health | Select-Object -ExpandProperty Content

# Should return the model list
Invoke-WebRequest http://localhost:8000/v1/models | Select-Object -ExpandProperty Content
```

---

## Step 2 — Connect Open WebUI

1. Open Open WebUI in your browser (default: `http://localhost:3000`).
2. Click your profile icon → **Settings** → **Connections**.
3. Under **OpenAI API**, click **Add connection** (or edit the existing one).
4. Set:
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: `none` (or any non-empty string — DiffKV ignores auth)
5. Click **Save** and then **Verify Connection**.
6. The model `diffkv-<your-model-name>` will now appear in the model dropdown.

> **Tip**: If Open WebUI is running inside Docker and DiffKV is on the host,
> use `http://host.docker.internal:8000/v1` instead of `localhost`.

---

## Endpoints Exposed

| Method | Path                   | Purpose                                |
|--------|------------------------|----------------------------------------|
| GET    | `/health`              | Health check (used by Open WebUI)     |
| GET    | `/v1/health`           | Alternate health check                |
| GET    | `/v1/models`           | List available models                  |
| GET    | `/models`              | Alias for `/v1/models`                 |
| POST   | `/v1/chat/completions` | Chat inference (streaming + non-streaming)|
| POST   | `/v1/sessions`         | Create a new persistent session        |
| DELETE | `/v1/sessions/{id}`    | Clear session history                  |
| GET    | `/v1/runtime_info`     | Live VRAM and serving mode telemetry   |
| GET    | `/docs`                | Interactive Swagger API docs           |

---

## Troubleshooting

| Problem                             | Fix                                                      |
|-------------------------------------|----------------------------------------------------------|
| Open WebUI shows no models          | Check `/v1/models` returns JSON; verify URL/port         |
| "Connection refused"                | Gateway not started, or port blocked by firewall         |
| CUDA out of memory                  | Switch to `--serving-mode lightweight` or reduce `--rank`|
| Slow first response                 | Normal — model loading + first Triton kernel compilation |
| Triton kernel compilation error     | Falls back to PyTorch automatically; no action needed    |
| Docker: can't reach host gateway    | Use `host.docker.internal` instead of `localhost`        |
