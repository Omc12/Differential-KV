# Differential KV: Deployment Guide

## Prerequisites
- Python 3.10+
- CUDA 11.8+ (RTX 30 series or newer recommended)
- `pip install -r requirements.txt`

## Local Deployment
1. **Model Preparation**: Ensure your model (Qwen, Llama, Mistral) is downloaded.
2. **Start Gateway**:
   ```bash
   python differential_kv_cli.py --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
   ```
3. **Verify Health**:
   ```bash
   curl http://localhost:8000/v1/models
   ```

## Production Configuration
- **Sparse Ratio**: Adjusted via `--sparse-ratio` (default 0.9).
- **Batch Size**: Controlled dynamically by `LatencyAwareBatchController`.
- **Recovery**: Enabled by default; sessions are saved to `./checkpoints`.

## Hardware Profiles
- **Standard**: Full Triton kernel acceleration.
- **Low-VRAM**: Aggressive token survival and 4-bit quantization.
- **CPU Fallback**: Graceful fallback if no compatible GPU is detected.