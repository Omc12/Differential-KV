# Phase 16 Native Runtime Candidates

This document outlines the specific systems from the Differential KV repository that have proven their mathematical validity and hardware-level value, but explicitly require native C++ / vLLM integration to escape Python orchestration bottlenecks.

## 1. Fused Anchor-Routed Sparse Attention (Sparse Prefill)
- **Concept:** Uses heavily compressed chunk centroids (K/Q pooled means) to globally route attention during long-context prefill without materializing an $O(N^2)$ memory footprint.
- **Why it failed in Python:** 32 sequential chunk loops caused 120ms of overhead. Fusing it via `flex_attention` and `torch.compile` failed because Triton heuristic compilation exhausted the 100KB per-block SRAM limit on consumer hardware.
- **Native Solution Required:** A custom C++ FlashAttention-3 style kernel that accepts explicit block-sparse boolean masks and executes the full sequence in one pass without dynamic compilation.

## 2. Asynchronous Tiered FFN Loading (Sparse Transformer)
- **Concept:** Evaluates the `gate_proj` activation magnitude on early chunks, determines the active FFN blocks, and pages them in from CPU to GPU dynamically for the late-stage transformer layers, saving ~5GB VRAM.
- **Why it failed in Python:** Synchronous `tensor.to(device)` blocking operations in the `forward()` pass stall the GPU pipeline for 0.1–2.0ms during unpredictable `seq=1` decoding or sudden topic shifts.
- **Native Solution Required:** A background C++ thread managing dedicated CUDA streams for asynchronous `HostToDevice` memory copies, allowing weights to stream in concurrently while the GPU computes the preceding layers.

## 3. Fused Sparse MLP Execution
- **Concept:** Uses Triton kernels to compute ONLY the active experts/blocks within the FFN layers during prefill, skipping up to 50% of the FLOPs.
- **Why it failed in Python:** PyTorch `index_select` overhead erases the math savings. While the Triton kernel `sparse_mlp_fused.py` works, wiring it into the HuggingFace `Qwen2MLP` forward pass dynamically creates excessive Python dispatch overhead.
- **Native Solution Required:** Direct integration into the vLLM custom operations library (e.g., alongside `vllm.model_executor.layers.fused_moe`), bypassing Python dispatch entirely.
