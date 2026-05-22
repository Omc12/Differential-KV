# Phase 21 Native Failure Modes

This document analyzes the realistic failure modes that occur under extreme, production-level serving stress.

## 1. Graph Invalidation Storms
**Scenario:** 100 concurrent users are generating text. Users randomly finish generating and drop out of the batch at a rate of 5 users per second.
**Failure Mode:** `StaticSparseDecodeGraph` requires a fixed batch size. Every time a user drops out, the PyTorch engine must recapture the CUDA graph, taking ~15ms. In a storm, this cascades, dropping overall TPS by >60%. 
**Solution:** vLLM solves this via "Padded Graph Capturing" (capturing graphs for batch sizes 16, 32, 64, etc., and padding inputs). Differential KV must adopt this vLLM feature natively.

## 2. Compression Backlog Overflow
**Scenario:** A massive influx of long-context prompts (e.g., ten 100K-token prompts simultaneously) triggers thousands of block compression requests.
**Failure Mode:** The `AsyncCompressor` queue overflows. The system falls back to keeping everything in the Dense Recency Window, which rapidly exhausts GPU VRAM, forcing immediate paging to CPU RAM. PCIe bandwidth is saturated, causing decode stalls across all active users.
**Solution:** The vLLM scheduler must become "Compression Aware." It cannot schedule massive prefill bursts if the compression worker queue is >80% full.

## 3. Paging Jitter (Mixed Compressed/Uncompressed Decode)
**Scenario:** A session requires attention over both Dense blocks (in VRAM) and Compressed blocks (reloading from CPU RAM).
**Failure Mode:** The synchronous Python `tensor.to(device)` call used to fetch the paged block blocks the entire GPU decode stream.
**Solution:** Must be replaced by vLLM's asynchronous block prefetcher (which uses dedicated CUDA memory streams).

## Verdict
Differential KV does not fail mathematically under stress; it fails *mechanically* when Python orchestration bottlenecks are exposed. Full vLLM integration solves these exact mechanical failures natively.
