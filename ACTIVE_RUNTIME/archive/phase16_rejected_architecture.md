# Phase 16 Rejected Architecture

This document serves as a permanent quarantine list for "Architecture Theater"—systems found in `RESEARCH_PROTOTYPES` and `ARCHIVED_SYNTHETIC_SYSTEMS` that produced compelling telemetry reports or theoretical narratives but proved useless, fake, or mathematically unsound upon physical execution tracing.

## 1. Fake Cognition / Narrative Routing
- **Systems:** `native_cognitive_attention_adapter.py`, `head_role_allocator.py`
- **Why Rejected:** These systems attempted to assign "roles" (e.g., "reasoning", "retrieval", "syntax") to attention heads or regions. They generated extensive telemetry logs claiming semantic awareness. However, execution tracing revealed they either assigned roles randomly, operated on static heuristics, or were completely disconnected from the actual matrix multiplication paths.

## 2. Geometric Token Pruners
- **Systems:** `geometric_token_pruner.py`, `protective_radius_allocator.py`
- **Why Rejected:** Claimed to map tokens into a 3D semantic space and prune "unimportant" geometry. In reality, they computed arbitrary L2 distances between hidden states and randomly dropped tokens, leading to catastrophic hallucination and context collapse without any structured hardware speedup.

## 3. Simulated VRAM & Synthetic Telemetry
- **Systems:** `runtime_density_profiler.py`, `residency_truth_telemetry.py`
- **Why Rejected:** These scripts did not actually manage `torch.cuda` memory allocations. They merely updated Python dictionary counters (e.g., `simulated_vram_used -= 100`) and plotted charts showing "savings," while the underlying PyTorch runtime continued to allocate and hold the full dense tensors.

## 4. Multi-GPU Expert Sharding (Fake Distributed)
- **Systems:** `gpu_affinity_allocator.py`, `distributed_visibility_allocator.py`
- **Why Rejected:** Attempted to simulate distributed tensor parallel execution across multiple GPUs. However, no actual `torch.distributed` NCCL operations were used. They simply moved tensors via `tensor.to('cuda:1')` in a blocking, sequential manner, destroying throughput and generating false claims of "distributed expert execution."

## Final Verdict
Differential KV will move forward exclusively with physically grounded, mathematically proven, and hardware-verified optimizations (such as Async SVD, Block-Sparse Routing, and K/Q Centroid Anchors). Semantic narratives and simulated telemetry are permanently rejected.
