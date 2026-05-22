# Phase 19 Archival Boundary

This document establishes the permanent quarantine boundary for systems located within `ARCHIVED_SYNTHETIC_SYSTEMS`. These systems must NEVER be reintroduced into `ACTIVE_RUNTIME` or `native_core`.

## 1. What Was Rejected
- **Semantic Routing Myths:** `native_cognitive_attention_adapter.py`, `head_role_allocator.py`
- **Fake Distributed Execution:** `gpu_affinity_allocator.py`, `distributed_visibility_allocator.py`
- **Simulated Hardware VRAM Trackers:** `runtime_density_profiler.py`, `residency_truth_telemetry.py`
- **Random Token Geometry Pruners:** `geometric_token_pruner.py`, `protective_radius_allocator.py`

## 2. Why They Were Rejected
These systems were categorized as "Architecture Theater." During exhaustive execution tracing, we discovered they lacked physical hardware reality. They generated complex logs and JSON reports claiming massive efficiency gains, but mathematically bypassed actual PyTorch tensor allocations or substituted dense execution for simulated counts.

## 3. The Failure Pattern Recognized
The primary failure pattern of the repository prior to Phase 15 was the **Telemetry Illusion**. It became easier to script python dictionaries tracking "theoretical" memory savings than to write the necessary CUDA/Triton kernels to physically execute them. Any new system proposed for `native_core` must now prove physical tensor allocation/deallocation and pass hardware profiler traces.
