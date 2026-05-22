# Phase 18.5 Reconstruction Index

This directory contains the [MEASURED] scientific reports and raw artifacts for Phase 18.5: **Semantic Geometry & Continuity Preservation (SGCP)**.

## Core Reports
1. [reconstruction_18_5_semantic_geometry.md](./reconstruction_18_5_semantic_geometry.md) - Summary of continuity recovery.
2. [reconstruction_18_5_continuity_preservation.md](./reconstruction_18_5_continuity_preservation.md) - Analysis of distal instruction following.
3. [reconstruction_18_5_compute_memory_balance.md](./reconstruction_18_5_compute_memory_balance.md) - Overhead vs. Recall metrics.
4. [reconstruction_18_5_failure_analysis.md](./reconstruction_18_5_failure_analysis.md) - Analysis of retrieval blur and NIAH decay.

## Raw Artifacts
- `raw_retrieval_accuracy.jsonl` - Token traces of the 8k/16k benchmarks.
- `raw_wallclock_trace.log` - GPU telemetry and prefill timing.

## System Components
- `memory/semantic_geometry_tracker.py`
- `runtime/adaptive_chunk_overlap.py`

**Status: ALL [MEASURED] ARTIFACTS VERIFIED**
