# Phase 18.6 Reconstruction Index

This directory contains the [MEASURED] scientific reports and raw artifacts for Phase 18.6: **Symbolic Fidelity & Anchor-Relative Memory Recovery (SFARMR)**.

## Core Reports
1. [reconstruction_18_6_symbolic_fidelity.md](./reconstruction_18_6_symbolic_fidelity.md) - Summary of exact-string prefix recovery.
2. [reconstruction_18_6_hybrid_retrieval.md](./reconstruction_18_6_hybrid_retrieval.md) - Analysis of compute overhead vs. symbolic precision.

## Raw Artifacts
- `raw_retrieval_accuracy.jsonl` - Token traces showing "ALPHA" recovery at 8k context.
- `raw_wallclock_trace.log` - GPU telemetry for the 8k budget run.

## System Components
- `memory/symbolic_fidelity_registry.py` - High-entropy token detection.
- `runtime/hybrid_memory_resolver.py` - Neighborhood-aware symbolic pinning.

**Status: ALL [MEASURED] ARTIFACTS VERIFIED**
