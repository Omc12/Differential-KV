# PHASE 18.7 — HIERARCHICAL MEMORY CAPSULES & SYMBOLIC FIDELITY RECOVERY (HMCSFR)

## 1. Overview
Phase 18.7 investigates the use of **Hierarchical Memory Capsules (HMCs)** to bridge the gap between semantic coherence and symbolic precision in sparse KV caches. By linking high-fidelity memory regions to semantic anchors, the system aims to preserve exact identifiers (IDs, code, APIs) without collapsing compute efficiency.

## 2. Directory Structure
- `reconstruction_18_7_memory_capsules.md`: Analysis of HMC creation and registry performance.
- `reconstruction_18_7_symbolic_fidelity.md`: [MEASURED] scores for exact retrieval at 4k/8k/16k.
- `reconstruction_18_7_geometry_preservation.md`: Analysis of semantic pathway continuity.
- `reconstruction_18_7_precision_tiers.md`: Breakdown of dynamic budget allocation.
- `reconstruction_18_7_compute_balance.md`: TPS vs. Fidelity correlation.
- `reconstruction_18_7_failure_analysis.md`: Detailed breakdown of symbolic degradation modes.

## 3. Raw Artifacts
- `raw_capsule_registry.jsonl`: Trace of all active capsules.
- `raw_symbolic_retrieval.jsonl`: Exact retrieval trial data.
- `raw_precision_allocations.jsonl`: Tiered budget distributions.
- `raw_geometry_paths.jsonl`: Graph traversal data.
- `raw_compute_overheads.jsonl`: VRAM and TPS telemetry.

## 4. Status
**IMPLEMENTATION**: [COMPLETE]
**VALIDATION**: [COMPLETE]
**MEASUREMENT**: [COMPLETE]
