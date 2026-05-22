# Differential KV Cache — Research Prototype

> **Status:** Phase 10 — Real-World Inference Validation & Hardware Grounding (RIVHG) COMPLETED  
> **Goal:** Transition Differential KV into an empirically and scientifically defensible runtime under real workloads.

---

## What This Is

A **scientifically defensible distributed sparse runtime** for experimental transformer KV-cache memory architecture.

This project has reached **Phase 9.75**, focusing on:
- **Metric Reconciliation**: Standardized taxonomy and semantic correctness.
- **Trace Enforcement**: All hardware claims backed by physical kernel traces.
- **Distributed Patching**: Synchronization smoothing and queue backpressure.

---

## Phase 9.75 — Metric Reconciliation & Patch Hardening (MRDPH)

| Component | Status | Description |
|---|---|---|
| Metric Semantics | RECONCILED | Unified taxonomy for all throughput classes. |
| Trace Enforcement | PATCHED | Mandatory hardware-trace backing for occupancy. |
| Sync Hardening | STABLE | Burst-reduction and jitter-aware synchronization. |
| Queue Stability | VERIFIED | Backpressure-controlled scaling up to 32 workers. |
| Real-World Inference| VERIFIED | End-to-end serving validation with real prompts. |
| Hardware Grounding | GROUNDED | Claims backed by physical CUDA traces and counters. |

---

## Project Structure (Phase 9.75)

```
differential-kv/
├── distributed/         # Patched sync logic and backpressure control
├── validation/          # Real inference harness and reproducibility tools
├── profiling/           # Hardware grounding and CUDA trace captures
├── benchmarks/          # Standardized long-context and multi-user suites
├── results/reconstruction_10/ # Real-world validation logs and traces
├── reports/             # Phase 10 validation and grounding reports
```

---

## Running Validation

```bash
# Execute Phase 9.75 Reconciliation & Patch Stack
# Execute Phase 10 Real-World Validation & Grounding
python run_phase_10_validation.py
```

Check `results/reconstruction_9_75/` for patched markdown summaries.
