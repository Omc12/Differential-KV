# Differential KV Cache — Research Prototype

> **Status:** Phase 9.75 — Metric Reconciliation & Distributed Patch Hardening (MRDPH) COMPLETED  
> **Goal:** Patch and harden the distributed runtime, reconcile inconsistent metrics, and enforce trace-backed hardware claims.

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

---

## Project Structure (Phase 9.75)

```
differential-kv/
├── distributed/         # Patched sync logic and backpressure control
├── validation/          # Semantic reconcilers and trace enforcers
├── results/reconstruction_9_75/ # Patched validation reports and logs
```

---

## Running Validation

```bash
# Execute Phase 9.75 Reconciliation & Patch Stack
python run_phase_9_75_validation.py
```

Check `results/reconstruction_9_75/` for patched markdown summaries.
