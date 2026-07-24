# Differential-KV Documentation Index

Welcome to the **Differential-KV (`dkv`)** documentation repository. Below is the organized directory of technical documentation, architectural specifications, and hardware audit reports.

- [BUILD.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/BUILD.md) — Comprehensive compilation and build instructions for macOS & Linux/CUDA.

---

## 🏛️ Architecture & Handoff Specifications (`docs/architecture/`)

- [SRL_ROUTER_ARCHITECTURE_REFERENCE.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/architecture/SRL_ROUTER_ARCHITECTURE_REFERENCE.md) — Two-level gating, chunk descriptors, and semantic query routing reference.
- [CUDA_DECODE_NEEDLE_FIDELITY_HANDOFF.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/architecture/CUDA_DECODE_NEEDLE_FIDELITY_HANDOFF.md) — Needle recall fidelity, fused decode attention, and CUDA parity handoff document.
- [NATIVE_LEGO_PORT_PLAN.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/architecture/NATIVE_LEGO_PORT_PLAN.md) — Native C++ engine porting and module migration roadmap.

---

## 📊 Audits & Performance Reports (`docs/audits_and_reports/`)

- [CUDA_TRITON_AUDIT.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/audits_and_reports/CUDA_TRITON_AUDIT.md) — Authoritative CUDA/Triton kernel audit, kernel correctness certification, and F1 residual alignment findings.
- [CUDA_VRAM_PERF_FINDINGS_2026-07-17.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/audits_and_reports/CUDA_VRAM_PERF_FINDINGS_2026-07-17.md) — Detailed VRAM footprint analysis and memory scaling profiles under extreme context windows.
- [CUDA_VS_MLX_PERFORMANCE_AUDIT_2026-07-15.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/audits_and_reports/CUDA_VS_MLX_PERFORMANCE_AUDIT_2026-07-15.md) — Comparative benchmark audit comparing Apple Silicon (MLX) vs NVIDIA (CUDA) execution engines.
- [CUDA_VS_MLX_TRACE_DIFF.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/audits_and_reports/CUDA_VS_MLX_TRACE_DIFF.md) — Trace-level execution difference breakdown between Metal and CUDA backends.
- [RELATIONAL_BINDING_REPORT.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/audits_and_reports/RELATIONAL_BINDING_REPORT.md) — Investigation report on relational binding retention and multi-needle recall.

---

## 📜 Development & Session Logs (`docs/logs/`)

- [ANTIGRAVITY_LOG_2026-07.md](file:///Users/omchimurkar1/Desktop/Differential-KV/docs/logs/ANTIGRAVITY_LOG_2026-07.md) — Development trajectory, milestone logs, and system changelogs.
