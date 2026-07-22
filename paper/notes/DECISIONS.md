> **ADDENDUM (2026-06-30): partially superseded — see `AUDIT_2026-06-30.md`.**
> Decisions 1–2 (compressed = primary, exact = upper-bound ablation; mx_peak + decode-phase +
> analytic memory headline) still hold. BUT the premise that compressed decode *misses the needle*
> (a "fidelity limitation / open problem") is **no longer true**: the 2026-06-29 residual fix
> (`max_residual` 32→64 + residual-key router, both now defaults) makes compressed decode recover
> the needle EXACTLY at 4k–64k. The paper reports recall as solved, the residual budget as the
> accuracy/memory knob (E6), and the corrected per-block ratio 2.85× (not 10×; residuals were
> omitted before). Data of record is the clean re-measured `active_modes_sweep_v2.json` +
> `active_modes_sweep_64k.json` + `residual_sweep.json`, not the pre-fix `active_modes_sweep.json`.

# Locked decisions (user, 2026-06-28)

1. **Decode framing = Honest full picture.** Primary subject of the paper is the TRUE
   compressed low-rank sparse-decode path (`DKV_COMPRESSED_DECODE=1`). Report its real
   prefill/throughput/memory AND its current needle-fidelity limitation (rank-16 SVD floor)
   openly as an open problem / limitation. The exact full-KV decode (`=0`) is presented as
   an *upper-bound ablation* (what perfect-fidelity retrieval would give on the same store),
   NOT as the headline DKV result. Re-measure `=1` cleanly across 4k–64k.

2. **Memory headline = mx_peak (MLX allocator peak), re-measured**, PLUS a separate
   steady-state decode-phase memory measurement (reset peak after prefill→decode boundary,
   then measure decode-phase peak) to isolate what compression actually saves vs. the
   prefill-dominated global peak. Also report analytic KV-store bytes (bounded, fixed pool)
   vs. dense full-KV bytes. Process-tree max(phys,RSS) demoted to a secondary/appendix
   metric with the divergence explained.

Consequences:
- Do NOT reuse REPORT.md's 45→17 tok/s "active" numbers as DKV decode results
  (those are the `=0` exact path).
- The clean paired dataset = `paper/generated/active_modes_sweep.json` from
  `paper/scripts/measure_active.py` (this run), + dense from PAPER_dense_sweep.json.
</content>
