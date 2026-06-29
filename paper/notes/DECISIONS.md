# Locked decisions (user, 2026-06-28)

1. **Decode framing = Honest full picture.** Primary subject of the paper is the TRUE
   compressed low-rank sparse-decode path (`DIFFKV_COMPRESSED_DECODE=1`). Report its real
   prefill/throughput/memory AND its current needle-fidelity limitation (rank-16 SVD floor)
   openly as an open problem / limitation. The exact full-KV decode (`=0`) is presented as
   an *upper-bound ablation* (what perfect-fidelity retrieval would give on the same store),
   NOT as the headline DiffKV result. Re-measure `=1` cleanly across 4k–64k.

2. **Memory headline = mx_peak (MLX allocator peak), re-measured**, PLUS a separate
   steady-state decode-phase memory measurement (reset peak after prefill→decode boundary,
   then measure decode-phase peak) to isolate what compression actually saves vs. the
   prefill-dominated global peak. Also report analytic KV-store bytes (bounded, fixed pool)
   vs. dense full-KV bytes. Process-tree max(phys,RSS) demoted to a secondary/appendix
   metric with the divergence explained.

Consequences:
- Do NOT reuse REPORT.md's 45→17 tok/s "active" numbers as DiffKV decode results
  (those are the `=0` exact path).
- The clean paired dataset = `paper/generated/active_modes_sweep.json` from
  `paper/scripts/measure_active.py` (this run), + dense from PAPER_dense_sweep.json.
</content>
