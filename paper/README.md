# DiffKV Paper Workspace

Implementation-driven technical report + conference paper for the **DiffKV active (MLX)
runtime**. Everything regenerates from measured data; no number is hand-typed into the paper.

## Build the PDFs (needs a LaTeX toolchain — e.g. Overleaf or a local TeX Live)
```
pdflatex main && bibtex main && pdflatex main && pdflatex main          # technical report
pdflatex conference && bibtex conference && pdflatex conference && pdflatex conference
```
(The development machine had no system LaTeX; sources compile with pdflatex/xelatex.)

## Regenerate figures, graphs, tables from measured JSON
```
diffkv_venv/bin/python3 paper/scripts/make_diagrams.py   # F1–F5 architecture/dataflow
diffkv_venv/bin/python3 paper/scripts/make_graphs.py     # G1–G5 from measured JSON
diffkv_venv/bin/python3 paper/scripts/make_tables.py     # T2/T3/T5 LaTeX tables
```

## Re-run the experiments (Apple Silicon + MLX)
Run with nothing else competing for CPU — the per-block NumPy SVD during prefill is CPU-bound,
and a concurrent process roughly doubles prefill time:
```
zsh paper/scripts/run_paper_measurements.sh   # ablation (4k-32k) + 64k reach + residual sweep
diffkv_venv/bin/python3 benchmarks/run_bench.py --engines dense \
    --contexts 4096 8192 16384 32768 --gen 128 --ram-cap-gb 7.5
```

## Layout
- `main.tex` / `conference.tex` — entry points (shared `sections/` via `\ifext`).
- `sections/`, `appendix/`, `preamble.tex` — paper source.
- `scripts/` — figure/graph/table generators + the instrumented measurement harness.
- `figures/` — F1–F5 (diagrams) + G1–G5 (data), PNG + PDF.
- `tables/` — generated LaTeX tabulars.
- `generated/` — measured JSON (the data of record).
- `bibliography/references.bib` — verify identifiers before camera-ready.
- `notes/` — architecture reconstruction, the measurement-conflict finding, locked decisions.
- `paper_outline.md`, `TODO.md`, `consistency_report.md`.

## Key honesty notes
- The paper's **primary** result is the true compressed sparse-decode path; the exact
  full-KV decode is an **upper-bound ablation** only. (An earlier benchmark conflated them;
  see `notes/MEASUREMENT_CONFLICT_REPORT.md`.)
- **Compression ratio is ~2.85x per block, not 10x.** The exact residual tokens (64 per block)
  dominate the per-block bytes; an earlier accounting omitted them. The 10x figure was wrong.
- **Needle recall is solved, not a limitation.** The residual mechanism (top-error exact tokens
  + residual-key router, added 2026-06-29) recovers the buried passcode exactly at every tested
  context. The primary limitation is now decode throughput (kernel dispatch), not fidelity.
- Memory headline = MLX allocator peak + decode-phase peak + analytic KV-state footprint;
  process-tree memory is secondary. The residual-budget/recall trade-off is characterized (E6).
- Data of record: `generated/active_modes_sweep_v2.json`, `active_modes_sweep_64k.json`,
  `residual_sweep.json` (this clean re-run), + `benchmarks/results/PAPER_dense_sweep.json`.
