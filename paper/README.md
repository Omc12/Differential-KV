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
```
diffkv_venv/bin/python3 paper/scripts/measure_active.py \
    --ctx 4096 8192 16384 32768 65536 --modes compressed exact --gen 128 \
    --out paper/generated/active_modes_sweep.json
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
- Memory headline = MLX allocator peak + analytic KV-state footprint; the compressed path's
  needle-recall fidelity limitation (rank-16) is reported, not hidden.
