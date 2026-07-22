# DKV Paper Workspace

**Author:** Om Chimurkar — <https://github.com/Omc12/Differential-KV>

Implementation-driven technical report + conference paper for the **DKV active (MLX)
runtime**. Everything regenerates from measured data; no number is hand-typed into the paper
(body-text numbers are LaTeX macros emitted by `scripts/make_facts.py`).

## Build the PDFs (tectonic is used here; pdflatex/xelatex also work)
```
tectonic main.tex          # technical report  (main.pdf, 31 pp)
tectonic conference.tex    # conference version (conference.pdf, two-column)
```
`main.tex` and `conference.tex` share the same `sections/` via the `\ifext` toggle.

## Regenerate facts, figures, graphs, tables from measured JSON
```
dkv_venv/bin/python3 paper/scripts/make_facts.py      # body-text number macros -> generated/facts.tex
dkv_venv/bin/python3 paper/scripts/make_diagrams.py   # F1-F6 architecture/dataflow + 3D memory view
dkv_venv/bin/python3 paper/scripts/make_graphs.py     # G1-G7 measured-data graphs
dkv_venv/bin/python3 paper/scripts/make_tables.py     # T1-T6 LaTeX tables
```

## Re-run the experiments — CLEAN protocol (Apple Silicon + MLX)
One isolated process per cell, **nothing else running** (figure/PDF builds or a second job
contend on the 8 GB machine and contaminate timings). Config = the active CLI's real settings
(preset=mid, serving_mode=balanced, rank 32, int4) with DKV forced on at every context:
```
CTXS="4096 8192 16384 32768" bash paper/scripts/run_clean.sh   # primary active-vs-dense sweep
bash paper/scripts/run_ablations.sh                            # decode-mode + residual sweep + 64k
```
Cell worker: `paper/scripts/cell_worker.py` (states the exact DKV config in each result JSON).

## Layout
- `main.tex` / `conference.tex` — entry points (shared `sections/` via `\ifext`).
- `sections/`, `appendix/`, `preamble.tex` — paper source.
- `scripts/` — `data.py` (single data loader) + `make_facts/diagrams/graphs/tables.py`
  + the CLEAN measurement drivers (`run_clean.sh`, `run_ablations.sh`, `cell_worker.py`).
- `figures/` — F1-F6 (diagrams) + G1-G7 (data), PNG + PDF. **Correctness-regenerated at rank 32;
  a final industry-deck aesthetic pass (blue shades, black body text) is still owed.**
- `tables/` — generated LaTeX tabulars.  `generated/` — measured JSON + `facts.tex`.
- `bibliography/references.bib` — verify arXiv ids before camera-ready.
- `notes/AUDIT_2026-07-07.md` — data-provenance decision; `consistency_report.md` — verification.

## Key honesty notes (clean rebuild, 2026-07-07)
- **Config of record = the active CLI: mid / balanced / rank 32 / int4**, DKV sparse forced on
  (`DKV_COMPRESSED_DECODE=1`) with decode-cache, block-sparse prefill, and adaptive bias on.
  Earlier runs at rank 16 and/or with concurrent load were wrong/contaminated and are superseded.
- **Data of record**: `benchmarks/results/clean_{active,dense}_{ctx}.json` (primary),
  `generated/active_modes_fresh.json` (decode modes), `generated/residual_sweep_fresh.json`.
- **Primary result** = the true compressed sparse-decode path; exact full-KV decode is an
  upper-bound ablation only.
- **Compression ratio (rank 32)**: R=128 -> 1.44x, R=64 -> 2.25x per block (residuals dominate the
  block; an earlier low-rank-only accounting overstated it).
- **Recall**: the buried passcode is recovered exactly at every tested context (4k-64k). The
  residual budget is a memory/speed dial with a wide recall-preserving range (needle held R=8..128).
- **Reach**: DKV reaches 64k (needle intact, 4.63 GB peak); the dense full-KV baseline **OOMs**
  at 64k on the 8 GB host. That OOM boundary is the memory story's sharp end.
- **Cost**: decode throughput. DKV decode (~17-20 tok/s, flat) is slower than dense wherever a
  dense cache still fits (through 32k). Reported in full; localized to reconstruction/merge overhead.
