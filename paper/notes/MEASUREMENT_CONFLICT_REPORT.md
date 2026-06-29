# BLOCKING: Measurement-integrity conflict in the "active" benchmark

**Status:** Phase 4 STOP gate. Do not write results until resolved.
**Date:** 2026-06-28

## What I found

The `benchmarks/REPORT.md` headline "active" numbers are **not** produced by DiffKV's
novel compressed sparse-decode kernel. They are produced by an **exact full-KV decode**
path. The actual compressed decode (the paper's core contribution) behaves very
differently. The selector is the env var `DIFFKV_COMPRESSED_DECODE`
(default `"1"` in code), read in `attention_forward` (`mlx_diffkv_wrapper.py:860`).

The working tree (`git diff HEAD`) shows `mlx_diffkv_wrapper.py` is the *newer*
compressed-decode implementation; committed HEAD was the older exact-decode one.

## Hard numbers I just measured (Apple M3, 8.6 GB, Qwen2.5-1.5B int4, same NIAH prompts)

### 4k (gen=48)
| config | decode tok/s | needle | output |
|---|---|---|---|
| `=0` exact full-KV decode  | **42.2** | **✓** | "OMEGA-7741-DELTA" |
| `=1` compressed sparse decode | **7.9** | **✗** | garbled "Omega-1" |

Probe B (`=0`) reproduces REPORT.md active-4k (45.1 tok/s, needle Y, identical text)
→ **REPORT's "active" = exact decode (`=0`).**

### 16k (gen=64), process-tree peak memory via orchestrator
| config | prefill s | decode tok/s | peak mem GB (max phys/rss) | MLX peak GB | needle |
|---|---|---|---|---|---|
| active `=0` exact     | 40.8 | 34.4 | 3.34 | **2.03** | ✓ |
| active `=1` compressed| 40.3 | 8.1  | 3.66 | **2.03** | ✗ |
| dense (full KV)       | 27.4 | 47.3 | 5.89 | **2.03** | ✓ |

## Two distinct integrity problems

**P1 — Decode path attribution.** The headline throughput + needle-correctness come from
exact full-KV decode, where DiffKV's low-rank compressed reconstruction is *not exercised
at decode time*. The genuine compressed decode is ~5× slower and currently **misses the
needle** (rank-16 SVD fidelity floor — corroborated by prior native needle-recall notes).
We cannot present the `=0` throughput/correctness as evidence for the compressed algorithm.

**P2 — Memory headline may be a metric artifact.** The trustworthy allocator-true metric
(`mx.get_peak_memory`) is **identical (2.03 GB) for active and dense at 16k**, because the
peak is set during *prefill* (full attention over the growing native cache), which both do
identically. The "active << dense" headline rests on the process-tree `max(phys,RSS)`
metric, which is internally inconsistent here (dense plateaus at 5.89 GB for both 16k and
32k; active's pure-MLX vs active's torch+MLX baselines don't line up with identical
mx_peak). The compression reduces *steady-state decode* memory, not necessarily *peak*.

## Why this matters for the paper

The paper's thesis ("low-rank compressed KV → long context in low memory") is only
honestly supported if we report the **compressed** path's real numbers, including its
current limitations, and pin down a memory metric that actually reflects the compression.
Mixing the `=0` decode numbers with a "compressed KV" narrative would be misattribution.

## Options for the user (see question)

A. Honest full picture (recommended): primary subject = true compressed decode (`=1`),
   report real mem/throughput + needle-fidelity limitation as an open problem; include
   exact decode (`=0`) as an upper-bound ablation. Re-measure `=1` cleanly across contexts
   (needs fresh sweep; some cells will show needle-miss — reported, not hidden).
B. Frame around the `=0` config "as benchmarked", compression = prefill-side memory mgmt;
   compressed decode = future work. (Weaker novelty; arguably mislabeled.)
C. Present `=0` vs `=1` as co-equal design points / ablation throughout.
D. Pause writing; first debug compressed-decode needle fidelity (rank/residuals) to try to
   make `=1` correct, then write.
</content>
