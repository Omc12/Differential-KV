# Paper Outline — derived from the implementation

**Title (working):** *DiffKV: Anchor + Low-Rank Differential KV-Cache Compression for
Long-Context Inference on Commodity Unified-Memory Hardware*

**Thesis (what the code actually does and the data actually shows):** compressing each
KV block to an anchor token plus a rank-16 joint SVD delta, paired with a small dense
recency window and a fused decode kernel that scores queries in low-rank space without
ever decompressing KV, lets a 1.5B model process 64k tokens (needle recovered) in ~6 GB
on an 8 GB Apple M3 — 2× the dense full-KV context reach and a far flatter memory slope
— at the cost of decode throughput and prefill time.

Section order is dictated by the execution flow: compression → memory layout → prefill →
decode kernel → evaluation.

1. **Abstract**
2. **Introduction** — KV memory wall; long context on commodity HW; contributions.
3. **Background & Motivation** — KV cost arithmetic; design space (quantize / evict /
   low-rank); unified memory & MLX; why anchor+delta low-rank.
4. **System Overview** — components & data flow (Fig: architecture). Active=MLX provenance.
5. **Differential KV Compression** — anchor decomposition, delta normalization, joint
   K/V randomized truncated SVD, adaptive rank, fp16 storage (Fig: compression pipeline;
   Algorithm 1; Table: per-block byte budget & ratio). Memory complexity.
6. **Runtime & Memory Architecture** — session state; chunked prefill w/ exact causal
   cache + streaming capture/compress; prefill→decode memory release (Fig: memory layout;
   Fig: cache lifecycle / execution flow; Algorithm 2).
7. **Fused Sparse Decode Attention** — low-rank score reconstruction; anchor+delta
   value reconstruction; dense branch; flash-style LSE merge; numerical robustness;
   mx.compile fusion (Fig: decode dataflow; Algorithm 3). Compute complexity.
8. **Implementation** — MLX monkeypatch, NumPy rSVD rationale, dtypes, presets, env
   knobs; optional relational-binding/SRL module (gated off in these experiments).
9. **Evaluation** — setup (HW/model/NIAH/memory metric); RQs:
   (E1) memory scaling vs context (Fig: memory), (E2) max usable context within 8 GB,
   (E3) prefill (Fig: prefill), (E4) decode throughput (Fig: decode),
   (E5) needle correctness. active vs dense (same engine/weights = clean ablation).
   (Tables: main results; per-run detail.)
10. **Analysis & Discussion** — what DiffKV buys vs costs; prefill-SVD cost; flat-slope
    explanation tied to O(r) per token; when DiffKV wins.
11. **Limitations & Future Work** — throughput gap; CUDA/Triton fused decode (placeholder
    for future eval, no fabricated numbers); larger models; quality beyond NIAH; SRL eval.
12. **Conclusion**
13. **References**
14. **Appendix A: Reproducibility** — env, versions, exact commands, protocol.
15. **Appendix B: Full measured tables** — generated from JSON.
16. **Appendix C: Notation & algorithms.**

## Figures (all original, consistent identity: white bg, dark-blue #1F3A5F accents,
##   light-blue #4F8FD0 highlights, gray grid)
- F1 architecture / component dataflow (TikZ-style, matplotlib)
- F2 compression pipeline (block → anchor + ΔKV → normalize → rSVD → U,V,anchor)
- F3 session memory layout (dense window + compressed pool, exploded)
- F4 cache lifecycle / execution flow (prefill chunk → capture → flush/compress →
     prefill→decode release → decode ingest → fused attn)
- F5 fused decode attention dataflow (sparse low-rank branch + dense branch → LSE merge)
- G1 peak memory vs context (active vs dense; OOM markers)        [data]
- G2 prefill time vs context                                      [data]
- G3 decode throughput vs context                                 [data]
- G4 combined 3-panel + needle correctness annotation             [data]
- G5 (optional) memory-savings / compression-ratio bar            [data/derived]
- CUDA placeholder panel reserved in G3/G4 design.

## Tables
- T1 model & DiffKV configuration (from code)
- T2 per-block storage budget & compression ratio (derived from dims)
- T3 main results: active vs dense across 4k–64k (prefill, tps, mem, needle) [data]
- T4 reproducibility env/versions
- T5 (appendix) full per-run detail [data]

## Data provenance (single, self-consistent paired dataset)
- active: benchmarks/results/PAPER_active_sweep.json (2026-06-28 re-run, 4k–64k)
- dense:  benchmarks/results/PAPER_dense_sweep.json  (2026-06-28 re-run, 4k–32k; OOM 64k)
- Both: same host (Apple M3, 8.6 GB), same NIAH prompts, same memory metric
  (max(phys_footprint,RSS) over process tree @20Hz), gen=128 greedy.
</content>
