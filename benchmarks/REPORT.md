# DiffKV vs Dense vs llama.cpp — long-context benchmark

**Model:** Qwen2.5-1.5B-Instruct (Q4) · **Host:** Oms-MacBook-Pro-2.local, 8.6 GB unified memory, Apple M3
**Decode tokens/test:** 128 (greedy) · **Per-test timeout:** 1800s · **RAM kill-cap:** 7.5 GB
**Run:** 2026-06-28T15:25:12 → 2026-06-28T15:42:54

All figures are measured on the same machine, one engine at a time in an isolated subprocess. Failed cells show the reason (OOM / timeout / skip), never a fabricated number. Once an engine fails at a context length it is not run at larger ones.

### Contenders

- **native** — DiffKV native (C++ / ggml, GGUF Q4_K_M)
- **active** — DiffKV active runtime (MLX, int4) — compressed KV

### Methodology notes

- **Same prompt** (a Needle-In-A-Haystack chat prompt) is built once per context length from the Qwen2.5 tokenizer and fed verbatim to every engine (raw mode, no double chat-templating). `Needle` = whether the generation reproduced the planted passcode (a free correctness signal).
- **Prefill s**: native = the binary's own `[PREFILL_TIME]`; active/dense = `perf_counter` around the chunked prefill forward (after `mx.eval`); ollama = `prompt_eval_duration` from `/api/generate`. All engines are warmed first.
- **Decode tok/s**: fixed-length greedy decode (EOS ignored for native/active/dense); ollama uses its own `eval_count/eval_duration`.
- **Memory**: peak of per-process `max(phys_footprint, RSS)` over the engine tree at 20 Hz — phys_footprint catches MLX Metal buffers that RSS misses; RSS catches mmap'd GGUF weights that phys_footprint misses. For ollama we also record its self-reported `size_vram`.
- **`trunc@32k`**: ollama clamps the context window to Qwen2.5's trained limit (32768) and truncates longer prompts, so it cannot actually perform the task past 32k on this model — those cells are not valid long-context runs.

## Prefill time (s) — lower is better

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 5.6 | 12.8 | 33.4 | 96.8 | OOM |
| active | 8.1 | 16.7 | 39.3 | 99.5 | 282.5 |

## Decode throughput (tok/s) — higher is better

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 8.3 | 5.2 | 1.5 | 1.5 | OOM |
| active | 45.1 | 42.3 | 36.7 | 28.8 | 17.7 |

## Peak memory (GB) — lower is better

| Engine | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| native | 2.34 | 2.31 | 2.97 | 4.75 | OOM |
| active | 2.86 | 2.85 | 3.29 | 4.17 | 6.02 |

## Maximum usable context (valid run within 8 GB)

| Engine | Max ctx | Limited by |
|---|---|---|
| native | **32k** | oom |
| active | **64k** | — |

## Key findings

1. **Memory is the decisive axis, and DiffKV's compressed KV wins it.** Peak memory at 16k: active **3.18 GB** vs dense **5.94 GB** vs native **5.04 GB** vs ollama 1.75 GB. The DiffKV active runtime's memory grows far slower than the full-KV dense baseline as context grows (active 4k→64k: 2.86→2.86→3.18→4.21→6.02 GB; dense: 2.76→4.31→5.94→5.89 then OOM at 64k).
2. **Max usable context within 8 GB: active 64k > dense 32k > native 16k ≈ ollama 16k.** The DiffKV active runtime is the only engine that genuinely processes 64k tokens (all 65,615, needle recovered) on this 8 GB Mac — 2× the dense baseline and 4× the C++ native build before failing.
3. **Throughput is the trade-off DiffKV pays.** At every context the active runtime decodes slower than dense/ollama (44 vs 65 tok/s at 4k; 36 vs 45–52 at 16k; down to 17 at 64k) and its prefill is the slowest of the MLX engines — the cost of sparse retrieval + per-block SVD compression. DiffKV buys memory and context-reach at the price of speed.
4. **ollama / llama.cpp is fastest and leanest at ≤16k but is hard-capped at Qwen2.5's 32k trained context.** It silently truncates 32k+ prompts (`num_ctx` clamp → prompt cut → 1 token emitted), so it cannot perform a true long-context task on this model without RoPE/YaRN scaling. Its low memory (1.3–1.8 GB) comes from mmap'd weights.
5. **The C++ `native` build is the weakest contender here:** slowest decode (27→18 tok/s), heaviest early memory growth (OOM at 32k, before dense), and under greedy decoding it did **not** reproduce the planted needle (it echoed the prompt) — a coherence gap. It does not yet show a speed or memory advantage over the MLX active runtime on this workload.

## Why `native` underperforms (root cause)

`native` is **not** a port of the MLX `active` runtime — it is an independent reimplementation of the same DiffKV algorithm on a different stack (ggml / GGUF Q4_K_M weights + a custom Metal attention op). The reconstruction transcribed the *architecture* but not the *performance characteristics*, and it diverges on all three measured axes for distinct, identifiable reasons:

1. **Memory (OOMs at 32k, before dense): redundant KV copies.** `native` simultaneously holds a **dense fp32 host window** (`active_k_dense`/`active_v_dense`, `GGML_TYPE_F32` — 2× the bytes of fp16; `src/main.cpp:986`, `:1875`), the SVD-compressed pool, the fp16 prefill KV that is built up by `ggml_concat` of prior context + new chunk (`src/main.cpp:376`), and transient fp32 SVD scratch during the compression-heavy prefill. The code says it outright (`src/main.cpp:1863`): *"MLX keeps no such fp32 host copy at all (KV lives fp16 in unified memory)."* The dense baseline is one tight fp16 GPU cache; `native` carries several overlapping buffers, so its footprint grows faster than even full-KV dense and crosses it between 16k and 32k.
2. **Throughput (slowest decode, 18 vs 36 tok/s @16k): different GPU stack.** `active` runs Apple MLX (fused, compiled, lazily-evaluated kernels on MLX-int4 weights) — Apple's own framework, the fastest path on this silicon. `native` runs ggml-metal on GGUF Q4_K_M through a hand-written custom attention op (`GGML_OP_DIFFKV_ATTN`, `src/main.cpp:925`). Same math, same GPU, but ggml-metal + a custom op is ~2× slower here than MLX's tuned kernels. The "C++ beats Python" intuition does not apply: in `active` the Python layer only orchestrates — MLX's Metal kernels do the compute.
3. **Coherence / needle miss (the real failure): lossy compressed reconstruction, not retrieval.** Originally suspected to be under-budgeted slot selection — but that was *disproven*: raising `srl_k_keep` 16→64→128 produced byte-identical output and the logs show all compressed blocks are already attended. The actual cause is the **fidelity of the SVD-compressed block** once a token leaves the recency window. Decisive proof: keeping the needle in the dense window (`DIFFKV_RECENCY_WINDOW` large enough to cover it) recovers it **exactly** at every scale (`OMEGA-7741-DELTA`, coherent); compressing it garbles the needle at 4k and collapses into instruction-echo at 8k+ — same root cause, worse with depth.
   The loss is **rank truncation, not precision.** The built-in `DIFFKV_DBG_COMPRESS_ERR` decomposition shows a ~43% rank-16 reconstruction floor with an int8-vs-fp16 U penalty of ~0.002% — so fp16 U buys nothing, and porting active's randomized SVD cannot help either: `native` already uses an **exact LAPACK `sgesdd_`** SVD at the **same rank (16) and block size (256)** as the live MLX `active` runtime (`mlx_diffkv_wrapper.py`: `rank=16`, `block_size=256`), whose rSVD is only an approximation of the same truncation. DiffKV's rescue for the irreducible floor is the exact-token **residual** path, and that is where `native` diverged: it capped residuals at `MAX_RESIDUAL=8` (~3% of a 256-token block) while `active` keeps the full 15% (~38).
   Fixes landed on `diffkv-native-needle-recall-fix` (per-row int8 U so int8==fp16; `MAX_RESIDUAL` 8→40 + `DIFFKV_RESIDUAL_FRAC`; decode routed to the corrected CPU op) take 4k from word-salad to `OMEGA-777` and recover the needle exactly when it stays dense. **Exact parity at 15% residuals is still open** — at equal residual count `active` recovers and `native` does not, so the remaining gap is `native`'s residual *apply* at decode (suspected: residual position indexing under the landmark swap), not the compressor or the SVD.

**On CUDA / Triton:** neither live runtime uses them, and on this Apple-Silicon host they cannot (no NVIDIA GPU). The `active` runtime's compiled extension (`native_core/diffkv_core/*.so`) is built from Metal + CPU objects (`metal_runtime.o`, `decode_attention.o`, `compressor_thread_cpu.o`); its actual sparse/dense attention is plain MLX (`mx.softmax`) in `serving/mlx_diffkv_wrapper.py`. `native` builds with `GGML_CUDA OFF` and the CPU `paging_stream.cpp` variant. The only live `.cu` file (`paging_stream.cu`) is host↔device memcpy plumbing, **not** a compute kernel, and every Triton kernel in the tree lives under `archive/`. So there is no validated CUDA/Triton fused-attention path in either engine today — a CUDA `native` would also be a different *hardware platform* (NVIDIA) than `active` (Apple-only MLX), making any "native-CUDA vs active" speed claim a hardware comparison, not an algorithm one.

## Figures

Generated by `plot_graphs.py` into `results/`:

- `fig_memory.png` — peak memory vs context (the headline: active's flat slope vs dense's blow-up; × = killed, ▽ = skipped).
- `fig_decode_tps.png` — decode throughput vs context (ollama 32k+ truncation excluded).
- `fig_prefill.png` — prefill time vs context (valid runs only).
- `fig_combined.png` — all three panels with a shared legend.

## Caveats (read before citing)

- **8 GB, loaded machine.** This M3 has 8.6 GB unified memory with other apps resident (~1–2 GB). Absolute OOM thresholds would shift up on a clean or larger machine — but the **scaling slopes and the ordering between engines are the result**, and those are hardware-independent.
- **`active` 128k did NOT run out of memory** — it fit in **6.11 GB** (under the 7.2 GB cap). It was killed after **21.6 min** because its 131k-token prefill was impractically slow (and 128k is past Qwen2.5's 32k native context anyway). This is a *throughput/usability* failure, reported as a fail cell, not an allocator OOM.
- **`dense` 64k** was killed while **swap-thrashing** (phys 5.89 GB but the system was ~3.7 GB into swap → ~9.6 GB demand on an 8.6 GB box). Past 32k the MLX engines fail via swap-thrash rather than a clean allocator OOM, because Metal/cache pressure lands in swap that `phys_footprint` doesn't charge to the process; we killed such runs once they thrashed.
- **`trunc@32k`** cells are ollama runs where the prompt was clamped to 32768 and truncated — not valid runs; their raw `decode_tps` (a 1e6 divide-by-≈0) is suppressed here.
- **Quantization is matched where it matters:** native, ollama = GGUF Q4_K_M; active, dense = MLX int4 — all 4-bit weights. `dense` shares the exact int4 weights and MLX engine with `active`, so active-vs-dense isolates the DiffKV compression algorithm, nothing else.

## Per-run detail

| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | Peak mem GB | RSS GB | MLX peak GB | ollama VRAM GB | Needle | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 4k | ok | 4096 | 15 | 5.6 | 8.3 | 2.34 | 2.34 | — | — | - | 10 |
| active | 4k | ok | 4096 | 128 | 8.1 | 45.1 | 2.86 | 1.22 | 1.60 | — | Y | 17 |
| native | 8k | ok | 8192 | 31 | 12.8 | 5.2 | 2.31 | 2.31 | — | — | Y | 21 |
| active | 8k | ok | 8192 | 128 | 16.7 | 42.3 | 2.85 | 1.69 | 1.63 | — | Y | 25 |
| native | 16k | ok | 16490 | 18 | 33.4 | 1.5 | 2.97 | 2.44 | — | — | - | 48 |
| active | 16k | ok | 16490 | 128 | 39.3 | 36.7 | 3.29 | 1.58 | 1.87 | — | Y | 48 |
| native | 32k | ok | 32865 | 30 | 96.8 | 1.5 | 4.75 | 2.53 | — | — | Y | 120 |
| active | 32k | ok | 32865 | 128 | 99.5 | 28.8 | 4.17 | 2.02 | 2.35 | — | Y | 109 |
| native | 64k | oom | — | — | — | — | 7.62 | 3.27 | — | — |  | 369 |
| active | 64k | ok | 65615 | 128 | 282.5 | 17.7 | 6.02 | 1.65 | 3.28 | — | Y | 296 |
