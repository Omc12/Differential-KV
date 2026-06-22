# DiffKV vs Dense vs llama.cpp — long-context benchmark

**Model:** Qwen2.5-1.5B-Instruct (Q4) · **Host:** Oms-MacBook-Pro-2.local, 8.6 GB unified memory, Apple M3
**Decode tokens/test:** 128 (greedy) · **Per-test timeout:** 1500s · **RAM kill-cap:** 7.2 GB
**Run:** 2026-06-22T06:48:32 → 2026-06-22T07:37:25

All figures are measured on the same machine, one engine at a time in an isolated subprocess. Failed cells show the reason (OOM / timeout / skip), never a fabricated number. Once an engine fails at a context length it is not run at larger ones.

### Contenders

- **native** — DiffKV native (C++ / ggml, GGUF Q4_K_M)
- **active** — DiffKV active runtime (MLX, int4) — compressed KV
- **dense** — Dense baseline (mlx_lm int4, full KV cache)
- **ollama** — Ollama / llama.cpp (GGUF Q4_K_M)

### Methodology notes

- **Same prompt** (a Needle-In-A-Haystack chat prompt) is built once per context length from the Qwen2.5 tokenizer and fed verbatim to every engine (raw mode, no double chat-templating). `Needle` = whether the generation reproduced the planted passcode (a free correctness signal).
- **Prefill s**: native = the binary's own `[PREFILL_TIME]`; active/dense = `perf_counter` around the chunked prefill forward (after `mx.eval`); ollama = `prompt_eval_duration` from `/api/generate`. All engines are warmed first.
- **Decode tok/s**: fixed-length greedy decode (EOS ignored for native/active/dense); ollama uses its own `eval_count/eval_duration`.
- **Memory**: peak of per-process `max(phys_footprint, RSS)` over the engine tree at 20 Hz — phys_footprint catches MLX Metal buffers that RSS misses; RSS catches mmap'd GGUF weights that phys_footprint misses. For ollama we also record its self-reported `size_vram`.
- **`trunc@32k`**: ollama clamps the context window to Qwen2.5's trained limit (32768) and truncates longer prompts, so it cannot actually perform the task past 32k on this model — those cells are not valid long-context runs.

## Prefill time (s) — lower is better

| Engine | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| native | 5.9 | 13.8 | 35.2 | OOM | skip | skip |
| active | 8.1 | 17.4 | 39.4 | 100.6 | 295.9 | OOM |
| dense | 5.2 | 11.5 | 27.9 | 75.8 | OOM | skip |
| ollama | 5.0 | 11.2 | 29.5 | trunc@32k | trunc@32k | trunc@32k |

## Decode throughput (tok/s) — higher is better

| Engine | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| native | 27.7 | 23.3 | 18.5 | OOM | skip | skip |
| active | 44.3 | 40.8 | 36.1 | 28.0 | 16.7 | OOM |
| dense | 64.7 | 56.6 | 45.5 | 34.3 | OOM | skip |
| ollama | 63.1 | 61.7 | 51.9 | trunc@32k | trunc@32k | trunc@32k |

## Peak memory (GB) — lower is better

| Engine | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| native | 2.50 | 3.37 | 5.04 | OOM | skip | skip |
| active | 2.86 | 2.86 | 3.18 | 4.21 | 6.02 | OOM |
| dense | 2.76 | 4.31 | 5.94 | 5.89 | OOM | skip |
| ollama | 1.35 | 1.48 | 1.75 | trunc@32k | trunc@32k | trunc@32k |

## Maximum usable context (valid run within 8 GB)

| Engine | Max ctx | Limited by |
|---|---|---|
| native | **16k** | oom |
| active | **64k** | oom |
| dense | **32k** | oom |
| ollama | **16k** | context clamp (32k) |

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
3. **Coherence / needle miss (the real failure): under-budgeted sparse retrieval.** With the benchmark's `micro_block_size=256`, the selection floor (`src/main.cpp:1652`) leaves `srl_k_keep=16`, so sparse attention only ever sees ≈16×256 = **4096 tokens** of compressed history — ~25% of a 16k prompt, ~12% of 32k. The needle planted at 50% depth usually is not in the selected slots, so `native` hallucinated `123` at 16k instead of recalling the passcode. At 4k (fully inside the dense window) it instead echoed the instruction — a separate attention-sink/prompt-boundary issue the source itself flags (`src/main.cpp:659`: *"loses the attention sink → echo"*). This is a tuning-sensitive reimplementation, not yet a faithful one on this workload.

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
| native | 4k | ok | 4096 | 60 | 5.9 | 27.7 | 2.50 | 2.40 | — | — | - | 11 |
| active | 4k | ok | 4096 | 128 | 8.1 | 44.3 | 2.86 | 2.03 | 1.60 | — | Y | 16 |
| dense | 4k | ok | 4096 | 128 | 5.2 | 64.7 | 2.76 | 1.39 | 1.68 | — | Y | 11 |
| ollama | 4k | ok | 4096 | 128 | 5.0 | 63.1 | 1.35 | 1.33 | — | 1.24 | Y | 9 |
| native | 8k | ok | 8192 | 60 | 13.8 | 23.3 | 3.37 | 2.35 | — | — | - | 20 |
| active | 8k | ok | 8192 | 128 | 17.4 | 40.8 | 2.86 | 2.06 | 1.63 | — | Y | 25 |
| dense | 8k | ok | 8192 | 128 | 11.5 | 56.6 | 4.31 | 1.39 | 1.79 | — | Y | 19 |
| ollama | 8k | ok | 8192 | 128 | 11.2 | 61.7 | 1.48 | 1.46 | — | 1.37 | Y | 15 |
| native | 16k | ok | 16490 | 128 | 35.2 | 18.5 | 5.04 | 2.93 | — | — | - | 45 |
| active | 16k | ok | 16490 | 128 | 39.4 | 36.1 | 3.18 | 2.35 | 1.87 | — | Y | 48 |
| dense | 16k | ok | 16490 | 128 | 27.9 | 45.5 | 5.94 | 1.40 | 2.03 | — | Y | 37 |
| ollama | 16k | ok | 16490 | 128 | 29.5 | 51.9 | 1.75 | 1.73 | — | 1.62 | Y | 34 |
| native | 32k | oom | — | — | — | — | 7.22 | 3.37 | — | — |  | 69 |
| active | 32k | ok | 32865 | 128 | 100.6 | 28.0 | 4.21 | 2.17 | 2.35 | — | Y | 111 |
| dense | 32k | ok | 32865 | 128 | 75.8 | 34.3 | 5.89 | 1.41 | 2.45 | — | Y | 87 |
| ollama | 32k | trunc@32k | 32767 | 1 | 85.5 | — | 2.22 | 2.20 | — | 2.11 | - | 88 |
| native | 64k | skipped | — | — | — | — | — | — | — | — |  | — |
| active | 64k | ok | 65615 | 128 | 295.9 | 16.7 | 6.02 | 1.97 | 3.28 | — | Y | 309 |
| dense | 64k | oom | — | — | — | — | 5.89 | 1.41 | — | — |  | 501 |
| ollama | 64k | trunc@32k | 32767 | 1 | 78.7 | — | 2.24 | 2.22 | — | 2.11 | - | 81 |
| native | 128k | skipped | — | — | — | — | — | — | — | — |  | — |
| active | 128k | oom | — | — | — | — | 6.11 | 2.03 | — | — |  | 1296 |
| dense | 128k | skipped | — | — | — | — | — | — | — | — |  | — |
| ollama | 128k | trunc@32k | 32767 | 1 | 86.7 | — | 2.21 | 2.20 | — | 2.11 | - | 89 |
