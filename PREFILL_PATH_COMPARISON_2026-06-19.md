# diffkv_native vs ACTIVE_RUNTIME — Prefill-Path Comparison (long-context prefill is slow)

**Date:** 2026-06-19
**Author:** read-only audit (no code changed in either tree)
**Scope:** the systems that run **prefill** (initial prompt ingestion before the first generated
token), traced end-to-end on both sides, focused on **long context**. Findings only — nothing
fixed. Non-prefill issues I noticed while reading are in §6.

> ## ✅ FIX STATUS (updated 2026-06-19, after applying changes)
>
> | Finding | Status |
> |---|---|
> | **§3.2** micro_block_size 16→256 | **FIXED** — `cli.py:880` + `main.cpp:1305` = 256; compression volume now ~matches MLX. |
> | **§3.1** compress-during-prefill | **FIXED (MLX-like)** — `immediate_prefill=false` (`streaming_sparse_ingest.cpp:468`) → blocks compress only as they fall out of the 512-token recency window, like MLX. |
> | **§3.4** lag from BLAS / 1 worker | **FIXED** — single-thread BLAS **restored** (`cli.py:658-661`; multi-thread BLAS was escaping background QoS and saturating all cores = the Mac lag) + compressor workers raised to 4 (`DIFFKV_COMPRESSOR_THREADS=4`, background-QoS so lag-safe). Queue overflow already gone via §3.2. |
> | **§3.3** O(L²) per-chunk prior-K/V re-upload + per-chunk readback/mask | **OPEN** — needs a persistent incremental KV buffer (like MLX `make_prompt_cache`). Correctness-sensitive ggml rewrite; not done yet (can't runtime-validate a 24k-prompt here). Tracked for an opt-in flag. |
>
> Build verified green after changes. The **lag** (the user's reported symptom) is addressed by
> the §3.4 BLAS fix; §3.3 remains as a pure prefill-*speed* item. Original findings below.
>
> ---
>
> **Reference = MLX.** Established in `OUTPUT_PATH_COMPARISON_2026-06-19.md`: on this Mac
> `DiffKVHFWrapper` is a conditional alias for **`MLXDiffKVWrapper`** (`hf_diffkv_wrapper.py:1212`),
> and MLX is installed in `diffkv_venv`. So the comparison below is C++ vs **`mlx_diffkv_wrapper.py`**,
> not the PyTorch fallback. This is the companion to the decode/output report; same method, fresh
> file as requested.

---

## 0. TL;DR — why C++ prefill is slow in long context

Three independent, compounding divergences, all of which scale badly with prompt length:

1. **C++ compresses the whole prompt during prefill; MLX does not (§3.1).**
   MLX's `compress_prefill_kv()` is a literal **no-op** (`mlx_diffkv_wrapper.py:635-636`). MLX keeps
   the prompt **dense** in a persistent prompt cache during prefill and only compresses the *tail*
   later (deferred at prefill-end + as the window slides during decode, `_compress_eligible_blocks`
   / `_flush_oldest_block`). C++ instead SVD-compresses **every block during prefill** via
   `ingest_prefill` (`main.cpp:2275`). So C++ does a large amount of SVD work in the prefill window
   that MLX simply skips.

2. **C++ block size is 16× smaller than MLX → 16× more SVDs and pool slots (§3.2).**
   Default `micro_block_size`: **C++ = 16** (`diffkv_native/serving/cli.py:860`), **MLX = 256**
   (`ACTIVE_RUNTIME/serving/cli.py:795`). For a 24k-token prompt that is ~1500 blocks/layer in C++
   vs ~94 in MLX. Combined with #1, C++ runs on the order of **L/16 × n_layers** SVD compressions
   during prefill (~36,000 at 24k × 24 layers) while MLX runs ~**L/256 × n_layers** *deferred*.

3. **C++ prefill is stateless per chunk → O(L²) re-upload + mask rebuild; MLX is incremental (§3.3).**
   C++ rebuilds the graph, reconstructs the full `chunk_len × intra_ctx_len` causal mask, and
   **re-uploads all prior-chunk K/V to the GPU every chunk** (`main.cpp:2130, 2196-2221`). Summed
   over chunks that is O(L²) host→device traffic and O(L²) mask construction. MLX passes a
   **persistent `prefill_cache`** (`make_prompt_cache`, `mlx_diffkv_wrapper.py:979-980`) so each
   chunk only feeds new tokens and the KV cache accumulates on-device — no re-upload.

Amplifiers (§3.4): the C++ SVD compressor runs on **one** worker thread (`async_compressor.cpp:25`)
with **forced single-thread BLAS/LAPACK** (`VECLIB/OMP/MKL/OPENBLAS_NUM_THREADS=1`, set only in
`diffkv_native/serving/cli.py:644-647`, *not* in ACTIVE_RUNTIME). At 24k tokens the compress queue
(`MAX_QUEUE_SIZE = 32768`) **overflows**, dropping SVD jobs (`async_compressor.cpp:56-59`) — a
quality bug on top of the slowness.

> Note on your recent commit (`492e215`): the §3.1 adaptive-k routing change and the
> `dense_start_positions` fix are **decode-side** and do **not** touch prefill. So this slowness is
> pre-existing — you're likely just exercising longer prompts now that decode is usable. See §6.1
> for one decode-side caveat in that commit.

---

## 1. The two prefill paths

| Stage | diffkv_native (C++) | ACTIVE_RUNTIME (MLX) |
|---|---|---|
| Driver | `src/main.cpp` chunked prefill loop `while (pos_start < L)` (`:2126`) | `serving/batch_engine.py::_step` prefill branch → `MLXQwenModel.__call__` (`mlx_diffkv_wrapper.py:966`) |
| Forward attn | per-chunk fresh ggml graph, dense flash-attn over prior+current (`build_prefill_ctx_graph`) | `self.mlx_model(inputs, cache=prefill_cache)` over a persistent prompt cache |
| KV between chunks | **re-uploaded from host every chunk** (`main.cpp:2211-2221`) | **kept on-device** in `make_prompt_cache` (`mlx:979`) |
| Compression timing | **during** prefill, streamed per chunk (`ingest_prefill` → `streaming_sparse_ingest.cpp` → `AsyncCompressor`) | **deferred** — `compress_prefill_kv` is `pass` (`mlx:635`); tail compressed at prefill end / during decode |
| Block size | `micro_block_size = 16` (`cli.py:860`, `main.cpp:1305`) | `micro_block_size = 256` (`cli.py:795`) |
| SVD | LAPACK `sgesdd`/`sgesvd` + Jacobi fallback (`lowrank.cpp:148-210`), 1 worker, single-thread BLAS | numpy/MLX SVD, deferred, far fewer calls |

---

## 2. How I traced it

- C++ prefill loop read in full (`main.cpp:2126-2282`): per-chunk graph build, mask build, prior
  K/V upload, forward, K/V capture, RoPE, `ingest_prefill`.
- MLX prefill read: `MLXQwenModel.__call__` (`mlx:966-1009`), `_get_or_create_prefill_cache`,
  `compress_prefill_kv` (`:635`), `_compress_eligible_blocks` (`:731`), `_flush_oldest_block`.
- Compressor: `async_compressor.cpp` (worker count, queue, overflow) + `lowrank.cpp` (SVD method).
- Block-size defaults: both `cli.py` argparse blocks. BLAS-thread env: both `cli.py`.

---

## 3. Prefill divergences (ranked by long-context impact)

### 3.1 [PRIMARY] MLX defers all prefill compression; C++ compresses during prefill
- **MLX:** `compress_prefill_kv(self, session_id): pass` (`mlx_diffkv_wrapper.py:635-636`). During
  prefill, K/V accumulate in the dense prompt cache (`capture_prefill_kv` writes token-by-token,
  `:605-633`). Compression only happens when the dense window overflows
  (`recency_window + block_size`, `:734`) — i.e. at prefill end (`compress_deferred_prefill_blocks`
  → `_compress_eligible_blocks`) and incrementally during decode. So **prefill itself does ~0 SVD**.
- **C++:** every chunk calls `runtime_manager.ingest_prefill(...)` (`main.cpp:2275`), which slices
  the chunk into `micro_block_size` blocks and submits each to the SVD compressor immediately. So
  the prompt is fully compressed *inside* the prefill window.
- **Effect:** C++ pays the entire compression bill during prefill; MLX pays a small fraction of it,
  later. The longer the prompt, the bigger this asymmetry.

### 3.2 [PRIMARY] `micro_block_size` 16 (C++) vs 256 (MLX) → 16× more SVDs/blocks
- Defaults differ by 16× (`cli.py:860` vs `cli.py:795`); C++ reads `DIFFKV_MICRO_BLOCK_SIZE`
  (`main.cpp:1307`), MLX takes `args.micro_block_size` into `block_size`/`micro_block_size`
  (`cli.py:480-481`).
- 24k-token prompt: C++ ≈ 1500 blocks/layer × 24 layers ≈ **36,000 SVDs**; MLX ≈ 94 blocks/layer
  (deferred). Even though each C++ SVD is on a smaller matrix (16×F vs 256×F), the **per-call
  overhead, queue churn, pool-slot bookkeeping, and routing-candidate counts all scale with block
  COUNT**, so 16× more blocks is 16× more of that fixed overhead — during prefill.
- **Reconstruction note:** the memory's "item J" lowered C++ `micro_block_size` 64→16 to "match
  Python (16)". That matched the **streaming_sparse_ingest / HF** value, not the **MLX** runtime
  value (256). Against the actual reference, C++ is 16× too granular. (Same wrong-reference pattern
  the output report flagged.)

### 3.3 [PRIMARY] Stateless per-chunk prefill: O(L²) re-upload + mask rebuild + graph rebuild
- C++ resets `prefill_ctx` (`main.cpp:2130`) and builds a **new graph each chunk**
  (`build_prefill_ctx_graph`, `:2170`) with `ggml_backend_sched_reset` + `alloc_graph` (`:2183-2184`).
- The full causal mask `chunk_len × intra_ctx_len` is reconstructed on CPU and uploaded every chunk
  (`:2196-2207`). Σ over chunks ≈ O(L²) CPU + upload.
- The prior-chunk K/V (`prior_k_tensors/prior_v_tensors`, size `pos_start × F`) is **re-uploaded
  every chunk** (`:2211-2221`). Σ ≈ O(L²) host→device bytes. For 24k tokens × 24 layers this is on
  the order of tens of GB of redundant transfer across the prefill.
- **MLX:** one persistent `prefill_cache` per session (`mlx:979-980`); each chunk feeds only its new
  tokens and attends over the accumulated on-device cache — no per-chunk re-upload, no per-chunk
  full-mask rebuild. (MLX even flags it must `mx.clear_cache()` the *peak* prefill activation at the
  decode transition, `:987-990` — i.e. it deliberately keeps KV resident through prefill.)
- This is the divergence most likely to dominate wall-clock prefill at very long context, because it
  is genuinely **super-linear** (O(L²)) and entirely C++-side.

### 3.4 [AMPLIFIER] One compressor worker + forced single-thread BLAS + queue overflow
- `AsyncCompressor::start()` spawns **1 worker** by default (`async_compressor.cpp:25`, "1 worker
  thread for preset low"); override via `DIFFKV_COMPRESSOR_THREADS`.
- `diffkv_native/serving/cli.py:644-647` forces `VECLIB/OMP/MKL/OPENBLAS_NUM_THREADS=1` (to avoid
  audio-driver preemption). ACTIVE_RUNTIME's `cli.py` does **not** set these. So C++ SVD
  (LAPACK `sgesdd`, `lowrank.cpp:192`) is single-threaded *and* runs on a single worker — i.e. the
  36,000 SVDs (§3.2) are effectively serialized.
- `MAX_QUEUE_SIZE = 32768` (`async_compressor.hpp:86`). A 24k-token prompt (~36k jobs) **exceeds**
  it → `"Job queue overflow! SVD job dropped"` (`async_compressor.cpp:56-59`) + `force_invalidate`
  (`:107`). So beyond the slowness, some blocks never get compressed (silent quality loss in long
  context). This compounds with the §3.1/§3.2 volume — the smaller the block size, the sooner the
  queue overflows.

### 3.5 [SECONDARY] Chunk size
- C++ prefill `chunk_size`: low=512 / mid=512 / high=2048 (`main.cpp:1899-1910`, env
  `DIFFKV_PREFILL_CHUNK_SIZE`). MLX uses `cfg.prefill_chunk_size` (preset-driven). Smaller chunks =
  more graph rebuilds (§3.3) and more `ingest_prefill` calls. Not the dominant term, but it
  multiplies the §3.3 per-chunk fixed cost.

### 3.6 [SECONDARY] Per-chunk RoPE recompute — already mitigated
- The committed code pre-rotates prior K once into `k_rotated_activations` and rotates only the new
  chunk (`main.cpp:2244-2272`). This is fine / not a regression. Listed for completeness.

---

## 4. What is NOT the prefill cause (so it isn't re-chased)

- The forward **attention compute** is O(L²) on *both* sides — inherent to causal prefill; not a
  divergence (MLX uses `mx.fast.scaled_dot_product_attention`, C++ `ggml_flash_attn_ext`). The
  divergence is the *re-upload/mask/graph* overhead around it (§3.3), not the matmul itself.
- `rank = 16` matches on both (`main.cpp:1481` ↔ MLX `:1029`).
- The recent decode-side commit (`492e215`) does not touch the prefill loop (verified via `git show`).

---

## 5. Quick way to confirm the ranking (no code change)

- Set `DIFFKV_MICRO_BLOCK_SIZE=256` and re-run the long prompt: if prefill time drops sharply, §3.2
  (+ its effect on §3.1/§3.4) is confirmed as a major term. (This also makes the block count match
  MLX.)
- Set `DIFFKV_COMPRESSOR_THREADS=4` (or more): isolates how much of the time is serialized SVD
  (§3.4) vs the forward/re-upload (§3.3).
- Time a prompt at 2k / 4k / 8k / 16k and check whether prefill grows **linearly** or
  **quadratically**. Super-linear growth points at §3.3 (re-upload/mask); roughly-linear-but-steep
  points at §3.1/§3.2/§3.4 (compression volume).
- (All read-only/env-only; nothing rebuilt.)

---

## 6. Other issues noticed in files I read (not prefill-related)

1. **Your commit `492e215` moved one decode constant away from the real reference.** It changed the
   factual neighbor threshold `0.35 → 0.45` "per HF ref" (`main.cpp:4003`) and relabeled several
   `mlx_diffkv_wrapper.py` comments to `hf_diffkv_wrapper.py`. But **MLX is the reference** and MLX
   uses **0.35 / 0.50** (`mlx_diffkv_wrapper.py:889/895`). So the original `0.35` was correct; the
   new `0.45` matches the *non-running* HF fallback. Low impact (factual-gated, multi-turn), but it
   drifted from the right reference. (This is fallout from my first-draft report, which I corrected
   afterward — flagging so it can be reverted if desired.)

2. **§3.1 adaptive-k vs MLX coverage.** The committed §3.1 routing fix scales attended blocks to
   `max(20, 0.15·N_total)` up to 200 — but that is the **HF/PyTorch `adaptive_k`** shape. The MLX
   reference attends **ALL** compressed blocks at decode (no routing cap). With `micro_block_size=16`
   producing ~1500 blocks, attending 200 is ~13% of context; MLX (mbs=256, ~94 blocks) attends 100%.
   So the fix helps but does not reach MLX-equivalent coverage. The `micro_block_size` mismatch
   (§3.2) is upstream of *both* the prefill cost and the decode-coverage gap — worth being aware
   that these two reports point at the same root knob.

3. **Compressor queue overflow is silent-ish** (one stderr line per drop, `async_compressor.cpp:58`)
   and invalidates the block (`:107`). On long prompts this means dropped/incomplete compression
   with no surfaced error to the user — both a quality and a debuggability issue.

4. **Stale/wrong comment after the edit:** `main.cpp:2851` now reads "mirrors
   `hf_diffkv_wrapper.py:1204-1238`" for n-gram detection (relabeled in `492e215`), but the line
   numbers were the MLX ones; the HF file's loop-detect lives elsewhere. Cosmetic, but the
   citations are now doubly wrong (wrong file *and* the numbers don't line up).

5. **Hardcoded paths fixed** — `492e215` replaced the absolute `/Users/omchimurkar1/...` model and
   binary paths in `diffkv_native/serving/cli.py` with `os.path`-relative ones. Good; noting since
   it was flagged previously.

---

*End of report. No source files in either tree were modified. Companion to
`OUTPUT_PATH_COMPARISON_2026-06-19.md`; the shared root knob is `micro_block_size` (§3.2 / §6.2).*
