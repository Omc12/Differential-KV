# Native (C++) port plan: pool right-sizing + lego streaming prefill

## STAGE 1 LANDED (2026-07-12) — correctness ✓, measured RAM win ✗ (see findings)

`DIFFKV_LEGO_PREFILL=1` (native, default OFF) implements the ring:
identity zone `[0, base_end=max(sp_min, sink+window))` + modular window zone
(`wnd_cap = window + 2·chunk`) for the persistent device prefill cache; routed
far blocks are gathered from the raw host mirrors (k_activations rotated
on-the-fly + v_activations — EXACT raw rows) into per-layer FAR tensors and
concatenated ahead of the range views. Gated on sparse decode + cached_len==0;
auto-disables when the ring wouldn't shrink anything (short prompts).

**Validated**: needle 3/3 (4k/8k/16k, 128-token decode, DIFFKV_ENGAGE_THRESHOLD=4096),
prefill time within noise of baseline, engages as designed (32k: 6144+2048
device rows vs 32865 — 81% device-cache reduction), zero stderr errors.

**FINDING — the device ring does NOT move the process peak**: prefill-only
phys_footprint A/B at 16k (4.085 vs 4.053 GB) and 32k (5.908 vs 5.916 GB) is
flat despite the cache being ~750 MB smaller at 32k. The native peak is set by
something else — candidates: full-length host mirrors (k_activations +
v_activations, ~940 MB @32k), engine slot storage, ggml scheduler/galloc
reserves, or Metal page-touch accounting. Decomposing this needs an
allocation-profiling session (phase-tagged footprint sampling) BEFORE building
stage 2 — do not assume mirror-ringing alone will surface the win either.

## STAGE 2 (start here — 2026-07-12): ring the HOST mirrors, not just the device cache

**Why stage 1's device ring measured nothing, and why stage 2 should**: stage 1
ring-sized `persistent_k_cache`/`persistent_v_cache` — but those are GPU/Metal
buffers (`ggml_backend_alloc_ctx_tensors` on `backend`), and Metal-buffer bytes
are apparently NOT counted by `proc_pid_rusage`'s `phys_footprint` the way MLX's
unified-memory buffers are (confirmed: 81% device-cache cut at 32k, zero
footprint change). The THREE full-length `std::vector<ggml_fp16_t>` host
mirrors — `k_activations`, `k_rotated_activations`, `v_activations`
(main.cpp ~2576-2583, `L * F_test` elements each, `n_layers` of them) — are
plain host RAM and DO count. At 16k/1.5B that's ~3 × 16490 × F_test × 2 bytes ×
28 layers ≈ 940 MB TOTAL not yet touched by lego. This is the actual target.

**Scope**: ring `k_activations` and `v_activations` (drop `k_rotated_activations`
entirely if possible — see below) the same way stage 1 ringed the device cache:
identity zone + modular window, trimmed in `micro_block_size` chunks so routed
blocks stay whole. The far-block gather in `_lego_capture_stream`'s C++
equivalent already reads these mirrors before uploading to the FAR device
tensors (stage 1) — so ringing them just means: (a) write incoming chunks at
the RING-MAPPED offset instead of `pos_start`, (b) read the far gather from the
ring using the SAME block-to-offset mapping, (c) trim + track `ring_start` on
the host side exactly like stage 1's `lego_map_span`/`lego_buf_off` (that logic
is reusable almost verbatim — it was written buffer-agnostic on purpose).

**`k_rotated_activations` note**: this one is ONLY used to seed
`persistent_k_cache` on a `cached_len > 0` continuation (main.cpp ~2868) and
during the per-chunk RoPE step (~3132, but ONLY `if (!decode_use_sparse)` —
i.e. it's already skipped entirely when sparse decode is active, which is
lego's precondition). So under lego it should be possible to **not allocate it
at all** rather than ring it — check the `if (!decode_use_sparse)` guard at
allocation (~2578) covers this; if so that's ~940MB/3 removed for free before
any ring logic.

**Consumers to re-audit before trimming `k_activations`/`v_activations`** (both
are read by more than the graph):
- decode-boundary dense-window seeding (reads the tail — should already be
  ring-compatible since the ring always holds the tail)
- SRL/factual-store build (`update_descriptors`, `finalize_srl_index` equivalent)
  — check if these read arbitrary absolute positions or only recently-ingested
  ones; if arbitrary, they need the ENGINE (compressed pool) as the source for
  far positions instead, matching what stage 1 did for the graph's far tensors
- `DIFFKV_DBG_EXPORT_CHECK` / `RECON_POS` debug paths — these already sample
  `k_activations` at arbitrary absolute positions for verification; either gate
  them off under lego (simplest) or read through the engine like the graph does

**Validation**: repeat exactly the stage-1 protocol (needle 3/3, margins vs
baseline, synthesis identical) PLUS `benchmarks/native_mem_profile.py`
(phase-tagged `phys_footprint` sampling; `PROF_CTX=16384 python
benchmarks/native_mem_profile.py` with/without `DIFFKV_LEGO_PREFILL=1`) to
confirm the peak actually drops this time before calling it done. Do not skip
the profiler step — that is exactly the check stage 1 skipped and how the
"no measured win" surprise happened.

Status: PLAN (2026-07-11), STAGE 2 SCOPED (2026-07-12). The MLX reference implementation landed in
`ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` (`DIFFKV_LEGO_PREFILL`); this doc
maps it onto `diffkv_native/src/main.cpp`. Read the MLX flag comments first —
all fidelity findings (studs-only default, knife-edge margins, synthesis
sensitivity) were measured there and apply here.

## Why (measured on MLX, same structure here)

Native prefill holds, per layer, BOTH a device raw cache and a host mirror for
the FULL prompt, plus the compressed pool:

- `persistent_k_cache[l]` / `persistent_v_cache[l]` — F16 device tensors `[head_dim, kv_heads, L]` (main.cpp ~2860).
- `k_rotated_activations[l]` / `v_activations[l]` — F16 host vectors `[L * F]` (main.cpp ~2577-2583).

At 16k on Qwen2.5-1.5B that is ~940 MB of raw KV duplication on unified memory.
The MLX lego mode cut peak process memory 4.38→3.29 GB @16k by bounding raw KV
to O(sinks + ring + chunk) and attending the far field via compressed pieces.

## Mapping

| MLX piece | Native equivalent | Status |
|---|---|---|
| per-chunk flush (`compress_deferred_prefill_blocks`) | streaming ingest (`IngestManager`, prefill ingest thread) | EXISTS |
| block materialisation (decode route math) | `materialize_routed_kv()` (main.cpp ~5032) | EXISTS — needs a studs-only variant (anchor + residual rows, skip low-rank recon) |
| residual router | anchor+residual scoring in the decode router; prefill has host-side lexical routing (`sp_block_tokens`) | EXISTS (lexical is fine for prefill — no readback needed) |
| sparse ranges in the prefill graph | `build_prefill_ctx_graph(..., sp_ranges)` | EXISTS — needs a second source (materialised far rows) besides the persistent cache |
| raw recency ring | replace full-[L] `persistent_*_cache` with ring-sized tensors + rolling offset | NEW |
| sinks raw copy | first `sp_sink_blocks*micro_block_size` rows pinned in the ring buffers | NEW (trivial) |
| engage gate | `sp_min` + "compressed far blocks exist" + total-prompt-will-decode-sparse | NEW — NOTE: below the decode memory-budget gate (~64k) slots stay dense-resident, so lego-prefill requires EAGER compression + raw-drop for far blocks, i.e. it overrides the budget gate for prefill state. This is the main design decision. |

## Steps done in stage 1 (device side — DO NOT redo)

Studs-only far-block materialisation, the ring-offset math (`lego_map_span`,
identity+modular-window), and the graph plumbing (chunk write spans, FAR
tensor concat ahead of `sp_ranges`, mask ordering) are ALL landed and
buffer-agnostic — stage 2 reuses this logic for the host mirrors, it does not
reimplement it. See `_lego_prefill_attend`-equivalent additions in
`build_prefill_ctx_graph` / the chunk loop (commit `5090e26`).

## Steps for stage 2 (host side — this is the open work)

1. Check whether `k_rotated_activations` can be skipped entirely under lego
   (see the note above) — do this first, it's the cheapest possible win.
2. Ring `k_activations` / `v_activations` using the same identity+modular-window
   scheme, reusing `lego_map_span`/`lego_buf_off` verbatim (parameterize them
   over which buffer they're mapping, if not already generic).
3. Re-audit the consumers listed above; redirect any that need arbitrary
   absolute-position reads on far positions to the engine (compressed pool)
   instead of the raw mirror, matching what the graph's far-tensor gather
   already does.
4. **Gates (MANDATORY per repo protocol)**: native NIAH 6-cell sweep + margins
   (`benchmarks/native_margin_probe.sh`), multi-needle, synthesis
   (`benchmarks/synthesis_eval.py --engine native`), conformance vs MLX, AND
   the phase-tagged footprint profiler (see Validation above) — confirm the
   peak actually moves this time. Expect the 16k/0.1-style knife-edge on MLX
   to have a native analogue: judge on margins + content, not a single
   exact-match cell.

## CUDA

The torch/CUDA path (`hf_diffkv_wrapper` + Triton kernels) cannot be tested on
this Mac. Port ONLY behind default-OFF env gates and add cert items to
`CUDA_TRITON_AUDIT.md`'s GPU checklist. The MLX session-pool growth
(`_ensure_block_capacity`) has a direct torch analogue if the HF wrapper's
session store pre-allocates (check `DiffKVHFWrapper` before porting — on Mac it
aliases to MLX, so the torch store is only exercised on CUDA machines).
