# Native (C++) port plan: pool right-sizing + lego streaming prefill

## STAGE 2 LANDED (2026-07-12) — host mirrors ringed, far blocks pool-sourced

Everything below this section is kept as historical planning context; this section
is the current state.

**What landed** (all inside `DIFFKV_LEGO_PREFILL=1`, still default OFF):
- `k_activations`/`v_activations` are allocated at `lego_buf_rows` (identity zone +
  window ring, same `lego_map_span` geometry as the stage-1 device ring) instead of
  full `[L]`. Chunk captures write through the SAME ring spans the device cache
  uses; the decode-boundary dense-window seeding reads the tail through
  `lego_map_span` (window ring capacity ≥ 2048 always covers the 768-row dense
  window). `k_rotated_activations` was ALREADY skipped under sparse decode
  (allocation gated `!decode_use_sparse`) — plan step 1 was free, verify-only.
- **Far field default = STUDS (`DIFFKV_LEGO_FAR=studs`)** — the MLX studs-only
  port, made possible by forcing UNIFORM residual sets: routed far blocks
  contribute ONLY rows that come out EXACT — the anchor + the block's
  residual-corrected rows (1 + 128 per block; `lego_emit_block_studs`, recon
  computed only AT residual positions so anchor+recon+residual reproduces the raw
  row). Routing selects whole INGEST blocks (stride mbs+1 = 257, 1:1 with pool
  slots, `lego_block_tokens` table); K comes out PRE-ROTATED per POOL_ROT_ABS
  (zero host RoPE on the pool path), V raw. Not-yet-compressed layers fall back
  to exact raw rows (anchor + last-128); "neither" is unreachable and zero-fills
  with a loud `LEGO FATAL`.
- **Why studs, measured (q8 native)** — both extremes fail one gate:
  - `recon` (full-block pool recon): synthesis COLLAPSES 26.7→10.0@8k /
    26.7→3.3@16k (low-rank noise poisons the far field — MLX saw the same);
    margins 11.6/12.5, NIAH 6/6, multi-needle 3/3.
  - `off` (pure omission, no far at all): synthesis holds 26.7/23.3 but margins
    drop 12.6/14.3 → **7.4/7.6** and multi-needle REFUSES (0/3) — prefill hidden
    states lose the far field entirely. (Factual-off confound ruled out: this
    cell also ran factual-off and synthesis matched baseline.)
  - `studs` = exact-only coverage, the middle. Gate numbers in the results block.
- **The uniform-residual unlock**: MLX studs-only looked unportable because
  native's adaptive residual budget (lowrank.cpp OPT-A) is LAYER-dependent
  (per-layer error medians) → per-layer stud counts, which the single shared
  prefill mask/klen cannot express. Resolution: under lego studs the native
  runtime sets `DIFFKV_RESIDUAL_UNIFORM=1`, which skips the adaptive cap so every
  full prefill block carries the full MAX_RESIDUAL=128 set (uniform across layers
  AND blocks). This is MLX parity ("prefill blocks always carry full residual
  sets") and costs no memory — pool residual tensors are pre-allocated at
  MAX_RESIDUAL per slot. It also makes decode materialisation strictly MORE
  exact for easy blocks.
- Consumers gated: `factual_store.build` force-disabled under lego (reads
  ring-evicted mirror positions; it is net-negative and default-off anyway),
  `DIFFKV_DBG_RECON_POS` gated off under lego (`DBG_EXPORT_CHECK` already was),
  the sparse→dense 0-blocks fallback hard-aborts under lego instead of backfilling
  from mirrors that no longer hold the data.

**Gates (2026-07-12, q8_0, ENGAGE=1024)** — the far-mode A/B that picked studs:
- NIAH 6-cell sweep: **6/6 in every mode tried** (recon, off) and 6/6 lego=0
  (refactor no-regression). The 4k cells auto-disable lego (ring wouldn't shrink)
  — they exercise the legacy path. Single-needle recall is decode-time and never
  moved.
- Multi-needle 16k: far=recon **3/3** (reverse listing order), lego=0 3/3 (in
  order), far=off **0/3** (refusal — rejected).
- Margins (8k/16k, depth 0.5): lego=0 12.62/14.32; far=recon 11.60/12.47;
  far=off **7.45/7.62** (rejected).
- Synthesis (native compressed 8k/16k): lego=0 **26.7/26.7**; far=recon
  **10.0/3.3** (rejected); far-omission 26.7/23.3.
- Conformance: PASS (1.19e-07).

**FINAL STUDS-DEFAULT RESULTS (2026-07-12, q8_0, ENGAGE=1024, new binary)**:
- NIAH 6-cell: **6/6**. Multi-needle 16k: **3/3**. Margins 8k/16k:
  **11.73 / 12.38** (baseline 12.62/14.32 — within ~1-2 nats, healthy).
- Synthesis (native compressed 8k/16k): **16.7 / 6.7** vs baseline 26.7/26.7.
  At 8k all 5 facts are retained (only the linkage is lost); at 16k 3 facts drop.
  This is the documented cost of the opt-in flag — the same trade MLX shipped as
  its studs default (MLX: 10.0/6.7 vs its 23.3/3.3 baseline). No far mode
  dominates: off wins synthesis but breaks margins/multi-needle; recon is
  strictly worse than studs on synthesis for the same margins.
- **Memory (the gate stage 1 failed) — the peak MOVES**
  (`benchmarks/native_mem_profile.py`, phase-tagged phys_footprint, MAX_TOKENS=8):
  - 16k: **3.477 GB vs 4.053 GB** lego-off → **−576 MB (−14.2%)**
  - 32k: **4.760 GB vs 5.716 GB** lego-off → **−957 MB (−16.7%)**
  - Prefill-end footprint: 2.50 vs 2.88 GB @16k; 2.80 vs 3.63 GB @32k (the ring).
    The 32k baseline's post-prefill spike (3.63→5.72 during factual build +
    decode setup while the full mirrors were still alive) is gone under lego
    (mirrors ringed + factual forced off). 32k run also ~10s faster end-to-end.
- Operational note: the native binary has a PRE-EXISTING intermittent
  hang-at-exit (full output produced, process never exits; observed on decode
  runs both before and after these changes) — spun off as its own task. Harness
  runs that wait on child exit should use a watchdog until it's fixed.

Status: **STAGE 2 DONE** — remaining follow-ups: CUDA port (below, unchanged),
exit-hang fix (separate task), optional deeper look at recovering the 16k
synthesis facts under studs (e.g. residual selection tuned for linkage tokens).

**Memory decomposition (native_mem_profile.py, the step stage 1 skipped)**:
- Baseline 16k: peak 4.04 GB (lego stage-1) vs 4.06 (off) — flat, and the peak is
  in DECODE SETUP (attend-all materialisation + Metal pipeline compile), AFTER the
  mirrors are freed. Prefill-end footprint ~2.86 GB both ways.
- Baseline 32k: peak 4.92 GB; prefill-end 3.64; the SRL/factual phase (mirrors
  still alive, factual reading them) adds +1.19 GB — THIS is what stage 2 removes.
- Block raw fp32 (`active_k/v`+`svd_k/v`) was ALREADY freed per-chunk during
  prefill (`ingest_prefill` lines ~238-261) — the "engine slot storage" suspect
  from stage 1 was innocent.

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
aliases to MLX, so the torch store is only exercised on CUDA machines). Lego is
NOT implemented on the torch/CUDA path at all (design notes only, C9 in
`CUDA_TRITON_AUDIT.md`) — nothing to default there yet.

## FINAL DEFAULT DECISION (2026-07-12): lego is opt-in on BOTH runtimes

Clean A/B under the CURRENT residual-selection defaults (owner-capture +
coverage-0.25 native / owner-capture native-coverage-0 MLX — see
`RELATIONAL_BINDING_REPORT.md`), which supersedes every earlier lego
measurement (all pre-dated owner-capture and are not a fair comparison):

| | NIAH | multi-needle | binding list-all | margins (8k/16k) | synthesis (8k/16k) | peak mem |
|---|---|---|---|---|---|---|
| native lego=1 vs 0 | 6/6 both | 3/3 both | 5/6 both, identical failure | 11.55/13.13 vs **12.48/14.26** | 16.7/23.3 vs **26.7/26.7** | **-13.9%/-17.5%** (16k/32k) |
| MLX lego=1 vs 0 | 6/6 both | 1/1 both | 6/6 both, identical | — | 0.0/6.7 vs **6.7/6.7** | savings (mechanism unchanged; see notes above) |

Recall/binding are genuinely unaffected by lego either way — the earlier
"synthesis identical" MLX claim was true at the time but does NOT survive
owner-capture (which changes which rows the residual set — and therefore the
mask/attended set under lego — actually contains). The remaining cost lands
entirely on synthesis (real-paper narrative fidelity) and, on native, margins
by ~1 unit. This is a genuine memory-for-fidelity trade, not a strict
improvement in either direction.

**Decision: `DIFFKV_LEGO_PREFILL` defaults to OFF on both runtimes** (native
already was; MLX flipped back — see `mlx_diffkv_wrapper.py` comment at
`self._lego_prefill`). Quality-by-default; opt in explicitly
(`DIFFKV_LEGO_PREFILL=1`) for memory-constrained long-context runs where the
prefill-peak reduction is worth the documented synthesis/margin cost. Don't
re-flip either default without a fresh A/B — residual-selection changes (like
owner-capture) can silently invalidate a prior lego measurement, as happened
here.

**RC8 default (both runtimes) — already resolved, restated for completeness**:
`DIFFKV_RC8_LICENSE` stays OFF on both. Validated end-to-end this session
(commit `fe30621`, AFTER coverage-0.25 became the native default, so the
verdict already reflects current residual-selection defaults — no re-test
needed): RC5/RC8 target comparison-interleave inversions already fixed by
owner-capture/coverage; the one live failure (value→entity REV swaps) is
byte-identical in DENSE (base-model limit, not ours to fix); RC8's factual-store
prerequisite is net-negative; RC8=1 live reproduces its own original disable
reason (drops a locked-out entity's own name). Full writeup:
`RELATIONAL_BINDING_REPORT.md`.
