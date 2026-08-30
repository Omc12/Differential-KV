# Changelog

All notable changes to Differential-KV are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed

* **Routing K: the divisor is fixed on both runtimes; the CUDA floor stays 16
  and MLX's becomes 2 — deliberately, and measured.** `max(16, 4096 // block_size)`
  lived in three places and bundled two separate issues.

  *The divisor was wrong.* CUDA divided by `block_size + 1` (payload **plus** the
  anchor row), giving `4096//257 = 15` at the 256-token block, and the `max(16, …)`
  existed only to round that back up. It now divides by the token payload, which
  gives 16 at 256 directly, so the floor no longer conceals an off-by-one.

  *The floor value was right on CUDA and wrong on MLX.* K trades synthesis
  quality against the **decode cache it sizes**, so a smaller K only pays where
  that cache is allocated. MLX's is on; CUDA's is gated on
  `DKV_DECODE_CACHE_CUDA`, which defaults to `0` (the `DKV_DECODE_CACHE=1` in
  `serving/decode_config.py` is read only by the MLX wrapper). Measured on CUDA
  (RTX 4070 SUPER, Qwen3.5-2B, `micro_block_size=1024`, `observed_block_span=1024`):

  | metric | K=4 | K=16 | verdict |
  |---|---|---|---|
  | peak VRAM @16k, fresh process per arm | 4213.0 MB | 4213.0 MB | identical |
  | decode ms/tok @32k, paired (A/A calibrated) | 61.06 | 63.10 | CI [−3.4, +9.8] contains 0 |
  | needle recall, 3 contexts × 3 depths | 9/9 | 9/9 | unchanged |
  | linkbench 16k, 24 seeds | 10/24 | 10/24 | = dense (10/24) |
  | synthesis, paired, n=8 | 27.9 | **44.2** | **−16.2, CI [−29.1, −3.4]** |

  So on CUDA a smaller K saves no memory, saves no time, and costs 16 synthesis
  points: **CUDA keeps 16, MLX takes the budget K=4** (its 377.9 → 151.1 MB
  saving is real). `K=8` is the knee (38.8 — resolvably above K=4, not resolvably
  below K=16 at n=8) if the CUDA decode cache is ever enabled.

  Two caveats recorded because they bound what the above proves. At 16k the
  context is 14–15 blocks, **below** K=16, so `nb > k_eff` was false and the
  "K=16" arm is *attend-all*, not "routing 16 blocks". And linkbench has **no
  power** here — dense scores the same as DKV at both 16k and 32k, so it is at
  the model's ceiling; the needle suite is the discriminating retrieval
  instrument, and it is unchanged.

  Also in this change:

  * `dkv_native/serving/batch_engine.cpp` defaulted `micro_block_size` to 256
    while `main.cpp` used 1024, so every **batch-served** native session silently
    ran at the block size linkbench scores 9/24 on against 24/24 = dense at 1024.
  * `hf_dkv_wrapper`'s dead `self.block_size = 256` (assigned, never read) removed
    — it was what made readers conclude CUDA ran 256.
  * `KVRuntimeManager`'s signature default 256 → 1024, and `micro_block_size` is
    now seeded into a **copy** of the caller's config dict so `DKVConfig`'s
    prefill-chunk floor can finally see the real block size (both terms were
    `None` before, so the floor always came from a stale literal).
  * `DKV_TOPK_FRAC` used `int()` on both runtimes (frac 0.3, nb 13 → 3, not 4);
    both now `ceil`, or one env var routes different block counts per runtime.
    Default stays `0.0` (off): measured at 32k, fracs 0.15 and 0.25 moved K from
    4 to 5 and 8 and retrieval did not change.
  * `observed_block_span` measured 1024 at 8k/16k/32k (0 at 2k, where no block is
    written) — **not** the "~32–64 short-context" its own comment claimed, which
    is what had predicted this change would be a CUDA no-op. Comment corrected.
  * `colab/probe_relevance_dist.py`'s CUDA path imported `HFDKVWrapper`, which
    does not exist, so it had never run. Fixed to `PyTorchDKVHFWrapper` and
    executed: at 32k with 27 blocks the top block holds a median 0.708 of
    softmaxed relevance, and a 0.90/0.95 mass threshold needs p90 K = 7/8 —
    landing on the same ~8 as the synthesis knee, from an independent direction.
  * Pinned by `ACTIVE_RUNTIME/tests/test_routing_k_budget_parity.py` (11 tests,
    source-level so they need no GPU), which asserts the shared 4096 budget, the
    per-runtime floors **and why they differ**, the payload divisor, `ceil`
    parity, and that `main.cpp`/`batch_engine.cpp` agree. Suite: 307 passed,
    18 skipped.

* **`DKV_RESIDUAL_RUN_RESERVE=query` (strict) now honours its own contract when
  no query is pinned** — the one known defect shipped with v1.2.0. In
  `_select_residual_rows`
  ([`lowrank.py`](ACTIVE_RUNTIME/native_core/compression/lowrank.py)) the
  `elif not _has_priority: pass` branch was tested *before*
  `elif _mode == "query": return [], set()`, and swallowed every unpinned
  strict-mode call — so strict `query` fell through to reserve-by-error and
  silently behaved as `all`, the exact opposite of its documented "reserve
  nothing". The two branches are swapped; strict mode is the only way into that
  chain without priority information, which is why the shadowed branch was both
  dead for the default and wrong for the only mode reaching it.

  **No shipped behaviour changes.** Verified by differential test against the
  v1.2.0 implementation over 4,000 randomised inputs per mode:

  | mode | result |
  |---|---|
  | `query_first` (shipped default) | identical, 4000/4000 |
  | unset (→ `query_first`) | identical, 4000/4000 |
  | `all` | identical, 4000/4000 |
  | `query`, query pinned | identical, 1963/1963 |
  | `query`, no query pinned | **changed — this is the fix** (1916/2037) |

  Unpinned strict `query` now hands the whole budget to the per-token error
  ranking instead of reserving every run. Any A/B previously run with
  `DKV_RESIDUAL_RUN_RESERVE=query` against unprioritised runs was comparing
  `all` with `all` and should be re-measured.
  `test_residual_capture.py::TestQueryScopedReservation::test_unasked_runs_do_not_claim_slots`
  passes; suite is 406 passed, 2 skipped, 0 failed.

---

## [1.2.0] — 2026-08-30

**349 commits · 140 files changed · +45,750 / −2,869 lines since v1.1.0.**

Test suite at this tag: **405 passed, 2 skipped, 1 failed** (`test_unasked_runs_do_not_claim_slots` — see Known limitations).

The correctness release. v1.0.0 and v1.1.0 shipped a runtime whose CUDA path was
substantially inert: the production Triton decode kernel never launched, the
residual router raised on every partial-rotary model, the content-aware boost had
never fired, and thirteen per-token gates were dead on every hybrid model. Those
paths now run, and the two engines agree.

Headline: **natural-text needle recall goes from 2/12 to 12/12 at 8k and 3/12 to
12/12 at 32k — equal to the dense control at both context lengths** — with no
budget raised.

### Fixed — code that never ran

* **Triton production decode kernel had never launched.** An autotune
  configuration broke its launch and the `try/except` fell back to PyTorch
  silently, so every "fused decode" measurement was really the unfused path.
* **The residual router raised on every partial-rotary model** — it had never run
  on Qwen3.5 or any other partial-RoPE architecture.
* **`DKV_COMPRESSED_DECODE` / `DKV_COMPRESSED_DECODE_MIN_CTX` had no CUDA reader
  at all**, a 32× divergence between the documented and actual behaviour.
* **The content-aware residual boost had never fired** — an off-by-one in its
  gate.
* **Thirteen once-per-token gates were dead on every hybrid model.**
* **Edge propagation never ran in the production router.**
* **`DKV_REMAT_CACHE` was unreachable** on the path production actually runs, and
  the re-materialisation cache had a 0% hit rate because routing was not frozen
  on the interval.
* **Routing ran only on layer 0** — the layer-0 branch shadowed every other
  layer; and a 40-block gate with no MLX counterpart kept the router off entirely
  at 2k and 8k.
* **`DKV_SPARSE_BIAS='auto'` was never adaptive on CUDA** — a constant `+2.0`.

### Fixed — reconstruction and RoPE frames

* **Compressed keys were never rotated.** Prefill stored pre-RoPE keys while the
  decoder was told the pool held post-RoPE ones. Four further prefill capture
  sites stored `K` in the wrong frame.
* **Both CUDA read-time residual rotations used the wrong position.**
* **RoPE tables were reshaped by `head_dim` instead of `rotary_dim`**, and fp16
  RoPE tables read as float32 caused nondeterministic decode. Partial RoPE on the
  CUDA dense-window path is fixed — Qwen3.5-2B could not decode at all before it.
* **All three residual readers now SUBSTITUTE rather than ADD**, matching MLX.
  Adding residuals on top of the lossy twin double-counted them.
* **Re-materialised blocks were missing their anchor row** — one real token
  dropped per block — and the remat cache reconstructed blocks without their
  residuals entirely.
* **Residuals were computed against the fp16 reconstruction while decode reads
  the int8 one.**
* **A skip block was only exact for its first `max_residual` tokens.**
* **The dense-window workspace was sized with the wrong block size**, trimming
  live context.

### Fixed — pool, paging and session state

* **A freed pool slot stayed "occupied" forever** and kept its tiering state, so
  eviction corrupted live blocks.
* **Eviction could zero a block the same decode step was about to read.**
* **The batched pool writer did not clear a recycled slot's stratified group** (a
  race on Qwen2.5-1.5B).
* **`clear_session` never cleared the decode block cache** — decode read the
  previous generation's pool slots.
* **`_MUTATION_OUT_ACTIVE` leaked across sessions** as a module global.
* **Block metadata was allocated as an inference tensor**, unwritable after
  prefill.
* `seq_lens` now describes what was **stored**, not what was offered.
* The block pool is sized by **attended** layers, not total layers.

### Fixed — serving and sampling

* **The batch engine owns its KV cache** — word salad becomes real text.
* **CUDA stream races in both the batch-engine decode and prefill streams** — the
  decode result was read before it finished.
* **Greedy sampling had no NaN guard**, and the engine's logits do contain NaN.
* **The dense-only decode path returned EMPTY attention on a zero-block step.**
* **The bypass KV cache is reset when a new sequence starts.**
* `DKV_DETERMINISTIC` now defaults ON — greedy decoding at long context was not
  reproducible without it, and the cause is the decode attention's reduction, not
  compression.

### Fixed — recall on natural text

The tiled-haystack NIAH suites in this repo inflate recall: the filler is one
sentence repeated, so a random code is a colossal outlier that the residual
budget is all but guaranteed to keep. Refilled from real papers, recall was
**2/12 at 8k**. The needle was found and *corrupted*, not missed —
`Falcon-9427-618`**`5`** for `...618`**`3`** — because Qwen splits that code into
eleven tokens and residual selection ranked them one at a time, losing the tail.
Three changes fix it, none of which raises a budget:

* **Residual selection takes a token RUN whole or not at all.**
* **The router scores the exact key**, instead of summing two different
  rotational frames.
* **Whole runs are ordered by whether the QUERY asks for them** before by how
  badly they reconstruct, and the reservation is scoped to those runs — falling
  back to SnapKV's observation window (the last 64 prompt tokens) when no
  question span can be pinned.

| filler | ctx | dense | DKV before | DKV now |
|---|---|---|---|---|
| natural text | 8k | 12/12 | 2/12 | **12/12** |
| natural text | 32k | 12/12 | 3/12 | **12/12** |

`multifact_eval_cuda` passes **9/9 for the first time** (relational 4/4,
multi-needle 3/3), and synthesis moves **13.3 → 30.0**, then **→ 46.7** with
rarity-aware capture. Cost: ~4% of prefill; decode and memory unchanged.

### Added

* **Physical 4-bit / 8-bit group-quantized residual buffers** (`group_size=64`),
  packed into `uint32`/`int32` on both MLX and CUDA/Triton. `DKV_RESIDUAL_QUANT`
  defaults to `int4` (**3.56× compression**). In INT4 mode the persistent fp16
  residual buffers are `None`, and only the routed top-K blocks are dequantized
  into transient scratchpads. **R=256 INT4 residuals (112.9 MB at 16k) cost 44%
  less than the old R=128 FP16 residuals (200.7 MB)** while keeping twice as many
  exact tokens.
* **Content-Aware / Rarity-Aware residual selection** on CUDA, MLX and the C++
  core (`DKV_RARITY_CAPTURE=1`). IDF-weighted rarity boosting with per-class
  multipliers (digits 20.0×, entity names 14.6×, rare terms 7.3×) and a
  punctuation-exclusion guard, so delimiters stop starving keywords of residual
  slots on JSON, code and logs.
* **Unrotated key storage on `mid` / `high` / `ultra`**, which is what buys dense
  parity. At 32k on Qwen3.5-2B: digit-table **24/24 vs dense 24/24** (rotated
  14/24), linkbench direct **47/48 vs 47/48** (rotated 40/48). Storing keys
  rotated costs **42% of exact digit recall**. It is not free — read-time
  rotation costs ~11% of decode on a hybrid model and ~27% on a dense-attention
  one; `low` and `DKV_ROTATED_POOL=1` take the speed instead.
* **`ultra` preset**, matching the dense control on synthesis.
* **Routed CUDA-graph decode** (`DKV_FAST_DECODE` / `--fastdc`, default ON),
  gated per session to whether a graph will actually be captured. Measured
  **1.41–1.48×** on replay where routing is non-selective; 25% at 4k and 9% at
  8k end-to-end. Not bit-identical to eager (one ULP at step 1, greedy argmax
  flips ~step 32); accuracy unaffected (digit-table 24/24, linkbench 23/48, equal
  to eager and to dense).
* **Re-materialisation cache ported to CUDA** from MLX and enabled by default.
* **Shared low-rank bases** (`DKV_SHARED_BASIS=1`, opt-in). Blocks whose delta
  subspaces agree read one basis row instead of each storing its own `V`, which
  is 39% of a pool slot. Pool **91.4 → 69.8 MB (−23.6%)** with 2.58× sharing and
  zero forced joins. **This is a capacity gain, not a memory saving** — peak moves
  only 0.6–1.8%, because weights dominate. Requires an unrotated pool and fp16
  KV; the pool now **refuses** both bad combinations at construction rather than
  warning, since each fails invisibly with pool MB unchanged.
* **Gemma 4 E2B hybrid-architecture support** via selective global-only DKV
  patching.
* **Port to the `transformers` 5.14.1 `AttentionInterface` registry.**
* **`DKV_POOL_ATTENDED_ONLY=1`** — sizing the pool by attended layers saves
  406.9 MB of a 542.5 MB pool at 11.4k on Qwen3.5 (6 of 24 layers attended).
* ~30 new test modules, including shared-basis (torch + MLX), pool recycling and
  aliasing, residual quantization, rarity capture, Metal decode-attention
  isolation, unrotated pool, sparse-prefill routing, and Gemma-4 wrapper tests.
* Extensive diagnostic and benchmark tooling under `colab/` — MLX↔CUDA parity,
  logit fidelity, needle depth sweeps, linkbench, tablebench, synthesis power,
  and batch-engine corruption reproducers.

### Changed

* **Default MLX block size 256 → 1024.** Measured on Qwen3.5-2B-4bit: linkbench
  **9/24 → 24/24 (= dense)**, needles 6/6 either way, session pool
  **135.6 → 60.0 MB**, and the pool against the dense KV it replaces
  **0.95× → 0.28× (3.61× smaller)**. At 256 the fixed 128-token residual budget
  stored half of every block verbatim, which is why the old default barely
  compressed. Use `512` for synthesis-shaped work.
* **Preset ladder standardized on INT4 residuals** with capacities 64 / 128 / 256
  for `low` / `mid` / `high`.
* **`DKV_DECODE_CACHE_INTERVAL` 16 → 4** (MLX). The routed block set is frozen
  for the interval, so a needle whose block is routed late stays invisible that
  long. The old default's claimed 15% speed justification does not reproduce —
  paired and A/A-calibrated, 16 vs 4 reads +0.095 ms/token, 95% CI
  [−0.486, +0.677].
* Fixed block size by default (MLX parity); adaptive sizing moved behind a flag.
* Model weights stream to the GPU instead of loading on the CPU first.
* Compiled kernels fall back to eager when the backend fails at call time,
  and CUDA-graph capture failures are surfaced rather than swallowed.

### Retracted

Several previously reported results did not survive re-measurement and are
withdrawn rather than quietly dropped:

* **The rank sweep** — it was randomised-SVD noise; `svd_energy` is the dial, and
  which of `rank` and `svd_energy` binds depends on the input, not the config.
* **"Algorithmic, not dispatch"** — MLX's re-materialisation cache had simply
  never been ported.
* **32k decode results** — that path is nondeterministic without
  `DKV_DETERMINISTIC`, so the numbers were not comparable.
* **The `ultra` unpatch control** was invalid; the component it claimed to
  establish is not established.
* **Two speed flags that shipped without a speed measurement**, and a decode
  regression that turned out to be a delta between two unmeasurable numbers.

### Known limitations

* `dkv_native/` (the standalone C++ engine) remains **experimental /
  work-in-progress**.
* `DKV_SHARED_BASIS` is opt-in and, on the models measured, has **no operating
  point where it currently pays** — on MLX decode is 0.5–0.65× in paired rounds.
* `DKV_ROTATED_POOL=0` is measured **inert on MLX** (same score and the same
  predicted answer on all 24 linkbench seeds) while costing ~39% of decode. It
  ships off by default there and is not part of MLX's `ultra`.
* NIAH numbers on a tiled haystack are inflated; use
  `colab/needle_depth_sweep.py --filler natural` to judge recall.
* **`DKV_RESIDUAL_RUN_RESERVE=query` (strict) does not honour its own contract
  when no query is pinned.** In `_select_residual_rows`
  ([`lowrank.py:405`](ACTIVE_RUNTIME/native_core/compression/lowrank.py)) the
  `elif not _has_priority: pass` branch shadows the `elif _mode == "query":
  return [], set()` branch below it, so with no priority information strict
  `query` mode falls through and reserves **every** run — i.e. it silently
  behaves as `all`, the opposite of the documented "reserve nothing". The
  shipped default `query_first` is **not** affected (it cannot reach that
  branch), so no default behaviour is wrong; but any A/B run with
  `DKV_RESIDUAL_RUN_RESERVE=query` against unprioritised runs was comparing
  `all` against `all`. Covered by the (currently failing)
  `test_residual_capture.py::TestQueryScopedReservation::test_unasked_runs_do_not_claim_slots`,
  left failing deliberately rather than skipped.
  **Fixed after release — see [Unreleased].**

---

## [1.1.0] — 2026-07-25

* **Multi-turn session cache isolation** — `clear_session()` now purges
  `self.sessions[session_id]` along with the residual gather caches (`_res_cache`,
  `_cache_kv`), preventing stale-cache contamination across turns. Covered by
  `ACTIVE_RUNTIME/tests/test_multi_turn_session.py`.
* **Prose entity recall suite** — `benchmarks/prose_fact_recall.py` evaluates
  unassisted proper-noun retrieval (10 non-digit facts) from 8k to 64k.
* **`DKV_DISABLE_REGEX_HEURISTICS=1`** to evaluate pure low-rank SVD compression
  independently of pattern-matching rules.
* **Benchmark reconciliation** — README Table 1 realigned with committed logs;
  PyTorch dense baseline (OOM ≥16k) explicitly separated from the MLX dense
  baseline (64k at 821 s prefill); 64k prefill crossover reported as **1.72×**
  (477 s DKV vs 821 s MLX dense); pattern-protected NIAH recall (100%) and
  unassisted prose entity recall (~60%) reported side by side.

## [1.0.0] — 2026-07-23

* First official release of the unified Differential KV-cache engine.
* Apple Silicon backend with Metal shader compilation and MLX runtime
  integration; Linux CUDA backend with C++/CUDA and PyTorch Triton kernels.
* Hardware auto-detection via `import dkv; dkv.info()`.
* Unified `pip install` support across environments.

[Unreleased]: https://github.com/Omc12/Differential-KV/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/Omc12/Differential-KV/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Omc12/Differential-KV/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Omc12/Differential-KV/releases/tag/v1.0.0
