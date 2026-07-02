# Optimization & Hardening Plan — Differential-KV (for Fable 5)

> **⚠️ 2026-07-02 (Fable 5 execution pass, commit `996ebb5`): read `SESSION_REPORT_FABLE5.md`
> FIRST.** Several of this plan's premises were disproven by measurement:
> - Native was NOT "coherent ≤13k" — NIAH failed at **all** contexts at HEAD. Root causes
>   (all fixed): a dense-window reset that erased generation history every step, a RoPE
>   ingest/decode scheme conflict (double rotation), and a sparse→dense fallback uploading
>   zero K. Pool K is now stored rotated at absolute positions (`POOL_ROT_ABS`, MLX parity);
>   decode does no pool rotation, making project-then-attend exact (and the default).
> - **1.1 (router port) is moot** — routing was never the failure; a residual-key router
>   already exists natively.
> - **3.1's precondition is stale** — the fused kernel already reads per-row U scales. The
>   actual 3.1 blocker now: port `POOL_ROT_ABS` (= delete pool rotation) into the two fused
>   ggml kernels, then selftest → flip default.
> - The remaining accuracy gap (verbatim digit recall from compressed blocks) is **shared
>   with MLX** (A/B on the identical prompt: MLX also fails). Mechanism: the joint K|V SVD
>   is blind to V (|K|≫|V| ⇒ V recon error 24–73%), so exact recall depends entirely on
>   residual capture, whose error-based ranking cannot see recall-critical tokens. Partial
>   fixes landed native-side (V rebalancing `DIFFKV_V_SCALE`, content-aware capture
>   `DIFFKV_RESIDUAL_TOKEN_BOOST`); the capture policy is now the top accuracy work item,
>   above any routing/kernel change. Tier 2 (2.1/2.2/2.4) remains open with design notes in
>   the session report.

**Author:** Opus 4.8 (exploration + planning pass) · **Date:** 2026-07-02
**Scope:** both engines — `ACTIVE_RUNTIME/` (Python, MLX on Mac / PyTorch+Triton on CUDA) and
`diffkv_native/` (C++17 / llama.cpp / ggml, Metal + CPU + CUDA).
**Audience:** Fable 5 — this is a work-plan, not a diff. Every item has files, an approach, and a
verification step so you can execute without re-deriving context.

---

## Status — what's already fixed vs. left for Fable 5 (updated 2026-07-02)

An initial hardening pass (Opus) landed the **safe, self-contained, inspect-verifiable** portability
fixes. Everything algorithmic/perf (needs building + benchmarking) is left for Fable 5.

**✅ DONE in this pass** (all portability/hygiene — no behavior change to the hot paths):
- **0.1** — hardcoded `/Users/...` Metal shader path removed. `diffkv_attention.mm` now resolves via
  `DIFFKV_METAL_DIR` env → executable-relative → CWD-relative → CMake-injected `DIFFKV_METAL_SOURCE_DIR`
  (this build's source tree). CMake wires the define. *(Consolidating onto embedded-metallib = 0.5, still open.)*
- **0.2** — native serving gateway paths now resolve relative to the checkout + honor
  `DIFFKV_BINARY_PATH`/`DIFFKV_MODEL_PATH`.
- **0.3 (partial)** — `.gitignore` now excludes `*.so`/`*.dylib`/`*.o`; the one-time `git rm --cached`
  untracking is documented in `BUILD.md §6` (left for the user to run at a clean checkpoint, so it
  doesn't muddy the current large uncommitted diff).
- **0.6** — `BUILD.md` added: fresh-machine build steps for both engines, no absolute paths.
- **Flag F** — `plot_graphs.py` personal `~/.gemini` dir → `PLOT_OUT_DIR` env / `./plots` default.

**⏭️ LEFT FOR FABLE 5** (why not now: needs a build + benchmark loop, or is algorithmic/high-risk):
0.4 (move `scratch/`), 0.5 (consolidate the two Metal loaders), 0.7 (user checkpoints uncommitted work);
all of **Tier 1** (native router port, NaN-assert mode, factual store), all of **Tier 2/3** (fused
kernels, batched SVD, adaptive/quantized compression, speculative, prefix reuse), **Tier 4**, and flags
G/H/I/J/K. Verification harnesses in §7 must gate each.

---

## 0. How to read this plan

Work the tiers **top-down**. Tier 0 is the "runs on a machine that is not the author's" gate — do it
first or nothing else can be validated elsewhere. Tiers 2–3 are the raw-speed/memory wins (this is
where most of the "push to the limit" asks live). Each item is tagged:

- **Impact** = user-visible payoff · **Effort** = eng time · **Risk** = blast radius / regression
  chance · **Confidence** = how sure I am the lever actually moves the number.

Two facts that reshape several of the suggestions you may have seen (from me and from Gemini):

1. **On macOS the Python "active" runtime is MLX-only.** `DiffKVHFWrapper` is aliased to
   `MLXDiffKVWrapper` on `darwin` ([serving/hf_diffkv_wrapper.py](ACTIVE_RUNTIME/serving/hf_diffkv_wrapper.py)).
   The whole "sophisticated" retrieval layer (FactualExactStore, SRL semantic routing, chunk-graph,
   Eagle, VSL) **only runs in the PyTorch/CUDA `KVRuntimeManager`** — on Mac it is gated OFF and was
   never wired into the hot path. So "use the SRL to drive adaptive compression" (a common suggestion)
   presumes a subsystem that is dormant on the platform you're actually testing on. Reframed in Tier 2.
2. **Decode is dispatch/overhead-bound, not FLOP-bound.** Both engines. Micro-optimising routers and
   knobs yields single-digit % (verified repeatedly). The 2–4× wins require a **fused single-dispatch
   decode kernel** (Metal), not more Python/graph orchestration. This is the through-line of Tier 2/3.

---

## TIER 0 — "Works on any device" blockers (P0, do first)

These are the reasons a fresh clone on another Mac / Linux box / CI runner will **fail or silently
misbehave**. None are algorithmic; all are hygiene/portability.

### 0.1 Hardcoded absolute author path in a *compiled* Metal loader  🔴 ✅ DONE (2026-07-02)
- **File:** [diffkv_native/runtime/diffkv_attention.mm:138](diffkv_native/runtime/diffkv_attention.mm)
- **What:** the Split-K decode Metal path loads `diffkv_decode.metal` by **reading the source file at
  runtime**, trying a few relative paths and then falling back to
  `@"/Users/omchimurkar1/Desktop/Differential-KV/.../diffkv_decode.metal"`. It is compiled into the
  binary (`CMakeLists.txt:65`). On any other machine every relative path misses (they assume a specific
  CWD) and the absolute fallback is nonexistent → the kernel silently fails to build (`g_pipeline`
  stays null) and this attention path is dead.
- **Fix:** stop shipping shader *source* and compiling at runtime. There is already an embedded
  metallib mechanism next door — [metal_runtime.mm](diffkv_native/native_core/diffkv_core/src/metal_runtime.mm)
  loads `diffkv_metallib[]` from [diffkv_metallib.hpp](diffkv_native/native_core/diffkv_core/src/diffkv_metallib.hpp)
  (a `xxd`-style byte array, fully portable). Do the same for the Split-K kernel: precompile
  `diffkv_decode.metal` → `.metallib` at build time (CMake custom command using `xcrun metal`/`metallib`)
  and either embed it as a byte array or resolve it relative to the executable
  (`[[NSBundle mainBundle] pathForResource]` or `dladdr`-derived dir), never a hardcoded home dir.
- **Impact:** high (portability) · **Effort:** low-med · **Risk:** low · **Confidence:** high

### 0.2 Hardcoded absolute paths in the native serving gateway  🔴 ✅ DONE (2026-07-02)
- **File:** [diffkv_native/serving/openai_compatible_api_gateway.py:94-95, 741-747](diffkv_native/serving/openai_compatible_api_gateway.py)
- **What:** `BINARY_PATH_DEFAULT`, `MODEL_PATH_DEFAULT`, and the model-size selection block all hardcode
  `/Users/omchimurkar1/Desktop/Differential-KV/...`.
- **Fix:** resolve relative to the file (`Path(__file__).resolve().parents[...]`) and/or env vars
  (`DIFFKV_BINARY_PATH`, `DIFFKV_MODEL_PATH`) with the repo-relative path as the default. The Active
  Python side already does this cleanly — mirror its convention.
- **Impact:** high · **Effort:** low · **Risk:** low · **Confidence:** high

### 0.3 Committed, architecture-locked build artifacts  🟠 ✅ PARTIAL (2026-07-02: .gitignore done; untrack step documented in BUILD.md §6)
- **What is tracked in git (verified via `git ls-files`):**
  - `ACTIVE_RUNTIME/native_core/diffkv_core/diffkv_core.cpython-314-darwin.so` (arm64 + CPython 3.14 only;
    **also modified on the current branch**),
  - `ACTIVE_RUNTIME/native_core/diffkv_core/build/**` object files (`bindings.o` 8.6M, `decode_attention.o`
    4.6M, `metal_runtime.o` 4.5M),
  - two `diffkv_decode.metallib` binaries.
- **Why it's a bug:** a `.so` compiled for `cpython-314-darwin/arm64` will not import on any other Python
  version, OS, or arch — but its presence masks a missing/failed local build (import "works" with a
  stale binary, then behaves wrong). Object files should never be in VCS.
- **Fix:** `git rm --cached` the `.so`/`.o`/`build/` tree, add to `.gitignore` (a `**/build/` rule
  already exists but only ignores *future* untracked files, not already-tracked ones — the untrack is
  required). Keep source `.metal`; generate `.metallib` at build. Document the build step (see 0.6).
- **Impact:** med (correctness-masking) · **Effort:** low · **Risk:** low · **Confidence:** high

### 0.4 `scratch/` is 181 throwaway scripts, all with hardcoded paths  🟡
- **What:** `scratch/` (61 files) + `ACTIVE_RUNTIME/scratch/` (120 files), most containing
  `sys.path.insert("/Users/omchimurkar1/...")` and hardcoded binary/model paths.
- **Why:** it's fine for a research repo, but it should not be mistaken for shippable code and it bloats
  clones. There is also `plot_graphs.py:6` pointing at a personal `~/.gemini/...` dir.
- **Fix:** move `scratch/` out of the tracked tree (or gitignore it), or clearly quarantine under a
  `research/` dir excluded from any packaging. Not urgent, but do it before anyone treats these as APIs.
- **Impact:** low · **Effort:** low · **Risk:** none · **Confidence:** high

### 0.5 Two divergent Metal-loading mechanisms in native  🟡
- **What:** `metal_runtime.mm` embeds a metallib (portable, used by the default fused-op path);
  `diffkv_attention.mm` compiles source at runtime (non-portable, item 0.1). Redundant and confusing.
- **Fix:** consolidate on the embedded-metallib approach for both. Delete the runtime-source-compile path
  once 0.1 lands. Reduces surface area and removes a class of "works on my machine" bugs.
- **Impact:** med (maintainability) · **Effort:** med · **Risk:** low · **Confidence:** med

### 0.6 No reproducible build / run docs for a clean machine  🟠 ✅ DONE (2026-07-02: BUILD.md added; bootstrap.sh still optional)
- **What:** there is no single "clone → build native → build the CPython ext → run" script that works
  from a fresh checkout with no absolute paths. `CMakeLists.txt` + `setup.py` exist but rely on the
  committed artifacts (0.3) and submodule state.
- **Fix:** add a top-level `BUILD.md` + a `scripts/bootstrap.sh` that: inits the `llama.cpp` submodule,
  builds `diffkv_native`, builds the `diffkv_core` CPython extension in-place, and runs the smoke tests.
  Pin the `llama.cpp` submodule commit (it currently shows as dirty/modified in `git status`).
- **Impact:** high (this IS "works on any device") · **Effort:** med · **Risk:** low · **Confidence:** high

### 0.7 Uncommitted work at risk 🟠 (housekeeping, flag to the user — not for Fable to auto-commit)
- The current branch has substantial uncommitted changes (factual-store MLX port, GPU residual
  correction, native decode edits, the modified `.so`). This is real work that isn't checkpointed.
  **Recommend the user commit/stash before Fable starts** so there's a clean baseline to diff against.
  Do **not** commit on the user's behalf without being asked.

---

## TIER 1 — Correctness gaps (fix before chasing speed on top)

### 1.1 Native (C++) DiffKV-at-scale gibberish above ~13k tokens / ~50 blocks  🔴 (biggest correctness gap)
- **Status:** known, unfixed. Coherent ≈ MLX up to ~13k (after the `ignore_c=false` self-token fix,
  [main.cpp:3590](diffkv_native/src/main.cpp), and the sampler top-k fix). Beyond ~50 blocks the CPU
  custom-op path degrades identically to the fused path — so it is **not** a kernel bug; it is
  routing/capture at scale (the MLX side hit and *solved* the same class of bug with the **residual-key
  router** + `max_residual=64`).
- **Approach:** port MLX's two fixes to native:
  1. **Residual-key top-K router** (`_block_relevance_residual` in
     [mlx_diffkv_wrapper.py:515](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py)) — rank blocks by exact
     `q·k` over each block's anchor + its most-distinctive residual keys, not by an SVD/min-max summary
     (summaries average away the single outlier token you're searching for → drop the needle block at
     scale). This is the single highest-leverage native correctness fix.
  2. **`MAX_RESIDUAL` parity** — native's residual budget and highest-error-first ordering must match
     MLX (`argsort(errors)[-max_res:][::-1]`). See memory `project_native_needle_recall_rootcause`.
- **Impact:** high (unlocks native long-context at all) · **Effort:** high · **Risk:** med · **Confidence:** med-high
- **Verify:** `diffkv_native/tests/test_niah_native.sh` + `make_niah_prompt.py` at 16k/32k; compare
  needle recall against the MLX `benchmarks/niah_recall.py --bench` baseline (MLX is exact 4–32k).

### 1.2 Pervasive `mx.where(isnan, 0.0, ...)` NaN-masking in the MLX decode kernel  🟡
- **File:** [compute_decode_attention_static, mlx_diffkv_wrapper.py:278, 304, 310–322](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py)
- **What:** the kernel scrubs NaNs at 4+ points and clamps `lse` to `-1e9`. Recall is exact today, so
  this is currently harmless, but blanket NaN→0 masks a real numerical fragility (empty-block softmax
  over all `-inf`, the flash LSE merge when one pool is empty). If a future change shifts the regime,
  these guards will hide a correctness bug instead of surfacing it.
- **Approach:** don't rip them out — instead add a `DIFFKV_ASSERT_FINITE=1` debug mode that *raises*
  instead of masking, run the parity + NIAH suite under it, and confirm the guards are dead paths (no
  NaN actually occurs) vs. load-bearing. Document which is which. Cheap insurance.
- **Impact:** med · **Effort:** low · **Risk:** none · **Confidence:** high

### 1.3 Factual store: OFF by default, ~32% decode cost, partial on dense tables  🟡 (deprioritize, but document)
- **State (from HANDOFF_ACTIVE_MLX_2026-07-01):** now correct on realistic multi-entity prompts (4/4)
  via positional query→value linking, but plain sparse is *also* 4/4 there — so the store only earns its
  ~32% cost on facts residuals can't capture (dense tables), where it's 4/5 bare-value, 2/5 exact-key.
- **Recommendation for Fable:** **do not** invest in 5/5-exact on synthetic codes (high behavioral risk,
  low ROI — see the handoff's VSL analysis). If touched at all, the worthwhile move is **query-type-gated
  enable** (turn the store on only for relational/multi-entity queries) so the 32% cost is paid only when
  it can help. Otherwise leave gated OFF.
- **Impact:** low-med · **Effort:** high (if pursuing exact) · **Risk:** high · **Confidence:** high (that it's low ROI)

---

## TIER 2 — Performance: Active runtime (MLX) — the highest-value speed work

### 2.1 Fused single-dispatch Metal decode kernel  🟢 (THE lever, do this)
- **Why:** decode is dispatch-bound. Today `execute_decode_attention` orchestrates ~a dozen small MLX
  ops per layer × 28 layers per token; even after the `@mx.compile` static-shape rewrite
  ([_execute_decode_attention_compiled, mlx_diffkv_wrapper.py:363](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py))
  and the WS1 sync-removal, sparse @4k ≈ 16 tps vs dense @4k ≈ 36 tps. A **custom Metal op** that does
  router-score → top-K gather → SVD reconstruct → anchor+delta+residual+dense attention → flash-LSE
  merge in **one kernel** removes per-op dispatch overhead and the intermediate materializations.
- **Approach:** MLX supports custom Metal kernels (`mx.fast.metal_kernel`). Start from the math in
  `compute_decode_attention_static` (it's already a clean, closed-form reference). Mirror the FlashDecode
  online-softmax structure used natively. Keep the pure-Python path as the correctness oracle behind a
  flag. **Reuse the C++ Metal kernel** in `diffkv_decode.metal` where possible — this is the same math,
  so 2.1 and 3.1 should share a shader.
- **Impact:** very high (the 2–4× that knobs can't give) · **Effort:** high · **Risk:** med
- **Confidence:** high (this is the diagnosed bottleneck) · **Verify:** `tests/test_diffkv_kernel_parity.py`
  must stay green; `benchmarks/niah_recall.py --bench` exact 4–32k; tps via
  `paper/scripts/measure_active.py --single compressed,16384 --gen 40`.

### 2.2 Batched GPU SVD in prefill  🟢 (biggest prefill win)
- **Why:** prefill @16k ≈ 45s, dominated by ~1,736 per-block SVDs (62 blocks × 28 layers). Even though
  `_compress_block` now does reconstruction in MLX ([mlx_diffkv_wrapper.py:1222](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py)),
  the SVD itself (`compress_mlx_block`) is called **per block, serially**.
- **Approach:** batch the randomized-SVD range-finder across *all blocks of a layer* (and ideally across
  layers) into a single batched matmul + batched QR/SVD on GPU. rSVD is seeded (`DIFFKV_SVD_SEED`), so
  determinism is preserved. This is Gemini's "batched SVD" idea and it's sound — the caveat is that MLX's
  `linalg.svd` batching support/perf must be checked; if thin, implement rSVD as batched matmuls + a
  small batched eigendecomp on the `k×k` Gram matrix (rank ≤ 16, so `k×k` is tiny and CPU-eig may even
  win). Expect multi-× prefill speedup.
- **Impact:** high · **Effort:** med-high · **Risk:** med (numerical parity of batched rSVD)
- **Confidence:** med-high · **Verify:** identical compressed blocks vs. serial path (seeded);
  prefill wall-time in `measure_active.py`.

### 2.3 Content-adaptive per-block rank/residual budget  🟢 (the *right* framing of "semantic compression")
- **Why:** today rank (16) and `max_residual` (64) are uniform. The documented failure mode is
  content-dense blocks (a table crammed into one 256-tok block) overflowing the 64-residual budget
  (memory: `max_residual 64→128` took the adversarial case 1/5→4/5) while prose blocks waste budget.
- **Approach — do NOT tie this to SRL** (SRL is off on Mac). Use a **cheap, content-agnostic signal
  already computed in `_compress_block`**: the per-token reconstruction-error spread (`errors`, line 1223)
  and residual-norm distribution. Blocks with a heavy tail (many high-error tokens = numbers/names/code)
  get a larger residual budget and/or +rank; smooth prose blocks get less. This directly attacks the
  only measured accuracy loss and *reduces* average memory. Keep a global cap for the 8 GB ceiling.
- **Impact:** high (accuracy + memory) · **Effort:** med · **Risk:** med (variable-width residuals
  break the `all_blocks_full` fast-path — see the vectorized `mx.take` gather; you'll need a bucketed or
  padded layout) · **Confidence:** med · **Verify:** `benchmarks/relational_ab.py` adversarial +
  `--natural`; memory footprint from `measure_active.py`.

### 2.4 Scale the block pool to the prompt (short-context memory regression)  🟢
- **Why:** `_create_empty_session` pre-allocates the full `max_blocks=256` pool (≈682 MB) even at 4k,
  which is *larger* than a full 4k dense KV. This is the main reason "sparse from start" (now the
  default) regresses short-context memory. Called out as deferred follow-up #2 in the handoff.
- **Approach:** thread a per-session `max_blocks` sized to `ceil(prompt_len/block_size) + generation
  headroom` through `_create_empty_session`, the overflow-shift in `_compress_block`, the `hard_cap`,
  and `DummyMLXPool`. Grow (realloc + copy) only if the session exceeds it.
- **Impact:** med-high (removes the flip's downside) · **Effort:** med · **Risk:** med (many sites to
  thread; snapshot/clone/restore must carry the per-session size) · **Confidence:** high
- **Verify:** peak mem at 4k/8k sparse ≤ dense; no OOM at 64k; existing clone/rollback tests pass.

### 2.5 Speculative decoding (Qwen2.5-0.5B draft → 1.5B target)  🟡 (real 2–3×, but sequence it last)
- **Why:** both GGUF/model sizes are present. Speculative decode is a genuine 2–3× generation lever.
- **Caveat that makes this hard here:** the target must verify K draft tokens **in parallel against the
  compressed KV cache** — i.e. a *batched* (multi-query) version of the sparse decode kernel. That only
  becomes clean **after 2.1** (the fused kernel) exists and can accept `q: [K, H, D]`. Attempting it on
  the current per-token orchestration will be slow and fragile.
- **Approach:** land 2.1 first; extend the fused kernel to a small query batch; add a draft-model loop +
  rejection sampling. On Mac use the MLX 0.5B as draft.
- **Impact:** high · **Effort:** high · **Risk:** high · **Confidence:** med · **Sequence:** after 2.1.

---

## TIER 3 — Performance: Native (C++) engine

### 3.1 Fused FlashDecode Metal kernel with per-row int8 U scale  🟢 (unlocks the fast path)
- **Why:** the fast paths (`build_native_sparse_attn` fused subgraph + the `GGML_OP_DIFFKV_ATTN` metal
  op) are **disabled by default** ([is_native_attn_enabled → false, main.cpp:64](diffkv_native/src/main.cpp))
  because they still assume a *single* block SVD scale, but compression switched to **per-token-row int8
  scales** (`lowrank.cpp`) to fix needle recall. So today native decodes via the slower CPU custom op.
- **Approach:** teach the fused Metal op + `build_native_sparse_attn` to read `get_U_row_scale()`
  (the per-row scales are already stored and already consumed by the CPU op in `decode_attention.cpp`).
  Once the fused path is numerically exact vs the custom op (there's a `DIFFKV_SELFTEST` harness that
  proves byte-parity on the working range), flip `DIFFKV_NATIVE_ATTN` on by default. This is Gemini's
  "FlashDecode" item and it's the correct native speed lever — but the *precondition* is the per-row
  scale port, not the online-softmax structure (which the subgraph already has).
- **Impact:** high (~1.2–1.8× decode measured for the fused path; more with online-softmax tiling)
- **Effort:** high · **Risk:** med-high · **Confidence:** med · **Verify:** `DIFFKV_SELFTEST`
  byte-parity, then `test_niah_native.sh` recall + TPS.
- **Note:** share this shader with 2.1 — same math, one kernel, two callers (MLX custom op + ggml op).

### 3.2 Quantize residuals + anchors to int8 (KV memory for 128k)  🟢
- **Why:** U is already int8; the **residual K/V and anchors are fp16** and dominate the per-block bytes
  (memory: residuals ≈ 71% of a block). At 128k on 8 GB this is the binding constraint.
- **Approach:** per-row int8 (or int4) quant for residual K/V and anchors with a stored scale, reusing
  the same dequant path the U scales already use. SVD factors are normalized → resilient to int8 (this
  is well-supported); residuals are raw KV, so validate int4 carefully (int8 is the safe default). This
  is Gemini's INT8/INT4 idea, scoped to the tensors that are *actually* still fp16.
- **Impact:** high (2× KV memory → longer context on the same box) · **Effort:** med · **Risk:** med
  (recall regression if scales are per-block instead of per-row — do per-row) · **Confidence:** med-high
- **Verify:** recall parity at 16k/32k; peak RSS via `diffkv_native/monitor_memory_native.py`.

### 3.3 Multi-turn prefix reuse (avoid full KV reset per turn)  🟡
- **Why:** documented tech debt — native resets the full KV cache on every conversation turn (no prefix
  reuse), so multi-turn re-prefills the whole history every time.
- **Approach:** retain the compressed block pool + dense window across turns; only prefill the new
  suffix. Needs careful session-state lifecycle (the sampler/position bookkeeping in `main.cpp`).
- **Impact:** med-high for chat workloads · **Effort:** med-high · **Risk:** med · **Confidence:** med

### 3.4 Remove hardcoded domain tokens from SRL scoring  🟡 (correctness/generality)
- **Why:** documented tech debt — physics-domain tokens (`"EP2"`, `"hermitian"`) are baked into
  production SRL scoring, and the stop-word list is duplicated in 3 places with diverging content. This
  makes retrieval quality domain-dependent in a hidden way.
- **Approach:** move these to config/data files; unify the stop-word list into one source of truth.
  Grep both `query_router.py` and `query_router.hpp` (SRL logic is mirrored in both).
- **Impact:** med (generality) · **Effort:** low-med · **Risk:** low · **Confidence:** high

---

## TIER 4 — New capability (only after Tiers 0–3 are healthy)

### 4.1 Pick ONE cross-platform second target and make it first-class
- **Why:** there are currently **three half-live sparse paths**: MLX (Apple-only), the Triton CUDA
  kernel ([native_core/sparse_decode/triton_fused_decode.py](ACTIVE_RUNTIME/native_core/sparse_decode/triton_fused_decode.py),
  PyTorch fallback exists), and the C++ native binary. Maintaining three is why "works on any device" is
  shaky. **Recommendation:** designate the **C++ native** engine as the portable target (it already has
  CPU + Metal + a CUDA `.cu`), get Tiers 0/1/3 solid, and treat MLX as the Apple-optimized fast path and
  Triton as CUDA-optimized. Don't try to keep all three at feature parity.
- **Impact:** high (strategic) · **Effort:** high · **Risk:** low · **Confidence:** high

### 4.2 PagedAttention-style block allocator (native) — research, lowest priority
- **Why:** Gemini's suggestion. Real value **only for concurrent multi-request serving with shared
  prefixes**. The current workload is single-session long-context, where the bounded pool already caps
  fragmentation. Do this only if/when multi-tenant serving becomes a goal.
- **Impact:** low now / high for serving · **Effort:** very high · **Risk:** high · **Confidence:** low
  (that it matters for the current use case) · **Sequence:** defer.

---

## 5. Non-production / "shouldn't be here" flags (consolidated)

| # | Item | Location | Severity | Status |
|---|------|----------|----------|--------|
| A | Hardcoded `/Users/omchimurkar1/...` Metal shader path (compiled in) | `diffkv_native/runtime/diffkv_attention.mm:138` | 🔴 | ✅ fixed |
| B | Hardcoded binary/model paths | `diffkv_native/serving/openai_compatible_api_gateway.py` | 🔴 | ✅ fixed |
| C | Committed arch-locked `.so` + `.o` build artifacts + `build/` tree | `ACTIVE_RUNTIME/native_core/diffkv_core/` | 🟠 | ⚠️ gitignore’d; untrack pending (BUILD.md §6) |
| D | Dozens of `DIFFKV_DBG_*` / experimental env flags interleaved in the hot path | `diffkv_native/src/main.cpp` (6280 lines) | 🟡 | ⏭️ Fable |
| E | 181 hardcoded-path scratch scripts in the tree | `scratch/`, `ACTIVE_RUNTIME/scratch/` | 🟡 | ⏭️ Fable/user |
| F | Personal `~/.gemini/...` output dir | `plot_graphs.py:6` | 🟡 | ✅ fixed |
| G | Hardcoded physics-domain tokens in SRL scoring | `query_router.{py,hpp}` | 🟡 | ⏭️ Fable |
| H | Stop-word list duplicated in 3 places, diverging | (SRL / tokenizer utils) | 🟡 | ⏭️ Fable |
| I | No C++ test suite (only shell scripts) | `diffkv_native/tests/` | 🟠 | ⏭️ Fable (BUILD.md added) |
| J | `__pycache__` committed inside the C++ production tree | `diffkv_native/native_core/` | 🟡 | ⚠️ untrack pending (BUILD.md §6) |
| K | Blanket NaN-masking hides numerical fragility | `mlx_diffkv_wrapper.py` decode kernel | 🟡 | ⏭️ Fable (1.2) |

`main.cpp` at 6,280 lines with ~40 debug env flags is the biggest maintainability risk: it makes the
"which path actually runs by default" question hard to answer (I had to trace it: default = CPU custom
op via `ggml_map_custom3`, fused op ON only when `DIFFKV_NATIVE_ATTN=1`). A pass to (a) split it into
translation units and (b) compile debug scaffolding out behind a single `DIFFKV_DEBUG` macro would pay
for itself before any of Tier 3.

---

## 6. Suggested execution order for Fable 5

1. **Tier 0 entirely** (0.1–0.6) — make it build & run on a clean non-author machine. *Nothing below is
   verifiable elsewhere until this is done.* Ask the user to checkpoint uncommitted work first (0.7).
2. **1.1** (native residual-key router) — the biggest correctness win; also de-risks Tier 3.
3. **2.1** (MLX fused decode kernel) + **3.1** (native fused kernel, per-row scale) — **share one Metal
   shader.** This is the core 2–4× speed program.
4. **2.2** (batched prefill SVD) and **2.4** (prompt-scaled pool) — independent, parallelizable.
5. **2.3 / 3.2** (adaptive + quantized compression) — accuracy + memory.
6. **2.5** (speculative) — only after 2.1.
7. **Tier 4** — strategic consolidation; defer 4.2.

## 7. Guardrails (every change must keep these green)

- `ACTIVE_RUNTIME/tests/test_diffkv_kernel_parity.py` (kernel parity — the ground-truth oracle).
- `benchmarks/niah_recall.py --bench` — exact needle recall 4k/8k/16k/32k (the HARD on-topic prompt;
  easy prompts hid real bugs before).
- `benchmarks/relational_ab.py` (`--natural` realistic 4/4; adversarial as a regression watch).
- Native: `diffkv_native/tests/test_niah_native.sh` + `DIFFKV_SELFTEST` byte-parity before flipping any
  fused-path default.
- Speed/mem: `paper/scripts/measure_active.py` (MLX) and `diffkv_native/monitor_memory_native.py`.

**Golden rule given the diagnosis:** if a change is a router/knob/heuristic tweak, expect single-digit %
— don't over-invest. The real wins are the fused kernel (dispatch), batched SVD (prefill), and
quantization (memory). Measure before and after with the harnesses above; decode is overhead-bound, so
trust wall-clock TPS over FLOP counts.
