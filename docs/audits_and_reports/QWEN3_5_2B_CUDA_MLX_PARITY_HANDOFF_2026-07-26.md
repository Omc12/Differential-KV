# Qwen3.5-2B port + CUDA/MLX parity — session handoff (2026-07-26)

**Status: IN PROGRESS, session interrupted mid-work.** This doc exists so a follow-up
session can resume without re-deriving everything. Read this fully before touching
`dkv_attention.py`, `dkv_backend.py`, or `mlx_dkv_wrapper.py` again.

## 0. What was asked

1. Run the full test suite that was previously run on Qwen2.5 against a newer Qwen
   model, in a separate results collection.
2. Audit DKV for hardcoded Qwen-specific assumptions (it must work for any model,
   any family) — fix what's found along the way, don't do a big upfront audit first.
3. (Mid-session) Model identity resolved via `AskUserQuestion`: neither "Qwen 3.6 2B"
   nor "Qwen 3.5 2B" existed as first guessed; confirmed real model is
   **`Qwen/Qwen3.5-2B`** (a hybrid linear-attention/full-attention VLM-family text
   model, NOT a plain dense transformer — see §1).
3. (Later) User explicitly said: fix CUDA to match MLX wherever they diverge; if CUDA
   does something *better*, flag it, don't just copy MLX blindly; keep both
   implementations in sync going forward — when a bug is found in one, port the fix
   to the other.
4. (Later) User asked for a full "what else differs between MLX and CUDA" audit —
   this was delegated to 3 parallel Explore agents; **all 3 completed in full**, see
   §5.
5. Native (`dkv_native` C++/GGML) runtime was explicitly **dropped from scope** —
   user said "we don't work on native, we are focusing on active for both cuda and
   mlx." Do not resume native work unless re-asked.
6. Session was interrupted (env reset) before final synthesis/commit. User's
   instruction at interruption: capture everything, resume next session from here.

## 1. The model: `Qwen/Qwen3.5-2B` is architecturally unusual

This is **not** a drop-in replacement for Qwen2.5 in engineering terms, even though
it's "the next Qwen." Confirmed via direct config/source inspection:

- **Hybrid layers**: `config.text_config.layer_types` is a 24-entry list of
  `"linear_attention"` (20 layers, a GatedDeltaNet/SSM-style recurrent mechanism, NO
  KV cache in the traditional sense) and `"full_attention"` (4 layers, every 4th,
  standard multi-head attention). DKV's whole point is compressing a KV cache — it
  can only ever act on the 4 full-attention layers per model instance (24 total for
  the 2B here → wait, confirmed empirically: **24 layers total, 6 are full_attention**
  — recheck exact ratio if precision matters, but the key fact — most layers are NOT
  standard attention — is solid and confirmed live).
- **QK-norm**: `Qwen3_5Attention`/`Qwen3NextAttention` apply RMSNorm to Q and K
  (per-head, on `head_dim`) before RoPE. Not present in Qwen2.5/Llama/Mistral.
- **Gated attention**: `q_proj` outputs `2 * num_heads * head_dim` (packs
  `[query | gate]`), split via `torch.chunk`/`mx.split`, and
  `o_proj(attn_output * sigmoid(gate))` — an extra multiplicative gate on the
  attention output that doesn't exist in Qwen2.5-style attention.
- **Partial rotary + mRoPE**: `rope_parameters.partial_rotary_factor=0.25` (only 25%
  of head_dim gets rotated) and `mrope_section`/`mrope_interleaved` (multi-axis RoPE
  inherited from the VLM lineage). Confirmed the *effective* rope_theta lives at
  `text_config.rope_parameters.rope_theta` (nested), not a flat `config.rope_theta`.
- **It's technically a VLM text backbone**: `architectures: Qwen3_5ForConditionalGeneration`
  (has a vision tower, image/video token ids). `AutoModelForCausalLM.from_pretrained`
  resolves to the separate `Qwen3_5ForCausalLM` class instead (confirmed: this one
  keeps `model.model.layers` flat, NOT nested under `.language_model.model.layers`
  like the VLM variant) — but `mlx_lm`'s own `qwen3_5.py` only implements the
  VLM-style `Model` class (`self.language_model = TextModel(...)`, `.layers` exposed
  via a `@property`, not `model.model.layers` directly). **This is why the
  `model.layers` vs `model.model.layers` accessor bug hit MLX specifically** — HF's
  loader picks the "nice" class, mlx_lm's loader only has the VLM-shaped one.

None of the above is Qwen-specific in the sense of "only Qwen has this" — hybrid
attention (Jamba, Zamba, Bamba, Qwen3-Next), QK-norm (all of Qwen3, Gemma2/3, OLMo2),
and nested rope-config conventions are all spreading across the field. Fixing these
properly is fixing DKV for *any* model with these traits, not a Qwen special-case.

**Local artifacts**: MLX 4-bit quant at `models/Qwen3.5-2B-4bit` (converted locally
via `mlx_lm.convert`, works). Full bf16 HF weights cached normally (`~/.cache/huggingface`,
used by the CUDA/MPS path via plain `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B")`).
A GGUF was also downloaded (`models/Qwen3.5-2B-Q4_K_M.gguf`, bartowski quant) for the
native path — **irrelevant now that native is out of scope**, can be deleted if disk
space matters.

## 2. IMPORTANT DISCOVERY: two parallel CUDA/PyTorch integration paths exist

This was **not initially known** and caused real confusion mid-session — read this
section before assuming anything about "the CUDA path."

There are two completely different mechanisms for how `PyTorchDKVHFWrapper` (in
`ACTIVE_RUNTIME/serving/hf_dkv_wrapper.py`) makes DKV intercept attention on the
CUDA/PyTorch/MPS backend:

### Path A (legacy): monkey-patching `layer.self_attn.forward`
- Code: `ACTIVE_RUNTIME/runtime/dkv_attention.py`, function `apply_dkv_attention_patch`
  (~2700 lines, defines a giant per-layer closure `dkv_forward` that fully
  reimplements Q/K/V projection, norm, RoPE, gating, attention, from scratch).
- Called from `hf_dkv_wrapper.py` unconditionally today (see below — both paths'
  setup code run; only one actually intercepts anything at runtime depending on the
  loaded model's `attn_implementation`).
- **This is the path THIS session did most of its CUDA work on** (see §3.3) — before
  discovering Path B existed.

### Path B (new, from an EARLIER separate session, found uncommitted in the working
tree partway through this session): `AttentionInterface` registry
- Code: `ACTIVE_RUNTIME/runtime/dkv_backend.py` (new file, ~500 lines). Registers a
  named backend with transformers 5.x's `AttentionInterface.register("dkv", ...)`
  system instead of monkey-patching. HF's own model code (already correct for
  QK-norm/gating/partial-rotary/mRoPE, since it's unmodified) does projection, norm,
  RoPE, and calls the registered function with **already-processed** Q/K/V — DKV's
  function only has to (a) recover unrotated K via a mathematically-exact
  inverse-RoPE step (`k = k_rot*cos - rotate_half(k_rot)*sin`, valid because RoPE is
  an orthogonal rotation), then (b) do its normal sparse/compressed decode or prefill
  routing, then (c) return `(attn_output, None)` — `o_proj` and the gate-multiply (if
  any) are applied by HF's own code afterward, not by DKV.
- Full design rationale + a 3-way benchmark (legacy monkeypatch vs new interface vs a
  "just store rotated keys, skip the inverse-RoPE" variant) is written up in
  **`things_after_paper.md` at the repo root** (gitignored — read it directly, don't
  rely on git to surface it). That doc's benchmark was run only against
  `Qwen/Qwen2.5-0.5B-Instruct` at 6000 ctx, **never against Qwen3.5-2B**.
- **Gating**: `hf_dkv_wrapper.py`'s actual runtime check is
  `if os.environ.get("DKV_USE_ATTENTION_INTERFACE", "1") == "1"` — **defaults ON**.
  (Note: `dkv_backend.py`'s own internal module-level constant reads the same env var
  with a **different default, `"0"`** — `_USE_ATTN_INTERFACE = os.environ.get(...,
  "0")`. This constant doesn't actually gate anything at runtime as far as this
  session traced — `hf_dkv_wrapper.py`'s check is what decides whether
  `attn_implementation="dkv"` gets passed to `from_pretrained` — but the mismatched
  defaults are exactly the kind of thing that causes silent confusion later. Worth
  cleaning up: pick one default, one source of truth.)
- **Consequence**: this session's earlier "CUDA generation test produces output
  identical to MLX" success (the `<think>\nThinking Process:...` match) was run
  WITHOUT setting `DKV_USE_ATTENTION_INTERFACE=0`, meaning **it almost certainly
  exercised Path B, not the Path A fixes this session spent the most effort on**.
  This was not understood at the time it was reported to the user as "the CUDA fix
  works" — that claim needs re-verification against the correct path (in progress,
  see §3.3/§6).

**Both paths' relevant setup/gating code call `apply_dkv_attention_patch` AND
register the `dkv` backend** unconditionally in current `hf_dkv_wrapper.py` — verify
exactly how they interact (does registering the AttentionInterface backend make the
monkeypatch a no-op because the class's `.forward` is never invoked once
`attn_implementation="dkv"` swaps in HF's own generic attention call path? Almost
certainly yes, since `attn_implementation="dkv"` changes which internal function HF's
`Qwen3_5Attention.forward` delegates to — but this was not 100%-confirmed by reading
code, only inferred. Confirm explicitly before assuming.)

## 3. Fixes made this session (all UNCOMMITTED changes are now safely committed —
see §7 — but re-verify after a session gap, environment resets have burned this
session once already)

### 3.1 MLX (`ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py`) — FULLY VERIFIED, working

All of these were found by actually trying to load/run Qwen3.5-2B through
`MLXDKVWrapper` and fixing whatever broke, iterating empirically (NOT by static
audit first — that approach was explicitly rejected by the user mid-session as
over-investigation).

1. **`_patch_attention_layers` crashed** on `model.model.layers[0].self_attn` — layer
   0 is a linear-attention layer with no `self_attn` at all. Fixed: only patch/touch
   layers that `hasattr(layer, "self_attn")`; raise a clear error only if literally
   none exist.
2. **`model.model.layers` doesn't exist** for this model class (mlx_lm's `Model` for
   qwen3_5 nests as `self.language_model = TextModel(...)`, only exposing `.layers`
   via a property). Fixed 3 call sites to use `model.layers` (the property mlx_lm
   itself provides for exactly this purpose) instead of `model.model.layers`.
3. **New helper `_resolve_attn_dims(attn)`**: resolves `(n_heads, n_kv_heads,
   head_dim)` across naming conventions (`n_heads` vs `num_attention_heads` vs
   `num_heads`) AND across gated-attention's doubled `q_proj` width (prefers reading
   `attn.head_dim` directly when present, since dividing q_proj's output width by
   n_heads gives *2x* head_dim for gated variants). Used at both the
   `MLXKVBlockManager` construction site and inside `_patch_attention_layers` (which
   now stamps `attn.n_heads/n_kv_heads/head_dim` onto every patched module).
4. **QK-norm support** added to `attention_forward`: applies `self.q_norm`/`k_norm`
   (if present) to queries/keys, in `(B,L,H,D)` layout, before transpose — matches
   HF's own `Qwen3_5Attention.forward` ordering exactly (verified by reading HF
   source, not guessed).
5. **Gated-attention support** added: detects `q_proj` output width == `2 *
   n_heads * head_dim`, splits into `[query | gate]`, applies `sigmoid(gate)` to the
   attention output at both `o_proj` call sites, before `self.o_proj(...)`.
6. **THE big bug** (this is the one that actually mattered — the others just
   prevented crashes; this one was silently producing garbage): at the
   prefill→decode boundary, when compressed decode engages, MLX's code used to do
   `self._prefill_caches.pop(cache_key, None)` — dropping the **entire** native
   KV-cache list as a memory optimization (correct for models where every layer is
   attention). For a hybrid model, that list also holds the linear-attention layers'
   `ArraysCache` recurrent state — dropping it reset those 18/24 layers to a blank
   state on **every single decode token**, since a fresh `MLXQwenModel.__call__`
   happens per generated token and each one independently got `cache=None` for those
   layers. Fixed: only null out the list entries belonging to `self_attn` (DKV-managed)
   layers; leave non-attention layers' cache entries untouched so their state
   persists correctly across decode steps. For non-hybrid models (every layer has
   `self_attn`), behavior is unchanged (falls back to the original full-pop).
   **Verified**: compressed-mode output went from repeating "of of of of..." garbage
   to byte-identical match with dense-mode and with plain `mlx_lm` output, at both
   short context (~20 tok) and long context (~2800 tok, real block compression
   engaged, correct needle retrieval of a random `ZEBRA-4471-QUARTZ`-style code
   confirmed).

### 3.2 CUDA — `native_core/kv_runtime_manager.py` — real hardcode removed

`_compress_block_sync`'s rope_theta resolution had:
```python
rope_theta = 10000.0
if hasattr(self, "model") and self.model is not None:
    rope_theta = getattr(self.model.config, "rope_theta", 10000.0)
if "qwen" in str(getattr(self, "model_id", "")).lower():
    rope_theta = 1000000.0
```
The literal `"qwen" in model_id` branch unconditionally overrides whatever the
model's own config said, for **any** Qwen-family model — including Qwen3.5, whose
real rope_theta (10,000,000, nested under `text_config.rope_parameters.rope_theta`)
is 10x different from the hardcoded 1,000,000. Confirmed `self.model`/`self.model_id`
DO get set on this manager instance (via `hf_dkv_wrapper.py:
self.manager.model_id = self.model_id; self.manager.model = self.model`, right after
construction) — so this branch is live code, not dead. Fixed: removed the qwen
string-check entirely; made the generic path also check nested
`rope_parameters`/`rope_scaling`/`text_config.rope_parameters` dicts, not just a flat
`config.rope_theta` attribute. **Not empirically re-tested after the fix** (no CUDA
GPU available; this is Path-A-adjacent code, may not even be reached if Path B is
what's actually active for a given model — see §2).

### 3.3 CUDA — `runtime/dkv_attention.py` (Path A / legacy monkeypatch) — fixed but
possibly not the path that's actually running

1. Config reading (`num_heads`, `num_key_value_heads`, `hidden_size`, `head_dim`) at
   the top of `apply_dkv_attention_patch` made nested-`text_config`-aware (same
   pattern as the rope_theta fix).
2. The version-guard's `model.model.layers[0].self_attn` probe (used to introspect
   the attention class's forward signature) fixed to find the first layer that
   actually has `self_attn`, same hybrid-architecture reasoning as MLX's fix.
3. The main patching loop (`for i, layer in enumerate(model.model.layers): ...
   layer.self_attn...`) fixed to `continue` past layers without `self_attn`.
4. **The big one**: the version guard used to unconditionally `raise RuntimeError`
   telling the user to pin `transformers==4.46.3` whenever it detected the
   post-4.48/5.x attention calling convention (`position_embeddings` positional,
   `past_key_values` renamed from `past_key_value`, `use_cache`/`output_attentions`/
   `cache_position` no longer explicit named params). Confirmed live via a real crash
   on this Mac (proxied through MPS, which runs the identical code path CUDA would):
   `RuntimeError: ... attention forward signature ('self','hidden_states',
   'position_embeddings','attention_mask','past_key_values','kwargs') lost the 4.x
   args...`. Rewrote to **detect and support both conventions in one `dkv_forward`**,
   since (confirmed by reading `Qwen3_5DecoderLayer.forward`'s actual call site)
   *every* argument is passed by keyword in both conventions — never positionally —
   so one signature can declare both `past_key_value=None` and `past_key_values=None`
   and normalize them at the top; `use_cache` is inferred from cache-object presence
   when not passed explicitly; the 3 return statements now conditionally return a
   2-tuple (new convention, confirmed via `hidden_states, _ = self.self_attn(...)`
   at the real HF call site) vs a 3-tuple (old convention) based on a
   `_new_cache_convention` flag computed once at patch time.
   - **CONFIRMED VERIFIED, isolated from Path B**: re-ran with
     `DKV_USE_ATTENTION_INTERFACE=0` explicitly forcing Path A. Output:
     `'...assistant\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Task: Say hello.\n    *   Constraint: Exactly '`
     — identical (up to the truncation point at `max_new_tokens=30`) to plain
     `mlx_lm`, MLX DKV (both modes), and CUDA Path B. This is the definitive result:
     **Path A's calling-convention rewrite works correctly on Qwen3.5-2B**, not just
     "ran without crashing" — confirmed by direct text comparison against known-good
     references, in isolation from Path B. Not yet tested at long context (real
     compression engaging) for Path A specifically — only Path A's *load and basic
     decode* is confirmed; MLX's long-context/real-compression verification has not
     been repeated for either CUDA path.

### 3.4 CUDA — `serving/hf_dkv_wrapper.py` — real, independently-confirmed bug fixed

The MPS model-loading branch used `device_map="mps"` (Accelerate's dispatch
mechanism). For Qwen3.5-2B specifically this hung for 4+ minutes at ~391% CPU with
RSS stuck near-zero (never progressing) — confirmed via direct process inspection,
not assumption. A plain `AutoModelForCausalLM.from_pretrained(...).to("mps")` (no
`device_map`) loaded the same model in ~7-15s. This mirrors reasoning the file's own
CUDA (non-MPS) branch already documents for an analogous case
(`device_map=` triggering `caching_allocator_warmup`, ~2x memory during load). Fixed
by adding an `.to(device)`-based branch for the unquantized-MPS case, matching the
existing unquantized-CUDA branch's pattern, falling back to the original
`device_map="mps"` branch only when quantization is requested (BitsAndBytes needs
device_map). **This fix is orthogonal to the Path A/B question — it applies
regardless of which attention backend loads.**

### 3.5 Benchmark scripts (not core DKV, but were producing invalid/crashing results)

- `benchmarks/bench_worker.py`: `run_dense` (the "Standard PyTorch dense" baseline
  used in `run_bench.py`'s A1 NIAH sweep) had `model_id =
  "Qwen/Qwen2.5-1.5B-Instruct"` hardcoded, **completely ignoring** whatever model was
  actually under test — meaning any active-vs-dense comparison for a different model
  would have silently compared apples (new model) to oranges (Qwen2.5) with no error.
  Added a dedicated `--pytorch-dense-model-id` flag (separate from `--dense-model-id`,
  which names the MLX-quantized path used by `run_active` — the two engines need
  different-format model identifiers, so they can't share one flag), threaded through
  `run_bench.py` → subprocess args → `bench_worker.py`'s own argparse.
- `bench_worker.py`: both `run_active` and `run_dense` had hardcoded memory-footprint
  formulas baking in Qwen2.5-1.5B's exact layer count (28) and head_dim (128) as
  literals. Made generic: reads actual `num_hidden_layers`/`head_dim`/`num_key_value_heads`
  from the loaded model's config (nested-`text_config`-aware, and for hybrid
  architectures, `run_dense`'s dense-KV-footprint estimate now only counts
  `full_attention` layers via `config.layer_types`, since linear-attention layers
  don't grow an O(T) KV cache at all; `run_active`'s estimate now uses the actual
  count of `self_attn`-bearing layers DKV can compress, not the total layer count).
- `benchmarks/run_ppl_mlx.py`: `compute_ppl_for_mode` concatenated logits for the
  **entire** context (`mx.concatenate(logits_list)` → shape `[N, vocab_size]`) before
  calling `cross_entropy` once. For Qwen3.5-2B's much larger vocab (~248k vs Qwen2.5's
  ~152k) at just 4000 tokens, this blew Metal's 4GB single-allocation limit
  (`RuntimeError: [metal::malloc] Attempting to allocate 7945246720 bytes...`).
  Fixed to compute a token-weighted running loss sum per-chunk (bounded to
  `O(chunk_len * vocab_size)`, `CH=512`) instead of materializing the whole sequence's
  logits at once. **Verified fixed** — retest showed a sane (if suspiciously-low,
  because the eval corpus is highly repetitive filler text — pre-existing test-design
  quirk, not something this fix should touch) PPL instead of crashing.

## 4. Verification status summary

| Path | Status | Evidence |
|---|---|---|
| MLX active (`mlx_dkv_wrapper.py`) | **Fully verified** | Dense-mode and compressed-mode outputs byte-identical to each other and to plain `mlx_lm`, at short (~20 tok) and long (~2800 tok, real compression engaged) context; correct random-code needle retrieval confirmed through the actual compression pipeline. |
| CUDA/MPS Path B (`dkv_backend.py`, AttentionInterface) | **Ran successfully once, on Qwen3.5-2B, by accident** (default-on env var) — output matched MLX/reference exactly. Never stress-tested past a ~18-token decode; never tested at long context (real compression engaging); never explicitly validated against this model's hybrid layers by anyone (its own writeup only covers Qwen2.5-0.5B). | One successful short generation. |
| CUDA/MPS Path A (`dkv_attention.py` legacy monkeypatch, this session's fixes) | **Verified, isolated from Path B** — forced `DKV_USE_ATTENTION_INTERFACE=0`, output matches all other references exactly. This is the path most of this session's CUDA effort went into. Only short-context/basic-decode confirmed; long-context/real-compression not yet retested for this path specifically (only MLX has that level of verification so far). | One successful isolated short generation, text-compared against 3 independent references. |

## 5. Parity audit — 3 Explore agents, all completed in full (verbatim findings preserved below, condensed headers only; see conversation transcript for full detail if this summary loses something)

Scope: compared `mlx_dkv_wrapper.py` against the CUDA-side files
(`kv_runtime_manager.py`, `dkv_attention.py`, `hf_dkv_wrapper.py`, plus modules they
delegate to: `native_core/compression/lowrank.py`, `residual_capture.py`,
`streaming_sparse_ingest.py`, `native_core/srl/query_router.py`,
`native_core/paging/paged_kv_store.py`, `runtime/native_block_pool.py`). **Important
caveat**: none of these agents knew about Path B (`dkv_backend.py`) — they compared
MLX against Path A (the legacy monkeypatch). Path B's own internals
(`_dkv_decode_forward_impl`/`_dkv_prefill_forward_impl` in `dkv_attention.py`) were
**not** covered by this audit and should be, in a follow-up.

### Agent 1 — compression internals (SVD/rank/residual policy)

Key structural fact: "the CUDA path" for compression is actually **3** independently-
implemented functions with different MLX-parity levels (`_compress_block_sync` <
`_compress_blocks_batch` < `compress_layer_blocks_gpu`, in that order of
completeness) — a fix ported to one doesn't necessarily reach the others.

1. **CUDA is ahead of MLX**: a hard "never lossily compress this block" mechanism
   (`skip_compression`/`force_exact`, `streaming_sparse_ingest.py:767-840`) for blocks
   containing long digit runs or short alphanumeric codes — MLX has zero equivalent,
   only a soft ranking boost within a limited residual budget. A code comment
   explicitly says this fixes **"the CUDA-vs-MLX random-code retrieval gap"** — i.e.
   this is a confirmed, validated case of CUDA being the more-correct side. **Action:
   port this to MLX.**
2. Block-pool eviction: CUDA has a reversible GPU→CPU paging tier
   (`PagedKVStore`, LRU, background thread); MLX's flat `DKV_MAX_BLOCKS` cap falls
   back to **irreversibly discarding** the oldest block's data in-place when full.
   CUDA's is safer/more complete; no comment suggests MLX's destructive fallback was
   deliberate policy (labeled "rare" in-line).
3. V-side rebalancing (`DKV_V_SCALE`/`v_gain`): MLX has it in all 3 of its compress
   call sites; CUDA only in the GPU-prefill-only path (`lowrank.py`), explicitly
   commented as "ported from MLX" — i.e. CUDA authors already consider MLX
   authoritative here, the port just didn't reach the other 2 CUDA paths.
4. Residual "coverage quota" (`DKV_RESIDUAL_COVERAGE_FRAC`): same pattern as #3 — MLX
   uniform, CUDA only in the GPU-prefill path. MLX's own docstring documents the
   concrete bug this fixes ("boosting needle digits evicted the adjacent 'TA' row at
   16k/0.9").
5. Owner/table/relational-capture boosting: **well-ported** to 2 of 3 CUDA paths via
   a shared `residual_capture.py` (explicitly a "pure-Python port", closes a
   documented audit finding "CUDA lacks the boost machinery entirely"). The
   "experimental relational edge capture" sub-piece was never ported — but MLX itself
   documents that piece as "EXPERIMENTAL, DEFAULT OFF, empirically WEAK", so this
   isn't a real gap.
6. Rank allocation: default schedule matches exactly between MLX and CUDA. But CUDA
   has an **unvalidated, unconditional** rank-boost (`DKV_RANK_BOOST`, +50% rank for
   content matching digit/math/definition heuristics) that MLX has zero equivalent
   of — and the CUDA code's own comment admits *"its accuracy cost has never been
   measured"* and the predicate "fires on ~100% of technical-prose blocks in
   practice" (i.e. it's nearly always-on for CUDA, never-on for MLX). **Flag as
   unresolved — needs a decision, not a blind port either direction.**
7. Both sides use a Gram-matrix shortcut before the small SVD, but CUDA's version
   (`torch.linalg.eigh` on the Gram matrix, toggleable, A/B-validated with a cited
   speedup + equal reconstruction error, has an exact-SVD fallback on failure) is
   more defensively engineered than MLX's (unconditional `mx.linalg.svd` on the Gram
   matrix, no toggle, no fallback) — and only reaches 1 of CUDA's 3 compress paths.
8. Anchor/landmark selection: CUDA re-scores tokens and can swap in a new anchor;
   MLX always anchors at the block's first token. No comment justifies either as
   "correct" — this is a design choice to make deliberately, not a bug to silently
   fix either direction.

### Agent 2 — decode/routing/prefill kernels

Ordered by risk of *silently wrong* (not just different-but-correct) output:

1. **Highest risk finding**: CUDA's default decode-time top-K block router
   (`DKV_ROUTER="residual"`, delegates to `native_core/srl/query_router.py:
   route_blocks_relevance`, itself commented as *"Direct port of
   mlx_dkv_wrapper._block_relevance_residual"*) force-includes an attention sink the
   MLX router has no equivalent forcing for, AND — more seriously — a sibling,
   disabled flag (`DKV_DECODE_PRUNE_K`) that uses the **exact same underlying
   function** carries the comment *"CONFIRMED DEAD END on A100 (2026-07-18)... the
   CUDA residual router drops answer-critical blocks at K=16 where MLX's does not."*
   Since the live default path uses that same function, **this risk isn't just
   theoretical for the abandoned flag** — it plausibly affects CUDA's actual default
   routing once enough compressed blocks accumulate. **Needs a real
   re-investigation**, not just a note.
2. Sparse/lego prefill (avoiding O(N²) dense prefill for long prompts): **completely
   absent from CUDA.** CUDA's own comment admits it "re-assembles history blocks,
   re-applies RoPE to all history, and runs an EAGER matmul+softmax+LSE-merge every
   chunk" — full dense attention, always, regardless of length. A `DKV_CONTIGUOUS_PREFILL`
   experiment exists but is explicitly flagged "UNVALIDATED on GPU" and still isn't
   sparse (just reorganizes the same dense computation). **This is a major,
   deliberate-scope-decision-needed gap**, not a quick port.
3. "High-Quality Mode" (`DKV_HIGH_QUALITY_ROUTING`, force attend-all): MLX has it
   (explicitly built to parity-match a *third*, C++ native implementation); CUDA has
   no equivalent escape hatch to force exhaustive attention for a ground-truth
   comparison baseline.
4. LSE-combination of compressed vs dense attention: the core bias formula
   (`DKV_SPARSE_BIAS`) is well-ported (identical doc comment in both files verbatim).
   But CUDA has **3 different combine shapes** depending on runtime conditions
   (2-way, 3-way when factual-store matches exist with an ad hoc extra similarity
   boost term MLX has no counterpart for, or a separate fused-Triton-kernel path that
   silently gets swapped out for a slower one if `DKV_SPARSE_BIAS` is set to anything
   nonzero) vs MLX's **always exactly one shape**. Any numeric-parity test needs to
   control for all of this.
5. "Decode cache" (`DKV_DECODE_CACHE` vs `DKV_DECODE_CACHE_CUDA`): **false friends** —
   same name, different semantics. MLX's version changes routing freshness (staleness
   up to N tokens between re-routes). CUDA's version only caches a gather/index-select
   step, never re-routes or re-materializes — porting one side's expectations to the
   other by name alone would be wrong.
6. NaN/Inf guarding: MLX proactively zeroes NaN/Inf at multiple intermediate points
   before they can poison a merge. CUDA only guards Inf (never NaN) and only inside
   the LSE merges — a NaN produced elsewhere propagates all the way through `o_proj`
   with just a debug print, no correction.
7. Minor: GQA repeat-then-matmul (CUDA, always materializes a repeated K/V copy) vs
   broadcast-without-materializing (MLX's hot paths) — perf-only, not correctness.
   Causal masking: MLX always builds an explicit mask array; CUDA relies on
   `is_causal=True` in places, which the code's own comments document as a
   previously-hit hazard (upper-left vs lower-right alignment on non-square Q/K)
   requiring hand-built workarounds elsewhere in the same file.

### Agent 3 — optional-subsystem presence/absence

Of 11 areas checked: **~half are genuine, working, independently-verified ports**
(factual-store/SRL, Context-Aware Decoding, the core owner/table/relational-capture
compression boost, repetition-loop/SFA decode safeguards, session
snapshot/eviction). Two are **clean, total MLX-only gaps** with no CUDA analogue
anywhere: **Lego streaming prefill** (~400+ lines, a dozen env vars — ties into
Agent 2's finding #2) and **quantized prefill KV cache** (`DKV_PREFILL_CACHE_BITS`,
small but a real long-context-memory feature). One (instruction pinning) is ported in
source but **inert on CUDA** because its prerequisite (sparse prefill) was never
ported. One thing runs the *other* direction — CUDA's attention-interception layer
has "RC5 comparison-sequencing" logic (`advance_comparison_entity`) that MLX sets up
the identical state for but **never actually calls** — i.e. a genuine MLX bug where
multi-entity comparison mode locks to entity 0 and never advances (flagged as
uncertain/needs-followup by the agent, not asserted with full confidence). Speculative
decoding exists as a real subsystem but is architecturally centered on the PyTorch
wrapper (not really "in" either wrapper file). Prefix-sharing
(`SharedPrefixManager`) is fully defined but **never instantiated/called anywhere in
the repo** — dead code on both sides equally, not a parity gap.

## 6. Test suite status (`benchmarks/results_qwen3.5-2b/`)

Ran via `benchmarks/run_qwen35_2b_suite.sh` (all phases) then
`benchmarks/run_qwen35_2b_suite_part2.sh` (resume script, starts from B3) after the
first run silently died (OS-level SIGKILL from memory pressure, not a real failure —
see §8 lesson). Both scripts still exist in `benchmarks/` and are safe to re-run or
adapt (they already thread `--pytorch-dense-model-id "Qwen/Qwen3.5-2B"` and
`--dense-model-id "$MODEL"` correctly per the §3.5 fixes).

| Phase | Status | Notes |
|---|---|---|
| A1_niah_sweep | **OK** (434s) | `active` passes at 4k/8k/16k/32k, correct needle recall every time. `dense` (uncompressed fp16 PyTorch baseline) **OOMs at 4k already** (vs Qwen2.5 baseline OOMing at 16k+) — a real, meaningful finding: Qwen3.5-2B's heavier per-token footprint (larger vocab, vision tower loaded even though unused, hybrid-layer overhead) makes the "dense OOMs, DKV doesn't" story even stronger for this model, not weaker. |
| B1_multi_needle | OK (172s) | |
| B2_multihop | OK (187s) | |
| B3_perplexity | FAIL, then FAIL again on retry | Both failures were **SIGKILL (137)** from concurrent memory pressure (running alongside a separate heavy CUDA/MPS test at the same time) — NOT the underlying bug, which the §3.5 fix already confirmed working (sane PPL, 0% delta, for context=4000, before the process got killed on a later context length). **Needs a clean re-run, alone, no concurrent heavy processes.** |
| B5_signal_ablation | OK (97s) | |
| B6_lego_mem | FAIL | Same SIGKILL-137/memory-pressure cause. **Needs clean re-run.** |
| B7_latency_breakdown | Unknown — was starting when the environment reset happened | **Needs re-run.** |
| binding_table, pareto_curve, extreme_context, niah_recall, prose_fact_recall, bench_gqa, C2 (RULER), C1 (LongBench) | **Never run** | Still pending — these are in `run_qwen35_2b_suite.sh`/`_part2.sh`, just never reached. |

## 7. Git state

Everything described above (MLX fixes, CUDA Path A fixes, `bench_worker.py`/
`run_ppl_mlx.py` fixes) landed in commit `8f9c7f1` — **but that commit's message and
description describe a much larger set of changes than this session made** (it also
bundled the Path-B `dkv_backend.py` work from the earlier, separate session, which
was sitting uncommitted in the same working tree). Do not assume commit `8f9c7f1`'s
message accurately scopes what to review — read the actual diff
(`git show 8f9c7f1`) or this document instead.

`things_after_paper.md` (repo root) is gitignored and won't show up in `git status`/
`git diff` — remember to read it directly; it is the only record of Path B's design
rationale and its (Qwen2.5-only) benchmark numbers.

## 8. Lessons for the next session

1. **Don't run the MLX benchmark suite concurrently with a heavy CUDA/PyTorch/MPS
   test on this machine.** It's an 8GB Mac; loading a second full fp16 model while
   the suite is mid-run caused observed free memory to drop to ~44MB and killed two
   suite phases with SIGKILL (137), not a real bug. Serialize heavy work.
2. **Check which of Path A / Path B is actually active before drawing conclusions
   about "the CUDA path."** `DKV_USE_ATTENTION_INTERFACE` defaults to `"1"` — Path B
   runs unless explicitly disabled. A successful test result under default settings
   says nothing about Path A (this session's `dkv_attention.py` monkeypatch fixes)
   without explicitly setting `DKV_USE_ATTENTION_INTERFACE=0` to isolate it.
3. **MPS is a legitimate stand-in for CUDA verification on this Mac** — `hf_dkv_wrapper.py`
   supports `device="mps"` as a first-class path, and it exercises the *same*
   `dkv_attention.py`/`dkv_backend.py` code CUDA would run, just on a different
   accelerator backend. Use it; don't assume CUDA-only code is untestable here.
4. When a background bash task or subagent needs to survive/resume across a session
   boundary, there is no automatic mechanism for that — write state to a file (like
   this one) and/or memory before the session ends. An "orphan" task notification
   (`__orphan_summary__`) after a reset means the harness lost track of a background
   shell job; it does not mean the job's file-level side effects (partial log files,
   etc.) are gone — check for them before assuming a re-run is needed from scratch.

## 9a. CUDA-only fix pass (post-audit, MLX untouched)

Per explicit user instruction: fixed the CUDA-side issues the 3 parity-audit agents
found (§5), staying entirely within CUDA files (`native_core/compression/lowrank.py`,
`native_core/kv_runtime_manager.py`, `native_core/srl/query_router.py`,
`runtime/dkv_attention.py`, `serving/hf_dkv_wrapper.py`) — MLX (`mlx_dkv_wrapper.py`)
was not touched, and Path B (`dkv_backend.py`, the other session's work) was
deliberately left alone too. Every fix verified for syntax
(`python3 -c "import ast; ast.parse(...)"`); several also verified empirically via
MPS (noted per-item below).

1. **v_gain V-side rebalancing** ported into `compress_lowrank` (sync path) and
   `compress_lowrank_batch` (the low-level SVD primitive underlying
   `_compress_blocks_batch`, i.e. the actual day-to-day decode-triggered path) —
   both previously lacked it; only the GPU-prefill-only path had it. Applied before
   the SVD, divided back out of the V-factor after, exactly mirroring the existing
   GPU path's already-validated approach. `compress_lowrank_batch`'s external
   contract (raw factors, no residual logic) is unchanged, so `_compress_blocks_batch`
   needed zero changes to benefit.
2. **Residual coverage-quota** (`DKV_RESIDUAL_COVERAGE_FRAC`) — the `_topk_with_coverage`
   helper was nested inside `_compress_layer_blocks_gpu_inner`; promoted to
   module-level in `lowrank.py` (removed the now-duplicate nested copy) and wired
   into both `compress_lowrank` and `_compress_blocks_batch`, which previously used
   plain `torch.topk` with no coverage concept at all.
3. **`DKV_RANK_BOOST` gating inconsistency**: `lowrank.py`'s `_block_boost_rank`
   (GPU path) already respected `DKV_RANK_BOOST=off`; the inline copy in
   `kv_runtime_manager.py`'s `_preprocess_block_for_compression` (used by both the
   sync and batch paths) applied the same 1.5x boost **unconditionally**, ignoring
   the env var entirely. Added the same gate check. Confirmed this doesn't affect
   pool-sizing math (`kv_runtime_manager.py` ~line 554 already correctly respects
   the same env var for capacity planning — only the per-block decision was the
   outlier).
4. **Double-normalization "inconsistency" — investigated, found to be a false
   positive.** The audit agent read `compress_lowrank` in isolation and correctly
   observed it has no internal per-token normalization step — but traced further
   (this session, not the agent) into its actual production caller
   (`_compress_block_sync` → `_preprocess_block_for_compression`) and confirmed the
   per-token normalization already happens there, before `compress_lowrank` is ever
   called. Production behavior already matches MLX's two-stage (per-token, then
   global scale) pattern. No change made — verify this reasoning before trusting an
   agent's file-local reading over the actual call graph in future work.
5. **Sink-forcing removed from `query_router.py`'s `route_blocks_relevance`** — the
   function's own docstring claims "plain top-K, direct port of
   mlx_dkv_wrapper._block_relevance_residual," but the code unconditionally
   reserved one top-K slot for a fixed "sink" block (lowest anchor_indices)
   regardless of relevance, with no comment explaining why (unlike virtually every
   other CUDA-specific addition in this codebase, which documents its "fixes X"
   rationale). A sibling flag (`DKV_DECODE_PRUNE_K`, disabled) using this same
   function carries a comment confirming it drops answer-critical blocks at matched
   K on A100. Removed the deviation to match the documented contract.
6. **NaN guarding added to both LSE-combine sites** in `dkv_attention.py` (3-way
   dense+sparse+facts, and 2-way dense+sparse). Previously only `torch.isinf(...)`
   was checked; `torch.maximum` propagates NaN from either operand, so a single NaN
   in any lse_* tensor would poison the shared `lse_max_masked` and thus every
   branch's weight. Changed each `torch.isinf(X)` guard to `~torch.isfinite(X)`
   (catches both), and added a final `torch.nan_to_num` safety net on the combined
   output before it's used, matching MLX's defense-in-depth pattern.
7. **Quantized prefill KV cache** (`DKV_PREFILL_CACHE_BITS`, MLX parity naming) —
   ported to `hf_dkv_wrapper.py`'s prefill loop. This surfaced a **separate, more
   serious pre-existing bug while implementing it**: the prefill chunk loop never
   passed `past_key_values` at all between chunks (each chunk got an implicit fresh
   `None`), so for ANY model with non-DKV-managed layers (i.e. any hybrid
   architecture's linear-attention layers), state was silently lost between chunks
   on any prompt longer than one `PREFILL_CHUNK` (~512 tokens) — MLX's equivalent
   (`_get_or_create_prefill_cache`) already builds one cache up front and threads it
   through every chunk correctly. Fixed by constructing an explicit
   `DynamicCache(config=self.model.config)` (or `QuantizedCache(...)` when
   `DKV_PREFILL_CACHE_BITS` is 4/8) once before the loop and threading it through
   every chunk call.
   - **First attempt regressed Qwen3.5-2B**: called `DynamicCache()` with no `config`
     argument, which crashed (`IndexError: list index out of range` in
     `transformers/cache_utils.py`'s `update_conv_state`) the moment a
     linear-attention layer tried to use it — a bare `DynamicCache` doesn't know a
     hybrid model's per-layer types without `config=`. Fixed by passing
     `config=self.model.config` (the constructor already supports this).
   - **Verified via MPS, twice**: (a) a 1768-token prompt — comfortably multi-chunk
     — completed with no crash and correctly retrieved a needle passcode planted in
     the prompt, confirming the chunk-continuity fix is real and correct on the
     exact hybrid model this session is about; (b) `DKV_PREFILL_CACHE_BITS=8`
     correctly hit the defensive fallback path (clear warning printed, fell back to
     `DynamicCache`, generation completed with output identical to every other
     reference) instead of crashing.
   - **Important finding from (b): quantized prefill cache is architecturally
     incompatible with hybrid models, independent of whether `hqq`/`quanto` are
     installed.** The actual exception caught was not "package missing" but
     transformers' own validation: `QuantizedCache is only supported for models
     with only full attention layers. We found the following invalid layer types:
     {'linear_attention'}`. So for Qwen3.5-2B specifically (and any other hybrid
     architecture), `DKV_PREFILL_CACHE_BITS=4`/`8` will **always** fall back to the
     plain fp16 cache, regardless of environment — this isn't a gap to close, it's
     a hard upstream constraint. The feature should still work for plain
     (non-hybrid, attention-only) models once `hqq` or `quanto` is installed, but
     that specific combination remains genuinely untested (no such model was handy
     to verify against in this session).

## 9b. Sparse-prefill port + defaults + long-context retest (this continuation)

Per explicit user instruction: "do Sparse/Lego prefill port to CUDA and make
DKV_RANK_BOOST default and anchor/landmark selection same as mlx." All three done;
retesting the "long context with real compression engaged" gap flagged in §4/§9-1
above surfaced two more real bugs and one real, unresolved, pre-existing correctness
gap.

1. **`DKV_RANK_BOOST` default → `off`** (`lowrank.py`'s `_block_boost_rank` and
   `kv_runtime_manager.py`'s pool-sizing check) and **anchor/landmark selection made
   flat** (`kv_runtime_manager.py`'s "Learned Landmark Scoring" gated behind
   `DKV_LANDMARK_RESCORE=0` by default — always picks the first token, matching
   MLX). Both match MLX's defaults now.
2. **Sparse prefill ported to Path A only** (`dkv_attention.py`'s new
   `_sparse_prefill_filter_blocks`: sink block + top-K anchor-relevance filter over
   `history_blocks`, `DKV_SPARSE_PREFILL=1` default, `_KMIN=8`/`_FRAC=0.25` matching
   MLX's `DKV_SPARSE_PREFILL` semantics). Path B's prefill was deliberately **not**
   touched (uncertain cross-chunk semantics, default-active, out of scope per the
   "don't touch Path B" instruction). "Lego" (the deeper memory-bounding extension)
   was **not** attempted — MLX's own `DKV_LEGO_PREFILL` is itself default-off/
   experimental, so parity only required the sparse (compute) piece.
   - **Verified no regression**: on an 8217-token prompt (~32 blocks @ block_size=256,
     comfortably engaging the filter), output is byte-identical with
     `DKV_SPARSE_PREFILL=1` vs `=0`. The filter does not change behavior for this
     prompt shape.
3. **Two real, previously-undiscovered Path A crash bugs found and fixed** while
   retesting at long context (exactly the gap §4/§9-1 flagged as untested) — both are
   in the shared `dkv_forward`/`apply_rotary_pos_emb`, so every call site benefits:
   - **Gated-attention `q_proj` shape crash**: Path A assumed `q_proj`'s output width
     is `num_heads*head_dim`, but Qwen3.5 packs `[query|gate]` per head (2x width,
     confirmed against the HF reference `Qwen3_5Attention.forward`, which does
     `torch.chunk(q_proj(x).view(..., -1, head_dim*2), 2, dim=-1)` then
     `attn_output * sigmoid(gate)` before `o_proj`). MLX's `attention_forward` /
     `_resolve_attn_dims` already handled this; Path A never did. Fixed by mirroring
     the exact reshape-then-chunk-on-last-axis split (a flat `chunk(2)` on the
     un-reshaped tensor is WRONG — it would split heads 0-3 from heads 4-7 instead of
     each head's query from its own gate) and applying `sigmoid(gate)` before
     `o_proj` at all 3 of Path A's return sites (bypass-dense, decode,
     prefill-engaged).
   - **Partial-RoPE shape crash**: Qwen3.5 sets `partial_rotary_factor=0.25`, so
     `cos`/`sin`'s last dim is 64, not the full `head_dim=256` — `apply_rotary_pos_emb`
     unconditionally did `(q*cos)+(rotate_half(q)*sin)` on the full-width q/k, which
     only works when rotary is applied to 100% of head_dim. MLX gets this for free
     (its `self.rope` is `mlx_lm`'s own native RoPE module, constructed with the
     correct partial `dims` already, so DKV never had to reimplement the slicing) —
     this is a case where Path A's manual reimplementation needed logic MLX's
     architecture never required. Fixed to match HF's own reference `apply_rotary_pos_emb`
     for Qwen3.5 exactly: slice to `rotary_dim = cos.shape[-1]`, rotate only that
     slice, concatenate the untouched remainder back. Backward-compatible by
     construction (when `rotary_dim == head_dim`, i.e. full rotary, `q_pass` is empty
     and behavior is identical to before). Also ported the same slicing into a new
     `_apply_rope_single` helper and used it at the 2 production dense-history K
     reconstruction sites that bypassed the shared function with their own inline
     `(k*cos)+(rotate_half(k))*sin`, plus the debug-only `DKV_VALIDATE_SRL=1` site (all
     had the identical latent bug). **Not fixed** (found but out of scope, narrow and
     already-known-inactive): the Factual Store's own inline K rotation
     (`dkv_attention.py` ~line 1633-ish, `dense_k_half`/manual rotate-half
     reimplementation) has the same bug class, but the Factual Store is
     `DKV_FACTUAL_STORE=0` by default and independently documented as net-negative/
     parked (`project_parked_systems_factual_ab.md`) — only matters if someone
     re-enables it on a partial-rotary model.
   - Both verified via the exact repro that found them: 8217-token MPS run, no crash,
     coherent fluent output, correctly identifies the needle starts with "ZEBRA" (see
     item 4 for why it doesn't get the rest).
4. **Root-caused (user asked to continue the investigation directly): Path A's
   retrieval-fidelity regression vs dense at long context.** This is likely exactly
   why §4's "Path A long-context" cell was still unverified — nobody had gotten this
   far before. Repro unchanged from the initial finding (8217-token prompt, needle
   `ZEBRA-4471-QUARTZ` at the very start, question at the very end,
   `apply_chat_template` required — instruct/thinking model, raw prompts just get
   echoed back). Ruled out first (don't re-derive): not the sparse-prefill filter
   (identical output with `DKV_SPARSE_PREFILL=1`/`=0`); not routing (identical output
   with `DKV_TOPK_BLOCKS=0`, i.e. attend-all); not the compression rule (`Rule 1b`
   correctly force-exact-stores the needle's block, confirmed via `DKV_TELEMETRY=1`).
   Isolated via a sequence of print checkpoints added directly in `dkv_forward` and
   `get_streaming_blocks` (added under `DKV_DEBUG_ROPE_SHAPE=1`, all since removed).

   **PRIMARY root cause, FIXED**: the decode-time "does this session have any
   compressed history at all" checks hardcoded `layer_idx=0` as a proxy —
   `dkv_attention.py`'s decode-bypass check (`kv_manager.get_streaming_blocks(sid,
   0)`) and its prefill-side twin, plus `kv_runtime_manager.py`'s
   `get_session_sequence_length` and `finalize_srl_index` (all 4 sites). This proxy
   is only valid when layer 0 is itself an attended (compressed) layer — true for
   every non-hybrid model, **false for Qwen3.5**, where layer 0 is `linear_attention`
   and never compressed. `get_streaming_blocks(sid, 0)` therefore always returned an
   empty list regardless of how much real compressed history existed at layers 3, 7,
   11, 15, 19, 23 — so **every single decode step was silently routed onto the
   pure-dense bypass path** (`dkv_attention.py`'s `is_bypassed` branch, a fresh
   `DynamicCache()` with zero knowledge of the compressed pool), never touching the
   ~8000 tokens of compressed context at all. Confirmed empirically: a checkpoint
   print right at the bypass decision showed `has_blocks=False` at every one of the 6
   attended layers, every decode step, for the entire generation.
   Fixed: `dkv_attention.py`'s two call sites now pass `captured_layer_idx` (this
   closure's own attended layer — always valid, since only attended layers get
   `dkv_forward` patched in) instead of hardcoded `0`. `kv_runtime_manager.py` gained
   a new `_any_attended_layer_with_blocks(session_id)` helper (scans all layers,
   returns the first with content, falls back to `0`) wired into
   `get_session_sequence_length` and `finalize_srl_index`, both of which had the
   identical hardcoded-0 bug and are used well beyond this one check (sequence-length
   queries feed pool sizing, engage-threshold checks, and more — this bug's actual
   blast radius on hybrid models is likely wider than just this one repro).
   **This is squarely the class of bug the original task asked to audit for**
   ("no hardcoded qwen stuff... dkv needs to work for any model, any family") — a
   layer-0-is-representative assumption that's implicit and invisible on every model
   tested before Qwen3.5, and silently wrong the moment a hybrid architecture shows up.

   **SECONDARY bug, EXPOSED but only partially mitigated**: fixing the above means
   decode now genuinely reaches the compressed-block reconstruction kernels for the
   first time on this model — which surfaced that the **compiled Metal shader**
   (`native_core/dkv_core/metal/dkv_decode.metal`) has **zero partial-RoPE
   awareness**: `AttentionParams` has no `rotary_dim` field, and every rotation site
   in the shader (~8 occurrences) does `for d in 0..D: ... cos_anc[k*D+d] ...
   partner = (d<D/2) ? d+D/2 : d-D/2` using the FULL `head_dim` (256) unconditionally
   — there is no concept of only 64 dims being rotary. The Python-side buffer feeding
   this kernel (`_cos_anc_2d`/`_sin_anc_2d`, built in `dkv_attention.py` around the
   `cos_sliced_arg`/`cos_sliced_cached` construction, two sites) was gathering only
   `[K, rotary_dim=64]` and then `.view(..., head_dim=256)` — a hard crash once this
   code path is actually reached (which, again, it never was before fix #1). Applied
   an interim, crash-safe, pure-Python fix at both sites (main decode-time
   `cos_sliced_arg` construction + the prefill continuation `_project_then_attend_history`
   equivalent): pad the gathered `[K, rotary_dim]` tensor to `[K, head_dim]` with
   `cos=1, sin=0` for the tail before the `.view()`. This is provably correct for the
   padded region (raw*1 + partner*0 == raw, an exact identity pass-through regardless
   of which partner value gets zeroed out) but does **not** fix the genuinely-rotary
   sub-range: the shader's fixed `D/2=128` pairing is still used for dims 0-63,
   instead of the mathematically-correct `rotary_dim/2=32` pairing a sliced
   `rotate_half` would use. Also fixed one more instance of the same
   full-`head_dim`-assumption bug in a different form — the pre-allocated-workspace
   dense-window rotation (`dense_k_half`/`half_d = head_dim // 2`,
   `dkv_attention.py` ~line 1914, the in-place-buffer variant of the same math the
   already-fixed `_apply_rope_single` sites use) — sliced to `rotary_dim` for the
   rotation and pass-through-copies the remainder, fully correct (not just
   crash-safe) since this one doesn't touch compiled native code.
   **Empirical result**: output changed from a clean early stop (`'...ZEBRA'` + EOS)
   to degenerate repetition (`'...Z!!!!!!!!!!...'`) — no crash, and a *different*
   failure mode than before, consistent with "compressed pool is now genuinely
   reached, but the anchor/residual key reconstruction is numerically corrupted
   enough to destabilize decoding" rather than "never sees the pool at all." This is
   real forward progress (the dominant bug is fixed and verifiably changed behavior),
   but **not a full fix** — full correctness needs the Metal shader itself (plus its
   ATen C++ counterpart, `decode_attention.cpp` — confirmed to have the identical
   hardcoded `half_d = D/2` pairing at lines 58-60 and 390-392, and the same
   `has_rope` full-`D` rotation at lines 124-148) to gain a real `rotary_dim`
   parameter and correct pairing, then a native-extension rebuild
   (`setup.py`, produces `dkv_core.cpython-314-darwin.so`) — a meaningfully
   larger-scope, higher-risk change (recompiling a shared binary every model/path
   depends on) than any other fix this session, and was NOT attempted — flagged to
   the user for an explicit decision rather than proceeding unprompted.

## 9c. §9b-4 continued: found and fixed 8 MORE real bugs, root cause still not fully
isolated — READ THIS FIRST before touching partial-RoPE / Path A decode again

User said "keep going, switch strategy" on §9b-4 (Path A partial needle-recall). This
section is the result: a huge amount of real, verified progress, but the exact
symptom (needle `ZEBRA-4471-QUARTZ` → only `ZEBRA` recovered, then decode collapses
into repeating `!`) is **still not fully resolved**. Read this whole section before
resuming — the control-flow discovery in item 1 invalidates assumptions made earlier
in §9b.

1. **CRITICAL correction to §9b's mental model — Metal/ATen path assumptions were
   WRONG.** Spent a long time fixing `_decode_attention_metal`/`decode_attention_aten`
   dispatch and the "has_dense/has_comp" 4-way branch in `dkv_attention.py` (all real,
   all still worth keeping — see item 3 below) believing THAT was the code executing
   for this test. It is not, on this Mac, ever. The actual control flow:
   - `kv_runtime_manager.py` (~line 754-759) **auto-sets `DKV_MPS_APPROXIMATE_ATTN=1`
     into `os.environ` as a side effect of construction**, if the var was ever unset —
     printing "[DKV] Enabled MPS approximate attention fast-path..." at startup (a
     banner seen throughout this whole session and misread as informational).
   - This makes `_is_mps_decode = True` (`dkv_attention.py`, the check right after the
     "MPS Fast Path (Phase 34)" comment) for literally every test in this
     investigation that didn't explicitly override the env var — including every one
     used to validate the (real, but irrelevant) Metal/ATen fixes.
   - `if _is_mps_decode:` wraps the ENTIRE "has_dense/has_comp" 4-way dispatch
     structure, the `_decode_attention_metal`/`_decode_attention_aten` calls, and the
     NaN-guarded LSE combine — none of that code path is reachable while
     `_is_mps_decode=True`. Its sibling `else:` (where that code actually lives) is
     genuinely dead for this whole investigation.
   - The REAL path: `_is_mps_decode=True` → `_DKV_CORE_AVAILABLE and
     hasattr(_dkv_core, "fused_decode_attention_combined")` (both True, confirmed
     live) → `_dkv_core.fused_decode_attention_combined(...)` → on `device.is_mps()`,
     internally calls `decode_attention_metal` directly inside
     `decode_attention.cpp` (no Python-level flag check at all — `_DKV_HAS_METAL_ATTN`
     is irrelevant here, it only gates the dead branch above).
   - **This means the earlier "force `_DKV_HAS_METAL_ATTN=False` to test the Python
     fallback" isolation test (§9b) was invalid** — that flag doesn't affect this
     call chain at all, so "all 3 kernels agree" was actually "the same one kernel ran
     three times." The Metal/ATen fixes from that phase are still correct and worth
     keeping (matches MLX behavior properly, needed on real non-Mac CUDA hardware
     where `_is_mps_decode` can never be true), just not what's failing here.
   - Confirmed via direct instrumentation (since removed): `_is_mps_decode=True`,
     `_DKV_CORE_AVAILABLE=True`, `has_combined=True` on every layer, every decode step.

2. **Compression-side sanity check (all clean)**: dumped `lr_delta.residual_K_values`
   / `residual_K_positions` at the moment `_compress_block_sync` finalizes the
   needle's block (anchor=0) — `force_exact=True` correctly derived from
   `block.skip_compression`, all 256 positions requested `[0..255]`, values
   `torch.isfinite(...).all() == True`, non-degenerate norms (2.3-6.8 range across
   the 6 full_attention layers). `_get_rotated_anchor_k` (called from
   `finalize_compressed_blocks`/`write_block`) is a **deliberate no-op** — returns
   `anchor_k` unchanged, with a comment already documenting exactly the
   double-rotation risk this session was worried about; ruled out.

3. **Inputs to the REAL call (`fused_decode_attention_combined`) verified sound**:
   instrumented every tensor immediately before the call — `_ca`/`_sa` (anchor
   cos/sin, padded to head_dim) non-empty (`numel=7680`, i.e. 30 blocks × 256) and
   finite; `_cos`/`_sin` (dense-window cos/sin, also padded by this session's fix)
   scale correctly with `L_dense` at exactly 256 per token (confirms the head_dim
   padding fix from §9b-4 item 3 is working); `_res_val_K` non-empty (16.7M elements)
   and finite; `_dk` (dense K workspace) non-empty. **Every single input is
   structurally and numerically sound.** Yet the output is still wrong. This strongly
   localizes the remaining bug to inside `decode_attention_metal`'s Metal shader
   arithmetic itself (or the C++/ObjC++ binding layer immediately around it),
   not anything upstream in Python.

4. **8 more real, verified bugs found and fixed while chasing this** (independent of
   whether they were the root cause of the symptom above — all confirmed correct via
   direct code reading, all in CUDA-side `ACTIVE_RUNTIME` files, none touch MLX or
   Path B):
   - Same partial-RoPE bug class (full-`D`/`D/2` pairing instead of
     `rotary_dim`/`rotary_dim/2`) found and fixed in THREE more Python functions in
     `native_core/sparse_decode/triton_fused_decode.py` that were not touched in
     §9b-4's first pass: `_pytorch_vectorized_sparse_attn_decode` (4 sites: VK/anchor
     rotation, fact-anchor-override K, dense-window K — this is the
     `HAS_TRITON=False` fallback for `native_triton_sparse_attn_decode`, itself only
     reachable when `_is_mps_decode=False`, i.e. currently dead on this Mac but real
     for `DKV_MPS_APPROXIMATE_ATTN=0`/testing scenarios), `fused_decode_mps` (1 more
     site missed the first pass — the fact-anchor-override `K_exact` rotation), and
     `_gather_routed_blocks_for_kernel` (4 sites: VK/anchor rotation, plus BOTH
     branches of `DKV_RESIDUAL_EXACT_ROPE` residual-K rotation — **this one matters
     for real CUDA+Triton hardware**, it's what `native_triton_sparse_attn_decode`
     calls to prep data for the actual Triton GPU kernel when `HAS_TRITON=True`,
     i.e. on real CUDA, not just Mac). Added a shared `_partial_rope_apply(x, cos,
     sin)` module-level helper (mirrors `apply_rope_to_keys` in `decode_attention.cpp`)
     and routed all of these through it instead of ad-hoc inline rotation.
   - Removed the two now-dead local `rotate_half` closures inside `fused_decode_mps`
     and `_pytorch_vectorized_sparse_attn_decode` (every call site now goes through
     the module-level helper instead).
   - All of these are genuinely exercised on real CUDA/Triton hardware even though
     none of them explained this Mac-specific test's symptom — worth keeping
     regardless of how item 3's Metal-kernel mystery resolves.

5. **Ruled out, with direct evidence, across this whole §9c investigation** (do not
   re-check these without new evidence): NaN/Inf anywhere in the (dead-path, but
   checked anyway) LSE combine; decode-time query position tracking (`cur_pos` in
   `hf_dkv_wrapper.py`, unchanged by any of this session's fixes, increments
   correctly); block routing/top-K selection (`DKV_TOPK_BLOCKS=0` retested
   post-bypass-fix, identical output — routing is not it); residual-capacity
   truncation (needle sits in the first ~15 tokens of its block, `max_residual_tokens`
   defaults to 64 for the "mid" preset, truncation keeps earliest positions —
   confirmed via `native_block_pool.py:506-508`'s `min(...)` + head slice); anchor-key
   double-rotation (`_get_rotated_anchor_k` is a no-op, see item 2); every Python-level
   decode kernel variant in the codebase (all now partial-RoPE-correct per item 4, and
   confirmed NOT the active path per item 1 anyway).

6. **Recommended concrete next step — build an isolated Metal kernel unit test.**
   Live-process diagnostics on the full 2B-parameter model (each iteration = full
   8217-token prefill + decode, ~2-3 min, only yields black-box "still wrong output")
   have been exhausted as a technique for narrowing this further. The right tool now:
   a small standalone Python script that calls `dkv_core.decode_attention_metal(...)`
   directly with tiny, hand-constructed tensors (e.g. 1-2 pool blocks, `D=8`,
   `rotary_dim=4`, known Q/K/V/cos/sin values chosen so the correct output can be
   computed by hand or with a 20-line NumPy/PyTorch reference), and diff the kernel's
   actual output against that reference. This isolates the Metal shader's arithmetic
   from the entire model/tokenizer/routing/compression stack and would show EXACTLY
   which term in `dkv_rope_rotate_dim` (or elsewhere in `dkv_decode.metal`) is wrong,
   instead of continuing to infer from a single collapsed token stream. Candidates to
   check first with such a test, in order of suspicion: (a) the `AttentionParams`
   struct layout / `sizeof()` between `metal_runtime.mm` and `dkv_decode.metal` — both
   were edited to add `rotary_dim` as the last `int32_t` field and should match
   byte-for-byte, but this was never verified with an actual memory dump; (b) the
   `[[buffer(N)]]` index bindings in the ObjC++ encoder (`metal_runtime.mm`) vs. the
   shader's declared buffer indices, especially since the dense-window buffers
   (`dense_K`/`dense_V`/`cos_dense`/`sin_dense`, buffers 22-25) were added in an
   earlier session before this one and could have an off-by-one; (c) whether
   `decode_attention_metal`'s C++ wrapper actually forwards `L_dense`/`has_dense`
   correctly into `AttentionParams` when called via `fused_decode_attention_combined`
   specifically (as opposed to being called directly, which may be better-tested).

## 9d. ROOT CAUSE FOUND AND FIXED — the NaN/garbage-collapse symptom is resolved.
Read this before §9c (which is now superseded on the "root cause unknown" question,
though its ruled-out list and background remain valid)

User said "keep going, switch strategy" again on §9c's dead end, then "i think first
context window has to be cleared properly then continue until its resolved
completely." This section is the result: **the root cause is found, fixed, and
empirically verified** — via the exact next step §9c recommended (isolating the
compiled kernel), just done by capturing REAL failing-run inputs/outputs instead of
hand-constructed synthetic ones (faster to get to ground truth, and guaranteed to
reproduce the actual bug).

### Method: dump real kernel inputs+outputs, inspect offline

Added a temporary env-gated hook (`DKV_DUMP_METAL_CALL=<dir>`, since removed) right
at the live `_dkv_core.fused_decode_attention_combined(...)` call site in
`dkv_attention.py`, saving every argument AND the output to `<dir>/call_NNNNN.pt` for
the first N calls of a real 8217-token test run. Loading these offline immediately
showed: **`out_val` is NaN starting from `call_00000` — the very first decode call,
first full_attention layer (3), before any routing/compression variability could
matter** — while every single input tensor (Q, dense_K/V, cos/sin, U_pool, VK_pool,
VV_pool, anchors, residuals, facts — all of them) was 100% finite, correctly shaped,
sane-valued. This conclusively localizes the bug to arithmetic *inside* the compiled
Metal shader itself, exactly as §9c-3 already suspected — but now with a
deterministic, single-call repro instead of a whole-generation black box.

### Bug 1 (the NaN root cause): `dkv_decode.metal`'s D=128 hardcoded buffers vs
Qwen3.5's D=256 head_dim

The kernel's threadgroup/register buffers were sized `128` throughout —
`threadgroup float q_shared[128]`, `threadgroup float ak_rot_shared[128]`, and the
per-thread `float thread_val[128]` — with an explicit comment "Support up to D = 128"
at the top of the shared-memory block. This was correct for every model DKV has ever
targeted before (head_dim ≤ 128) but **Qwen3.5-2B's head_dim is 256**. Every read/
write of `q_shared[d]`/`ak_rot_shared[d]`/`thread_val[d]` for `d in [128, 256)` was
touching memory outside the declared allocation — undefined behavior that manifested
as NaN (uninitialized/adjacent threadgroup memory, not zeroed between dispatches).
This is used in literally every score/value computation in the kernel (anchor score,
delta reconstruction, residual/fact overrides, dense window, final output write), so
it corrupted everything downstream once triggered.

**Fix**: bumped `q_shared`/`ak_rot_shared`/`thread_val` to `256`. To stay under
Metal's 32768-byte per-threadgroup hard limit (this pushed the total over budget),
shrank the `scores_anc_cached`/`q_proj_cached` block-count cache from 128→64→32
blocks (blocks beyond the cap already had a correct recompute-fallback path — this
only costs extra recompute for very large K, never a wrong answer). Added a loud
`if (D > 256) throw` guard in `metal_runtime.mm` so a future larger-head_dim model
fails immediately instead of repeating this silently.

**Verified**: dumped-and-reloaded `call_00000` through a genuinely-rebuilt `.so` —
`out_val` went from `nan=True` (1304/2048 elements) to fully finite on every one of
the 6 full_attention layers, first decode call, zero exceptions.

### Bug 2 (severe, independent): `rank` vs `pool_rank` stride confusion —
every block past slot 0 read a different block's data

`DKV_LAYER_ADAPTIVE_RANK` (default **on**) compresses different layers at different
ranks — for a base rank of 32: early layers 24, middle layers 48, late layers 16 (see
`get_layer_rank()` in `kv_runtime_manager.py`). The pool's `VK_pool`/`VV_pool`/
`U_pool` tensors are therefore allocated at `pool_rank` = the max across layers (48
here), but the kernel's addressing math (`slot_id * rank * n_kv_heads * D + ...` for
VK/VV, `slot_id * S_max * rank + t * rank` for U) used the flat, logical `rank` (32,
the caller-supplied value) as the **stride**, not the tensor's real width. For
`slot_id == 0` this is harmless (0 × anything == 0), but for every other slot the
computed offset silently landed on a *different, wrong* slot's row — confirmed
empirically: simulating the kernel's exact offset formula against the real dumped
`VK_pool` tensor showed e.g. slot_id=169's "read" actually lands inside slot 112's
data. Values there are finite (not NaN) but semantically wrong — this alone wouldn't
explain the original NaN, but independently corrupts every compressed block beyond
the first regardless of Bug 1.

**Fix**: added a `pool_rank` field to `AttentionParams` (both `dkv_decode.metal` and
`metal_runtime.mm` copies), derived the same way `S_max` already correctly is
(`VK_c.size(1)`, not trusted from the caller) — and switched every VK/VV/U_pool
*slot-level* stride multiplier to use it, while the `rank`-bounded loop (how many
components to actually read per slot) is untouched. `decode_attention.cpp`'s ATen
path (`_decode_attention_impl`, the non-MPS fallback used on real CUDA hardware,
untestable from this Mac) had the identical bug via `.reshape({K * rank, ...})` /
einsum calls that would have thrown a hard shape-mismatch crash the first time a
layer's real rank differed from the flat default — fixed the same way (`rank_real =
VK_pool.size(1)`, used throughout instead of the passed `rank`).

### Bug 3 (latent, not proven to trigger in this test, but real): dense-window RoPE
silently gated off by an unrelated flag

`metal_runtime.mm` computed a `has_dense_rope` local (whether the dense window has
real RoPE angles) but never actually wired it anywhere — the kernel's single
`params.has_rope` field, derived only from `cos_anc`/`sin_anc` (the *compressed-slot*
RoPE table), gated rotation for the dense window too. A decode step with zero
routed compressed slots but a populated dense window would skip rotating the entire
dense window. Fixed by adding a genuinely separate `has_dense_rope` field, used only
at the dense-window `dkv_rope_rotate_dim` call site.

### Bug 4 (quality, not correctness): kernel always used the flat rank, never each
layer's real adaptive rank

Given Bug 2's fix makes addressing correct, the kernel would still only ever
reconstruct using 32 of a boosted layer's 48 real stored components (the loop bound
`rank` was still the flat caller value). Fixed by having `dkv_attention.py` compute
`get_layer_rank(captured_layer_idx, ...)` (the same call compression itself used) and
pass that instead of `kv_manager.rank` — and bumped `q_proj_shared`/`red_w_proj`/
`local_w_proj`/`red_proj_temp`/`q_proj_cached` from 32→48 capacity to allow it
(re-shrank the block-count cache 64→32 for threadgroup-memory headroom; total budget
is now 32650/32768 bytes — tight but confirmed to compile and run). Added a
`rank > 48` throw guard mirroring the D>256 one.

### BUILD-SYSTEM GOTCHA — read this before touching `dkv_decode.metal` again

`setup.py`'s `build_ext --inplace` does **NOT** treat `dkv_metallib.hpp` (the
generated header embedding the compiled metallib bytes) as a dependency of
`metal_runtime.mm` (which `#include`s it). Editing *only* `dkv_decode.metal` and
rebuilding regenerates `dkv_metallib.hpp` correctly (visible in the build log: "
Embedding NNNNN bytes...") but **distutils sees `metal_runtime.mm`'s own mtime is
unchanged and skips recompiling it** — so the stale, pre-edit metallib bytes stay
baked into the `.so` silently, with a build log that looks completely successful.
This cost real time this session (spent a full test cycle believing the D=256 fix
had failed, before noticing the `.so`'s mtime hadn't moved). **Always
`rm -rf build && rm -f dkv_core.*.so` before rebuilding whenever `dkv_decode.metal`
changed** — don't trust an incremental `build_ext --inplace` for shader-only edits.

### Verified end-to-end result

Test: 8217-token prompt, needle `ZEBRA-4471-QUARTZ` (12 tokens: `Z EB RA - 4 4 7 1 -
QU ART Z`) near the start, Path A forced (`DKV_USE_ATTENTION_INTERFACE=0`), real
compression engaged (`DKV_ENGAGE_THRESHOLD=4096`).

- **Before any of this session's fixes**: NaN in decode output from the first call;
  end-to-end generation collapsed into repeating token 0 (`!`) after at most a
  fragment of the needle.
- **After Bug 1 fix alone**: NaN eliminated; output `'...Z'` (1 correct token, then
  stops) — confirms Bug 1 was the dominant garbage-cause.
- **After Bug 1+2**: output `'...ZEBRA'` (3 correct tokens: Z, EB, RA) then a clean
  stop via `<|im_end|>` — no more collapse into repeated garbage.
- **After Bug 1+2+3+4 (final state)**: same `'...ZEBRA'` result; bugs 3-4 didn't
  change this specific test's outcome (bug 3's trigger condition likely never
  occurred here; bug 4 improves reconstruction fidelity for boosted-rank layers but
  didn't flip this particular token) but are real, verified-correct fixes worth
  keeping (bug 3 is a genuine latent bug for other sessions/configs; bug 4 corrects
  a real quality gap for CUDA hardware and any Mac test with K>32 routed blocks).

### Remaining gap, precisely characterized and QUANTIFIED — confirmed a genuine
close call, not a further structural bug

Per-token logit trace (`DKV_DEBUG_PER_TOKEN`-style instrumentation, since removed)
at the exact failure point:

```
n=8218 'Z'  : top1=24.938  runner-up=21.297  (margin 3.64)
n=8219 'EB' : top1=17.156  runner-up=14.727  (margin 2.43)
n=8220 'RA' : top1=19.344  runner-up=16.562  (margin 2.78)
n=8221 EOS  : top1=21.078 (EOS)  '-'(correct)=19.547  (margin 1.53)  <- WRONG
```

The margin at the failing step (1.53 logits, EOS over the correct `-` continuation)
is real but notably smaller than the ~2.4-3.6 logit margins at the three correctly-
retrieved steps — the model's weakest, and only wrong, decision in the sequence.

**Follow-up (same day)**: computed exact sampling probabilities at this step for
several temperatures directly from the logits (no need to re-run generation at each
T — `softmax(logits/T)` off a single captured logit vector), correcting an initial
wrong suggestion to test `temperature=0.2-0.3` (T<1 *sharpens* toward the already-
highest logit — multiplies the gap, e.g. ×5 at T=0.2 — making EOS *more* dominant,
not less; T>1 is what flattens the distribution toward a genuine coin flip):

```
                  EOS (wrong)        '-' (correct, id=12)
T=1.0 (raw):      P=0.784            P=0.138
T=1.2:            P=0.650            P=0.153
T=1.5:            P=0.388            P=0.122
T=2.0:            P=0.078            P=0.033
```

At the three correctly-retrieved steps, the runner-up's T=1.0 probability was only
2.5-4.7%. At the failing step, the CORRECT token's T=1.0 probability is 13.8% —
roughly 3x higher than a normal "confident and right" runner-up, i.e. genuinely
non-trivial, not a fluke near-zero tail probability. This is a real, moderate-
confidence near-miss: not "the model has no idea" (~1/vocab≈0.0004%), not "EOS is
overwhelming" (>99%), and consistent with — now with actual numbers behind it — the
pattern already documented in `project_random_needle_code_fidelity_gap.md` (memory)
for a structurally identical `PREFIX-NNNN-SUFFIX` code on a different model: "Last
1/8 ... = greedy decode artifact (digits retrieved right, extra token), not KV."

**Conclusion: this is very likely NOT a further structural bug** — it's a smaller,
inherent precision gap in the Project-Then-Attend approximation (needle block IS
confirmed force-exact/lossless; dense with no compression at all retrieves the full
code with no ambiguity) surfacing as reduced-but-nonzero confidence at one specific
token transition, not a hard failure. **If a future session wants to close this
last bit anyway**: the isolated Metal-kernel-vs-NumPy unit test from §9c-6 (still
not done) is the only remaining technique that could reveal a further microscopic
numerical discrepancy that black-box logit tracing can't — but given the quantified
"real but moderate" margin above, this is now optional polish, not a required fix.

### Everything else in §9c stands

The full ruled-out list, the two parallel-path discovery, the `_is_mps_decode`
control-flow finding, and the 8 other real bugs fixed in that section are all still
accurate and still relevant — nothing here contradicts them, it just closes the one
item §9c left open ("root cause not fully isolated").

## 9e. Isolated Metal kernel unit test built (§9c-6's recommendation) — found and
fixed ONE MORE real bug: unguarded fact-override reads

Built `ACTIVE_RUNTIME/tests/test_decode_attention_metal_isolated.py`: calls
`dkv_core.decode_attention_metal(...)` directly with hand-constructed tensors,
compared against a from-scratch NumPy reference transliterated line-by-line from
`dkv_decode.metal`. Reference cross-checked against fully independent hand
arithmetic on a toy case before trusting it against the kernel (see method note
below) — necessary because early runs disagreed with the kernel and the reference
itself turned out to need a fix first (see "false alarm" below).

**Real bug found and fixed**: fact-override reads (`fact_pos[slot_id*3+tid]` for
tid 0,1,2) were **unconditionally executed** with no gate, unlike residuals (safely
gated by `max_residual==0`, so their load loop never runs) and dense (gated by
`L_dense==0`). When the host has no fact data, it substitutes a 1-element dummy
buffer (`metal_runtime.mm`'s existing `has_fact` local) — reading index 1 or 2 from
that 1-element buffer is out-of-bounds, and Metal doesn't bounds-check. A garbage
value read from adjacent memory could spuriously equal a real token position `t`,
triggering a bogus "fact override" that replaces that token's score AND reads
further out-of-bounds from the equally-1-element dummy `fact_val_K`/`fact_val_V`
buffers for its value. Confirmed with an ultra-minimal repro (1 slot, 1 token, no
RoPE, empty fact tensors): kernel output was wildly wrong (`[-2.77, -5.54, ...]` vs
hand-computed `[3.50, 7.00, ...]`) before the fix, exact match after.

**Fix**: added `has_fact` to `AttentionParams` (mirroring the existing
`has_dense_rope` pattern — `metal_runtime.mm` already computed a local `has_fact`
bool for choosing real-vs-dummy buffers but never wired it into the kernel-visible
struct). Kernel's fact_pos load site changed to
`has_fact ? fact_pos[slot_id*3+tid] : (int16_t)-1` — short-circuits the
out-of-bounds read entirely and sets the "no override" sentinel (-1, which no real
token position can ever equal) instead. One load-site fix protects all 3 usage
sites downstream (they only ever check `fact_pos_shared[fi] == t`).

**How likely is this to matter in practice?** Unclear, deliberately not chased
further this session — `pool.fact_anchor_positions`/etc. being `None` (which is
what triggers `dkv_attention.py` to pass the empty tensors this bug needed) would
require either genuinely zero compressed blocks yet (in which case `K_active=0`
anyway and the per-slot loop that contains this bug never runs) or some other,
not-yet-identified path where fact-tracking data ends up unset while blocks are
compressed. Fixed regardless since it's a genuine, confirmed correctness violation
with no legitimate reason to leave in.

**False alarm worth recording** (cost real debugging time): the first test run
showed several cases failing by 100s of percent — before finding the real bug
above, first had to rule out the reference implementation itself. Method: reduced
to the simplest possible case (1 slot, 1 token, D=4, no RoPE) and hand-computed the
expected output with completely fresh, independent arithmetic (not reusing any
helper from the reference file) — this matched the reference implementation
exactly, confirming the reference was sound, and by elimination pointed at the
kernel. Also separately fixed a test-methodology issue (comparing against
full-float32-precision inputs rather than the fp16-rounded values the kernel
actually receives inflated the apparent diff on several cases) and a tolerance-
check bug (flat absolute cap instead of `atol + rtol*|ref|`, which flagged a
large-magnitude, correctly-tiny-relative-error case as a false failure). Neither
of those was a kernel bug — worth remembering before trusting any future
"kernel disagrees with reference" result at face value.

**Final result: 11/11 cases pass**, including regression checks for every bug this
session fixed (D=256 buffers are load-bearing for the realistic-scale case; rank<
pool_rank addressing; `has_dense_rope`; `has_fact`) plus GQA, partial RoPE,
residual/fact overrides, and a case matching Qwen3.5-2B's actual mid-layer config
(D=256, rotary_dim=64, rank=32, pool_rank=48) at realistic scale (40 pool slots, 20
routed, block capacity 257).

**Environment gotcha found while building this**: a stale editable pip install
(`__editable__.dkv_core-1.1.0.pth` in the venv's site-packages) resolves a bare
`import dkv_core` to an unrelated, outdated build at the repo root
(`/dkv_core.cpython-314-darwin.so`, dated well before this session), NOT the real
development build at `native_core/dkv_core/`. Any future standalone script
importing `dkv_core` without first inserting `native_core/dkv_core` onto
`sys.path` will silently run stale kernel code with no error — exactly what
happened on the first run of this new test file. `dkv_attention.py` avoids this
via its own explicit `sys.path.insert` at import time; anything that imports
`dkv_core` independently needs to do the same.

### CUDA/Triton GPU kernel path — audited, one class of bug found, one fixed, one
UNTESTABLE from this Mac

User asked directly whether Bugs 1-2 (Metal) also exist in the real Triton/CUDA
kernels (`native_core/sparse_decode/triton_fused_decode.py`, the `@triton.jit`
kernels that run on real NVIDIA hardware, `HAS_TRITON=True`). Audited by reading —
this Mac has no CUDA/Triton to run it on, so this is code-reading + one forced
smoke test, not full verification like the Metal fixes got.

- **Bug 1 analog (D-size hardcoding) — very likely NOT present.** Triton's
  `HEAD_DIM`/`D`/`RANK`/`R` kernel parameters are `tl.constexpr`, not fixed-size C
  arrays — Triton auto-specializes/recompiles a distinct kernel binary per unique
  constexpr value combination (confirmed by `@triton.autotune(key=["R", "D"])`'s own
  comment: "Triton pre-compiles one kernel per (R, S_MAX, D) combination"). There is
  no way for a 256-wide head_dim to silently alias a 128-sized buffer the way Metal's
  hand-declared arrays did. Not empirically tested (no hardware), but structurally a
  different, safer mechanism by design.
- **Bug 2 analog (rank/pool_rank stride corruption) — NOT present, also by design.**
  `_gather_routed_blocks_for_kernel` gathers each routed block's real pool tensors via
  `pool_for_kernel.V_K[indices]` (tensor-native fancy indexing, immune to stride
  confusion — same reason the ATen path was safe once ported to `.index_select`), and
  the kernel launch passes each gathered tensor's OWN `.stride(N)` values as explicit
  runtime arguments (`stride_vk_n, stride_vk_r, ...`) rather than recomputing offsets
  from `pool_idx * R * ...` the way Metal did. Verified this pattern in both
  `_fused_sparse_decode_kernel` and `_fused_decode_combined_kernel`.
- **Bug 4 analog (flat rank instead of per-layer adaptive rank) — WAS present, FIXED.**
  All 4 call sites of `native_triton_sparse_attn_decode`/`_combined` in
  `dkv_attention.py` (lines ~2444/2483/2508/2622 pre-fix) passed `R=kv_manager.rank`
  (the flat base value) instead of each layer's real `get_layer_rank(...)` value —
  same class of completeness gap as the Metal fix, not corruption (per Bug 2 analog
  above, the addressing itself was always safe). Fixed identically: compute
  `_layer_active_rank` via `get_layer_rank(captured_layer_idx, ...)` at each site,
  pass that instead. This touches the "has_dense/has_comp" 4-way dispatch branch —
  dead code on this Mac (`_is_mps_decode` is always `True` here) but the ACTUAL live
  decode path on real CUDA hardware.
- **Separate bug found and fixed while smoke-testing the above**: forcing this branch
  open on Mac (`DKV_MPS_APPROXIMATE_ATTN=0`, which routes to the `HAS_TRITON=False`
  Python fallback `_pytorch_vectorized_sparse_attn_decode`) crashed with `RuntimeError:
  Expected arguments of same type but got Float and Half` inside
  `_attend_and_reconstruct_v_compiled`'s dense-window matmul (`triton_fused_decode.py`
  line ~649) — missing a `.float()` cast that the *identical* pattern two lines above
  (the compressed-block matmul) already had. Fixed by adding the same cast. This is a
  genuinely separate, pre-existing bug, not something introduced by the rank fix —
  likely invisible until now because this whole branch is essentially never exercised
  (dead on Mac, superseded by the real Triton kernel on actual CUDA hardware whenever
  `HAS_TRITON=True`).
- **What this does NOT verify**: the actual Triton kernel arithmetic on real GPU
  hardware. The smoke test only exercises the Python fallback (same code family, same
  `R` value now flowing through, but not the same binary/execution path a real GPU
  would use). If a future session has access to a CUDA machine, re-running the same
  needle-recall test there (`DKV_USE_ATTENTION_INTERFACE=0`, real GPU, `HAS_TRITON`
  true) would be the first real validation of any of this — nothing in this session
  did that.

## 9f. Residual/fact RoPE parity ported to Metal (REAL fix), routing DEFINITIVELY
ruled out, and the remaining gap traced to the Project-Then-Attend approximation
itself — READ THIS BEFORE ATTEMPTING ANY FURTHER "needle recall" WORK

User asked to "fix this error properly, dig deep". This section is that dig. It
contains one real shipped fix, one important correction to a previously-recorded
(wrong) conclusion, and a precise mathematical account of why the symptom
persists — plus a concrete proposal that is deliberately NOT shipped.

### 1. REAL BUG FOUND AND FIXED: 3-way residual/fact RoPE parity split

`residual_K` and `fact_K` each store the EXACT (or exact-minus-reconstruction)
key for ONE specific token, at within-block offset `p` — i.e. absolute position
`anchor + p`. Three implementations disagreed on where to rotate them:

| path | residual-K rotation | status |
|---|---|---|
| real CUDA+Triton (`_gather_routed_blocks_for_kernel`) | TRUE position `anchor+p` | correct (`DKV_RESIDUAL_EXACT_ROPE`, default on) |
| Mac/CPU fallback (`_pytorch_vectorized_sparse_attn_decode`) | anchor position | WRONG |
| **Metal kernel** (`dkv_decode.metal`, the live path here) | anchor position | WRONG |

The CUDA fix is recorded in-code as A100-validated: random-code recall 75%->88%,
"recovered a digit-drop code". MLX does the equivalent (appends exact rows as
real tokens at their true positions). Metal and the PyTorch fallback never got
the port. Corroborating clue that had been under-weighted: the PyTorch fallback
run reached `ZEBRA-99` while Metal stopped at `ZEBRA` — two implementations of
the same algorithm giving different answers is itself proof one has a bug.

**Fixed** by porting exact-position rotation to the Metal kernel (new
`has_exact_res_rope` + `rope_full_rows` params, buffers 26/27/28 carrying the
model's raw full-sequence cos/sin tables plus per-slot anchor positions) and to
the PyTorch fallback. All three now agree. The tables are passed as a POINTER to
the already-cached `session_dict["rope_cos"]`, not a per-token gather, so decode
throughput is unaffected.

Implementation notes worth keeping:
- `dkv_rope_rotate_dim` gained an explicit `rope_stride` param. It is NOT always
  `D`: the anchor/dense tables are host-padded to head_dim (stride `D`), while
  the full-sequence table is the model's raw `[max_pos, rotary_dim]` (stride
  `rotary_dim`). Every call site now states it explicitly.
- Alignment invariant this depends on: `anchor_pos[k]` must correspond to
  `slot_indices[k]`. Verified — `block_indices` and `anchor_indices` are filtered
  by the same mask in the SRL reroute path (`block_idx_in_layer` / `has_match`).
- **pybind trap**: `torch::Tensor()` as a `py::arg` default SEGFAULTS AT IMPORT
  (evaluated during module registration, before the dispatcher is ready), and
  `torch::empty({0})` does too. Use `c10::optional<torch::Tensor>` +
  `py::arg(...) = py::none()`. Cost ~2 debugging cycles; don't repeat.

**Verified**: 14/14 isolated-kernel unit tests pass, including 3 NEW cases that
exercise the exact-position path. Those new cases were explicitly checked to be
DISCRIMINATING — computing the reference with anchor-position rotation instead
differs by ~6x the test tolerance (0.157 vs 0.028), so they would genuinely fail
against the old behavior rather than passing vacuously. The 11 pre-existing
cases still pass (backward-compat fallback path unchanged).

**But it did NOT fix the symptom.** Model-level output is unchanged: `...ZEBRA`
then EOS. This is a genuine correctness/parity fix worth keeping on its own
merits, but it is not the cause of this symptom.

### 2. CORRECTION: §9c's "routing is not it" was drawn from an INVALID experiment

§9c-5 recorded `DKV_TOPK_BLOCKS=0` as "retested post-bypass-fix, identical output
— routing is not it". That test ran BEFORE the Metal NaN bugs (§9d) were fixed.
With NaN output, every experimental variant produces identical garbage, so the
experiment could not have distinguished anything. Same trap as the
`_DKV_HAS_METAL_ATTN=False` isolation test §9c itself flagged as invalid — it
recurred, and the conclusion was trusted anyway.

**Re-run properly against the now-correct kernel**, with one-shot diagnostics
(since removed). Results:

```
exact_res_rope ACTIVE=True  cos_full=(8218,64)  anchor_pos_n=30  K_slots=30
routed anchors: [0, 257, 512, 769, ... 7425]   (30 blocks)
needle block (anchor 0) routed? True
anchor-0 residual positions: [0,1,2,...,39,...]   (needle sits ~20-35)
ANSWER TAIL: '...ZEBRA'   FOUND: False
```

So, with valid evidence this time: the fix genuinely activates at runtime; the
needle's block IS routed (even with attend-all); residuals DO cover the needle's
positions. Routing, storage coverage, and kernel arithmetic are all ruled out.
The old conclusion happened to be right, but it had not actually been tested.

### 3. WHERE THE REMAINING ERROR ACTUALLY COMES FROM (design, not defect)

For a token at within-block offset `t`, the kernel computes:

```
base       = q . RoPE(anchor_K + recon_delta[t],  anchor)      <- rotated at ANCHOR
correction = q . RoPE(residual_delta[t],          anchor + t)  <- rotated at TRUE pos
score      = base + correction
```

Mathematically correct would be:

```
score = q . RoPE(anchor_K + recon_delta[t] + residual_delta[t], anchor + t)
```

The VALUES reconstruct exactly (`anchor + recon + residual == exact`), but the
BASE term is rotated at the wrong position. Residual error:
`RoPE(base, anchor) - RoPE(base, anchor+t)` — a phase error that grows with
distance from the anchor (t can reach 255).

This is the Project-Then-Attend approximation itself, not a bug: rotating the
whole block at a single anchor is exactly what allows precomputing
`q_proj = q . RoPE(VK, anchor)` once per block and then spending only `rank` ops
per token instead of reconstructing full D-dim keys. Exact per-token positional
scoring is fundamentally incompatible with that speedup. It also explains why
the §9f-1 fix was necessary but insufficient: it corrects the CORRECTION term
while the BASE it is added to stays anchor-rotated — the hybrid the CUDA-side
comment already hedges about.

### 4a. CORRECTION (after reading MLX): MLX does NOT fix the PTA phase error

Re-read of `compute_decode_attention_static` shows MLX rotates the compressed
block's anchor + SVD terms at the ANCHOR position too — exactly the same
approximation. MLX does not solve the phase error for ordinary compressed
tokens; it makes the tokens that MATTER bypass PTA entirely, by masking their
lossy twin to -inf and attending the exact row in the DENSE pool at its true
position.

That is functionally identical to `DKV_RESIDUAL_EXACT_KEYS` (§9g-2): both give
the position a score of `q . RoPE(exact_K, true_pos)` and discard the low-rank
estimate. So the MLX-parity fix is the exact-keys mode, NOT the anchor-retiming
idea below.

IMPORTANT: the earlier "exact-keys alone changed nothing" measurement was taken
while the dense window was still broken (attending padding + OOB cos/sin reads,
§9g-3) and before the workspace layout fix (§9g-5). It needs re-running in the
corrected configuration before drawing any conclusion from it.

### 4b. PROPOSED CHEAP PARTIAL FIX — NOT SHIPPED, superseded by 4a as the
MLX-faithful route

For residual positions only (<=64 per block, where the kernel ALREADY pays a
full D-dim dot), recompute the anchor term at the true position:

```
score_anc_t = q . RoPE(anchor_K, anchor + t)     // instead of q . RoPE(anchor_K, anchor)
```

Cost: ONE extra D-dim dot per residual token (not the D*rank of full
reconstruction). It captures the dominant share of the phase error because
`anchor_K` is the bulk of each key; only the (small) `recon_delta` term stays
mis-rotated.

Deliberately NOT implemented: it changes a core accuracy/throughput tradeoff in
a kernel shared by every model and path, and that is an owner decision, not
something to slip in at the end of a debugging session. If pursued, gate it
behind a flag, measure decode tps, and re-run the 14-case kernel suite.

### 5. One experiment that came back UNINTERPRETABLE (recorded so it isn't re-cited)

Moved the needle to the END of the prompt (into the uncompressed dense window)
intending a clean compressed-vs-dense discriminator. DKV returned `ABC123` — a
hallucinated placeholder, WORSE than the compressed case. But that test changed
TWO variables at once (storage location AND prompt structure/phrasing), and the
matching dense (no-DKV) control repeatedly crashed on this 8GB Mac (SIGABRT/
SIGSEGV during weight load). **Without the control this proves nothing** — the
model may simply ignore a needle buried at the end of 600 filler sentences.
Do not cite `ABC123` as evidence of anything. If re-attempting: change ONE
variable, and get the dense control first.

## 9g. MLX used as the reference (user request) — led to a SEVERE dense-window bug
and an MLX-parity residual mode. READ THIS BEFORE TOUCHING THE DENSE PATH.

User: "mlx works fine can you refer to it and fix here". Doing that produced the
single most severe defect found in this whole investigation, plus a precise
account of the architectural difference between MLX and the CUDA/Metal design.

### 1. What MLX actually does differently (the architectural answer)

MLX never forms the "anchor-rotated base + correction" hybrid the Metal/CUDA
kernels use for residuals. In `compute_decode_attention_static`:

```python
# lossy SVD twin of a residual position is REMOVED from the compressed pool
delta_s = mx.where(res_mask, -inf, delta_s)
# ...and the EXACT row is attended as a REAL dense token at its true position
dense_k_for_attn = mx.concatenate([res_k_all, dense_k], axis=1)
```

So an exact/residual token is scored as an ordinary token with correct RoPE, and
its approximate twin contributes nothing. No phase error by construction.

Crucially, DKV **already has this exact mechanism** — it is the `fact` override
path (`fact_anchors_K` stores `k_orig[pos]`, and the kernel REPLACES the score:
`final_t_score = exact_k_sum * scale`). It is just capped at 3 slots/block, while
the residual path (up to max_residual=64 slots, the one that actually covers a
needle) uses the lossy additive form: `res_K_vals = (delta_K - recon_K)` with
`final_t_score += ...`.

### 2. MLX-parity residual mode added (DKV_RESIDUAL_EXACT_KEYS, default OFF)

Implemented the minimal exact equivalent: store residual K as the full
anchor-relative delta `(exact_K - anchor_K)` instead of `(delta_K - recon_K)`,
and have the kernel REPLACE the token's score with
`q . RoPE(anchor_K + res_val_K, anchor+t)`. Since `anchor + (exact - anchor) ==
exact`, this is mathematically exact, costs two D-dim dots (no D*rank
reconstruction), and changes no tensor shapes or memory.

The V side needs NO change and was verified rather than assumed: `res_val_V` is
`(delta_V - recon_V)` and the block already contributes `w*(anchor_V + recon_V)`,
summing to `w*exact_V`; V carries no RoPE so it has no phase error.

**MEASURED: enabling it REGRESSES output** (`...ZEBRA-2024` with the layout fix
alone drops back to `...ZEBRA` with exact-keys on). Root cause diagnosed — the
implementation is INCOMPLETE: there are (at least) THREE places that construct
`residual_K_values`, and only ONE was converted:

  1. `compression/lowrank.py:~477`  compress_lowrank        -- CONVERTED (flag-aware)
  2. `compression/lowrank.py:~1274` GPU/batched path        -- NOT converted, still `(delta_K - recon_K)`
  3. `kv_runtime_manager.py:~3530`  another force-exact path -- NOT converted

With the flag on, some blocks store `(exact_K - anchor_K)` while others still
store `(delta_K - recon_K)`, but the kernel REPLACES the score for all of them
assuming the former. Blocks from the unconverted paths therefore get their score
replaced by a wrong value — strictly worse than the additive correction it
displaced. **All three WERE then converted** (each site now reads the same env var, with a
comment stating the forms must match) — and an audit confirmed there is no
fourth producer: every other `delta_K - recon_K` in `native_core` is an
error-NORM computation, not a stored value.

**Re-tested with the complete rollout: STILL REGRESSES** (`...ZEBRA` vs
`...ZEBRA-2024` for layout-fix-alone). So the mixed-semantics hypothesis was
WRONG — exact-keys genuinely hurts here, it was not a rollout artifact. The
storage math is verifiable (`res = delta_K * token_norms = exact_K - anchor_K`,
so `anchor + res == exact`), and the V side legitimately stays in correction
form, so the defect is not obviously in the arithmetic. Leaving it OPT-IN and
UNRECOMMENDED. If revisited: instrument whether REPLACING a token's score
distorts the softmax against the block's own anchor token (whose score is still
`q . RoPE(anchor, anchor_pos)`), since replace-vs-add changes the relative
weighting that the V-side `w_total_anc` accumulation assumes.

Default OFF because it changes the stored residual SEMANTICS — every decode path
consuming `residual_K_values` must agree, and the Triton/CUDA kernel still
expects the correction form. Both the compressor (`compression/lowrank.py`) and
the Metal host read the same `DKV_RESIDUAL_EXACT_KEYS` var so they cannot
disagree. **Measured: enabling it alone did NOT change the needle result**
(still `...ZEBRA`). Kept as a correct, reversible parity option; do not enable
on CUDA until the Triton kernel is ported.

### 3. SEVERE BUG FOUND: the dense window attended PADDING as real tokens

The decisive experiment was moving the needle to the END of the prompt, into the
UNCOMPRESSED dense window, with a proper dense control this time:

```
needle at END, dense (no DKV) : 'ZEBRA-4471-QUARTZ'   FOUND=True
needle at END, DKV            : 'ABC123'              FOUND=False   <-- hallucination
```

DKV loses a needle sitting in the window that is supposed to be EXACT. Cause,
confirmed against the real captured kernel inputs:

```
dense_k   = (1, 2, 1419, 256)   <- FIXED-SIZE workspace, padded to max_dense_len
cos_dense = (1, 1,  538, 256)   <- only 538 rows are REAL tokens
```

`dense_k_assembled` is passed to the kernel **unsliced**, and the kernel did
`L_dense = dense_K.size(-2)` — i.e. it took the workspace's ROW STRIDE as the
TOKEN COUNT. It then looped `t` over `[0, min(1419, 768)) = [0, 768)`, so it:

- attended **230 padding rows as if they were real tokens**, giving stale/garbage
  KV real softmax weight, and
- read `cos_dense[t]`/`sin_dense[t]` for `t` up to 767 when those tables have only
  538 rows — an **out-of-bounds read** on every decode step.

This corrupts the dense window, which holds the most recent context INCLUDING the
question tokens — so it degrades every answer, not just needle recall.

**Parity note**: the Triton kernel already carries BOTH `L_dense` (padded stride)
and `L_dense_valid` (real token count) as separate params, with an explicit
comment that "positions >= L_dense_valid are padding". Metal only ever had one.
Another case of a fix existing on one path and never being ported.

**Fixed**: added `L_dense_valid` to Metal's `AttentionParams`, derived host-side
from `cos_dense`'s row count (the host sizes that to exactly the valid count, so
it is authoritative and needs no ABI change). Loop bounds now use the valid
count; buffer addressing still uses `L_dense` (the true memory layout). Also
added a one-shot warning when the valid count exceeds `dense_w_shared`'s 768
capacity, since beyond that the kernel silently drops the NEWEST dense tokens —
previously a silent context loss.

### 4. IMPORTANT CORRECTION — the strict dense bound REGRESSED output; it is now
gated OFF by default, and it exposed a DEEPER layout bug

Enabling the strict `L_dense_valid` loop bound made things WORSE, not better:

```
needle-at-END, DKV, before strict bound : 'ABC123'                (wrong, coherent)
needle-at-END, DKV, WITH strict bound   : 'ABC' + repeated tok 0  (garbage collapse)
needle-at-START, DKV, WITH strict bound : '...ZEBRE'  (vs '...ZEBRA' before)
```

Cause, found by reading `assemble_dense_window_kv`'s allocator rather than its
docstring: write offsets are **cached per (layer_idx, anchor_idx) ACROSS decode
steps** (`dense_offsets`), and on the non-`new_alloc` path it does
`curr_idx = min(offset + 1 + active, max_dense_len)` from the CACHED offset. When
a block is trimmed away, the survivors keep their old offsets — so live data can
sit at a HIGH offset with a gap at the front. `L_dense` is therefore a token
COUNT that does NOT describe the workspace's occupied EXTENT, even though the
docstring says "positions >= L_dense contain stale/zero data".

So the historical `min(L_dense, 768)` bound was over-attending, and that
over-attend is what accidentally covered live tokens parked at high offsets.
Bounding the loop by the count drops them → garbage collapse.

**Current state (what ships)**:
- Loop bound: unchanged historical behavior by default. Strict bound is opt-in
  via `DKV_DENSE_VALID_LEN=1`.
- The cos/sin row index IS now clamped unconditionally, since reading those
  tables past their end is never correct under any bound. Pure memory-safety;
  in-range rows are bit-identical.
- `12_padded_dense_workspace` runs under the strict flag (the kernel test builds
  a densely-packed workspace, which is the layout strict assumes).

**Gating verified**: with the flag off, needle-at-END returns to coherent output
(`ABC12345`, no collapse) and needle-at-START returns to the `...ZEBRA` baseline
— i.e. the regression is fully reverted and the session leaves the default path
no worse than it found it.

**Before enabling `DKV_DENSE_VALID_LEN=1`**, fix the layout itself — either pack
the workspace densely every step, return the true occupied extent instead of a
count, or pass a real per-row validity mask. Until then the strict bound is
knowingly unsafe, and the padding-attend it was meant to fix remains present.

**Regression test added**: `12_padded_dense_workspace` builds a dense workspace
with padding rows set to 25.0 (vs ~0.3 for real values) so that attending any of
them overshoots tolerance by orders of magnitude. Note that ALL 14 pre-existing
kernel cases passed an UNPADDED dense window — which is exactly why this bug
survived a supposedly thorough kernel test suite. When adding kernel tests,
reproduce the host's real buffer shapes, not just its logical ones.

## 9h. LAYER-BY-LAYER DIFFERENTIAL vs DENSE GROUND TRUTH — the measurement that
should have been done first. DKV's attention is UNCORRELATED at the first
compressed layer. START HERE.

User asked to go layer by layer and find where DKV diverges. Method: hook every
`self_attn` module and capture its OUTPUT at the FIRST decode step, for DKV and
for plain dense. At that step both runs have seen an identical prompt AND an
identical input token, so any per-layer difference is purely DKV's KV
reconstruction — no sampling divergence. (Compared against dense in the SAME
framework rather than against MLX directly: MLX runs 4-bit weights vs fp16 here,
so an MLX-vs-PyTorch diff would be dominated by quantization noise. Dense is the
exact reference DKV is supposed to reproduce.)

Qwen3.5-2B has only 6 full_attention layers (3, 7, 11, 15, 19, 23) — the only
ones DKV compresses. Result:

```
 layer    cos_sim    rel_err   dkv_norm  dense_norm
     3   0.047131    1.73989      2.051       1.393   <- FIRST DKV layer, already broken
     7  -0.090622    1.60944      2.811       2.395   <- anti-correlated
    11   0.041744    2.13648      4.811       2.492
    15   0.318857    1.00977      4.288       6.427
    19   0.641366    0.77946      5.679       7.291
    23   0.721478    0.74271     11.325      11.438   <- partially recovers
```

**This is not a precision issue.** cos ~= 0.05 at layer 3 means DKV's attention
output is essentially ORTHOGONAL to ground truth at the very first layer it
touches. Everything this investigation chased before now — digit fidelity,
residual RoPE phase, PTA anchor rotation, top-K routing — are second-order
effects on top of an output that is already uncorrelated. The later layers
improving (0.05 -> 0.72) is the residual stream re-anchoring them, which is why
the model still emits superficially coherent text and why the failure looked
like "subtly wrong digits" from the outside.

**Reconciles with the kernel being correct.** The isolated kernel suite passes
15/15 against an independent NumPy reference, so the arithmetic is right — which
means the INPUTS handed to it are wrong. The defect is upstream of the shader:
pool contents, the dense window, routing, or the LSE merge.

**BISECTED — compression is the culprit, everything else is provably clean.**
Re-captured with `DKV_ENGAGE_THRESHOLD=999999` (compression never engages, pure
dense path through the wrapper):

```
 layer    cos_sim    rel_err        compression ON (for contrast)
     3   0.999999    0.00100        0.047131
     7   0.999999    0.00143       -0.090622
    11   0.999998    0.00175        0.041744
    15   0.999999    0.00156        0.318857
    19   1.000000    0.00106        0.641366
    23   0.999999    0.00150        0.721478
```

rel_err ~0.001 is pure fp16 noise. So the wrapper's ENTIRE non-compressed
machinery is correct: gated attention, partial RoPE, QK-norm, the dense window,
the LSE merge, the bypass branch, the hybrid layer_types handling. All of it
reproduces dense to 6 nines. **The defect is exclusively in the compressed-KV
path.**

This retro-justifies keeping the §9a-§9e fixes (they were real bugs in that
machinery) while explaining why none of them moved the needle: they were not
where the dominant error lives.

**Next probe (running when written)**: `scratchpad/recon_fidelity.py` — a
self-contained check, inside ONE DKV run, of whether the pool reconstructs the K
it was ingested. It spies on `ingest_streaming` to record ground-truth unrotated
K during prefill, then rebuilds each compressed block from the pool with the
kernel's own math (`anchors_K + (U[t] @ V_K) * U_scale * block_scale`) and
compares. Splits the remaining space cleanly:
  - reconstruction BAD  -> compression/storage is broken (SVD, quantization,
    scales, or the pool write path); nothing decode-side can fix it.
  - reconstruction GOOD -> storage is fine and the defect is in routing /
    kernel-input assembly / the sparse-vs-dense LSE merge.

**RESULT: RECONSTRUCTION IS BROKEN.** Rebuilding each compressed block's K from
the pool with the kernel's own math and comparing to the K actually ingested:

```
 blk  anchor  slot   seq   cos(recon,true)   rel_err
   0       0     0   256          0.400704    1.0052
   1     257     1   254          0.378804    1.0270
   2     512    12   256          0.382424    1.0262
   3     769    13   254          0.375578    1.0324
```

`rel_err ~ 1.0` means the error is the same magnitude as the signal — the
reconstruction carries essentially no usable information. And the cos is
UNIFORM at ~0.38-0.40 across every block, which is the signature of an
ANCHOR-ONLY reconstruction: if the delta term contributed nothing, K_recon
collapses to the anchor broadcast over the block, and cos(anchor, tokens) for
semantically similar tokens lands right about there.

So the low-rank delta (`U @ V_K`) is contributing garbage or nothing. That is
upstream of every decode-side thing this investigation has touched: it is the
SVD/compression/pool-write path itself.

**Solver running**: `scratchpad/recon_solve.py` reports, per block, the
anchor-only baseline plus `cos(candidate_delta, true_delta)` for a matrix of
candidate formulas — `[R,kv,D]` vs `[kv,R,D]` V_K layout, and the four
combinations of applying `U_scale` / `block_scale` / both / neither. Whichever
candidate scores near 1.0 IS the correct formula, which names the bug directly
(a layout transposition, a doubled scale, or a missing one). If NONE scores well,
the stored factors themselves are wrong (bad SVD, wrong slot, or a pool-write
that never landed) rather than merely mis-applied at read time.

**Reusable tooling**: `scratchpad/capture_layers.py <dense|dkv> <out.pt>` and
`scratchpad/compare_layers.py`. Cheap, decisive, and it localizes a defect to a
layer in two runs. Should have been step one of this whole investigation rather
than step N — every earlier hypothesis was argued from black-box token output
when a direct per-layer differential against ground truth was available.

## 9i. TWO REAL COMPRESSION BUGS FOUND AND FIXED (sync path) — plus the four
probe defects that had to be corrected before any measurement could be trusted

### The instrument had to be debugged before the code could be

Four successive versions of the reconstruction probe produced confident-looking
numbers that meant nothing. Recording them because each is an easy trap:

1. **Omitted residuals** — measured SVD-only and concluded "reconstruction is
   broken". A rank-24 SVD capturing ~60% of the delta is the DESIGN POINT; the
   residuals are the exactness mechanism. Grading without them is meaningless.
2. **Mathematically identical "alternative" layouts** — the `[kv,R,D]` candidate
   was `VK.permute(1,0,2)` with a matching einsum re-index, which cancels out.
   All six candidates scored identically; that uniformity was the tell.
   (Also: cosine is scale-invariant, so the `U_scale`/`block_scale` variants
   could never have been discriminated by cos either.)
3. **Stale rank columns** — used `pool_rank` (48) instead of the LAYER's adaptive
   rank (24). The pool writes only `[:seq,:rank]`; the rest holds whatever the
   slot's previous occupant left. (Turned out to be zeros here, so this one was
   harmless — but only by luck.)
4. **Off-by-one alignment** — a block is an anchor at `anchor_idx` PLUS `seq_len`
   deltas, so delta `t` is at `anchor_idx + 1 + t`. Comparing against
   `anchor_idx + t` yields ~0.4 correlation purely from adjacent-token
   similarity, and makes residuals look useless because they are misaligned too.

**The fix that made the instrument trustworthy**: an ANCHOR ROUND-TRIP CHECK —
compare `pool.anchors_K[slot]` against `K_true[anchor_idx]`. The pool stores that
verbatim, so it validates alignment and slot mapping WITHOUT depending on any
reconstruction math. It reads `cos=1.000000 rel=0.00000`. Every probe of stored
data should carry an independent self-check like this from the start.

### Bug 1 (REAL, measured improvement): residuals left in NORMALIZED space

`_compress_block_sync` called `compress_lowrank(...)` WITHOUT `token_norms`.
`compress_lowrank` works in normalized delta space and only un-normalizes its
residuals when given those norms (`if token_norms is not None: res *= norms`).
The caller then un-normalizes the FACTOR (`U_scaled = lr_delta.U * token_norms`)
— so the reconstruction was in ORIGINAL space while its correction stayed in
NORMALIZED space, leaving residuals too small by a per-token `||delta||`.

Since force-exact blocks (digits, alphanumeric codes — the needle) depend
ENTIRELY on residuals, this silently defeated the exact-recall mechanism.

Measured (layer 3, block anchor=0, alignment verified):
```
residual contribution to reconstruction cos : +0.0010  ->  +0.0105   (~10x)
positions WITH a stored residual, cos       :  0.586   ->   0.626
```

### Bug 2 (REAL but measured NEUTRAL here): residuals vs the fp16 recon

The pool stores `U` as int8 and the kernel reconstructs from THAT, but
`compress_lowrank` derives residuals from the fp16 reconstruction — so the
residual corrects a reconstruction the decoder never produces. The batched path
already fixes this (`compression/lowrank.py` ~1232, "Recompute recon against the
SAME int8-dequant U decode reads"); the sync path never received it. Ported.

**Trap when porting**: the batched version omits `* lr_delta.scale` because it
forces `block.scale = 1.0`. The sync path keeps the real scale. Dropping the
term measured WORSE than no fix (cos 0.626 -> 0.487). With the term restored it
is exactly neutral (0.626). Kept — guarded by a try/except fallback, and it
makes the sync path consistent with what decode reads — but labelled
measured-neutral, NOT claimed as a win.

### End-to-end effect (kernel suite stayed 15/15 throughout)

Per-layer cos vs dense at the first decode step (compression-off reference is
0.999999 at every layer):
```
 layer     before      after
     3   0.047131   0.156204   (3.3x, still essentially uncorrelated)
     7  -0.090622  -0.107071
    11   0.041744   0.035919
    15   0.318857   0.381655
    19   0.641366   0.890527
    23   0.721478   0.848605
```

Real, broad improvement — and NOT a fix. Layer 3 remains far from 1.0.

**AND THE END-TO-END NEEDLE OUTPUT REGRESSED**: `...ZEBRA-2024` (4/12 needle
tokens) -> `...ZEBY` (3/12). So every internal correctness metric improved
(reconstruction 0.586->0.626, layer cos 0.047->0.156 / 0.641->0.891 /
0.721->0.849) while the observable output got worse.

This is the SECOND time this session that a provably-more-correct change made
output worse (the first was the dense strict bound, §9g-4). The pattern is
consistent with the compressed path having multiple errors that were partially
CANCELLING: fixing one component in isolation unbalances the cancellation and
can degrade the token stream even as the underlying representation improves.

Practical consequence: **token-level output is a poor optimisation target here.**
The layer differential and the reconstruction probe are the trustworthy signals,
because they compare against ground truth directly instead of through a
sampling decision that only reflects the argmax. Judge future compression fixes
on those, and expect the token stream to stay noisy until the remaining
storage defect(s) are also fixed.

### STILL BROKEN — the next probe is specified

Reconstruction at positions that HAVE a stored residual is 0.626, but by
construction `anchor + recon + residual` must reproduce the exact key. At least
one more defect exists in compression/storage. Next, in order:
  1. Compare `block.residual_K_values` IMMEDIATELY after compress against
     `pool.residual_K_values[slot]` — does the POOL WRITE alter them (dtype,
     truncation, wrong slot)?
  2. Verify `pool.scales[slot] == lr_delta.scale`.
  3. Verify `pool.V_K[slot]` equals `lr_delta.V`'s K-half in the assumed
     `[R, kv, D]` layout (probe this by reconstructing ONE token two ways).
Scope note: the GPU/batched path has NEITHER bug — its `U` and `deltas` are both
original-space and it already recomputes against int8-dequant `U`. This is
sync/Mac-path only, and nothing here ran on real CUDA hardware.

## 9j. THE ARCHITECTURAL ANSWER — MLX stores EXACT KEYS, not corrections.
This is the root difference and the roadmap to fix the CUDA/Triton path.

User directive: "keep the fixes ... refer to mlx for everything so you can fix
active runtime's cuda/triton code properly, like make the code like mlx's".
Line-by-line read of `mlx_dkv_wrapper.py::_compress_block` gives the answer.

### What MLX stores (mlx_dkv_wrapper.py ~3875)

```python
res_k_active = mx.take(block_k_t, top_k_indices + 1, axis=0)   # FULL EXACT KEY
res_v_active = mx.take(block_v_t, top_k_indices + 1, axis=0)   # FULL EXACT VALUE
```

`comp_res_k` holds the **verbatim K/V of the chosen tokens** — NOT
`(delta - recon)`. At decode (`compute_decode_attention_static`) MLX then:

```python
delta_s = mx.where(res_mask, -inf, delta_s)                     # DELETE the lossy twin
dense_k_for_attn = mx.concatenate([res_k_all, dense_k], axis=1) # attend exact row as a REAL token
```

So a residual token is REMOVED from the compressed pool entirely and re-enters
attention as an ordinary dense token at its true position, with exact K and V.
Nothing about it is approximated: not its score, not its value, not its RoPE.

### What DKV's CUDA/Metal path does instead

Stores `(delta_K - recon_K)` — a CORRECTION — and ADDS it to a score whose base
was reconstructed from the lossy SVD and rotated at the block ANCHOR. The token
also keeps participating in the block's V accumulation (`w_total_anc`,
`svd_v_contribution`). It is a patch on an approximation, where MLX performs a
replacement.

**This is why reconstruction at residual positions measures cos 0.626 instead of
~1.0** even after the two §9i fixes: `anchor + recon + residual` can only be
exact if `recon` is the exact reconstruction the residual was computed against,
and the compressed path keeps re-introducing approximation around it.

### Why the earlier DKV_RESIDUAL_EXACT_KEYS attempt (§9g-2) regressed

It implemented HALF of MLX's design: it stored `(exact - anchor)` and made the
kernel REPLACE the score. But it never removed the token from the compressed
pool, so that token still contributed through `w_total_anc` and the SVD V path —
double-counting against its own exact correction. MLX's `-inf` twin-masking is
not an optimisation; it is load-bearing.

### Roadmap to actually match MLX (in dependency order)

1. **Storage**: make residual K/V hold the verbatim exact key/value of the chosen
   token (as MLX does), not a delta and not a correction. Same tensor shapes.
2. **Twin removal**: give the kernel a per-block residual MASK over the S_comp
   delta axis (MLX's `comp_res_mask`) and set those tokens' delta scores to
   -inf so they contribute ZERO to both the softmax and the block's V
   accumulation.
3. **Re-entry**: attend the exact rows as ordinary tokens at their TRUE
   positions — the Metal kernel already has the machinery (`has_exact_res_rope`,
   the full-sequence rope tables, and the `fact` override path which is exactly
   this mechanism at 3 slots/block).
4. **Only then** re-measure. The instrument to use is `recon_final.py` (residual
   positions must reach cos>0.999) and the layer differential — NOT the token
   stream, which §9i showed is a poor optimisation target while multiple
   partially-cancelling errors remain.
5. Port the same three changes to the Triton kernel before enabling on CUDA;
   nothing in this investigation has run on real GPU hardware.

Note the `fact` override path already implements steps 1+3 correctly for its 3
slots (`fact_anchors_K` stores `k_orig[pos]`, kernel REPLACES the score). The
work is generalising that to the full residual set and adding step 2.

## 9. Concrete next steps, in priority order

1. ~~Root-cause §9b-4 (Path A partial needle-recall)~~ **RESOLVED — see §9d.** Root
   cause was a compiled-Metal-kernel bug (D=256 threadgroup/register buffer
   overflow, the source of the NaN/garbage-collapse symptom), plus a second,
   independent kernel bug (rank/pool_rank stride confusion causing every compressed
   block past the first to read the wrong slot's data). Both fixed and verified —
   NaN is gone, output no longer collapses into repeated garbage. A much smaller,
   precisely-characterized residual gap remains (needle recall stops 3/12 tokens in,
   on a moderate — not razor-thin — logit margin); §9d has the exact numbers.
   **UPDATE (same day)**: quantified via exact softmax-at-multiple-temperatures off
   the captured logits (no re-run needed) — the correct continuation has P=13.8% at
   T=1.0 (vs 2.5-4.7% for runner-ups at the three correctly-retrieved steps), rising
   to ~39% at T=1.5. Confirmed genuine close call, not a further structural bug —
   this item is now OPTIONAL polish, not required. (Note: T<1, e.g. 0.2-0.3,
   sharpens toward whatever's already highest and would NOT test this — only T>1
   flattens the distribution enough to matter.)
2. ~~Build the isolated Metal kernel unit test from §9c-6~~ **DONE — see §9e.**
   `ACTIVE_RUNTIME/tests/test_decode_attention_metal_isolated.py`, 11/11 cases pass.
   Found and fixed ONE MORE real bug in the process (unguarded fact-override reads,
   `has_fact`). Re-run this test first if `dkv_decode.metal`/`metal_runtime.mm`
   change again — it's fast (no model loading) and catches exactly this bug class.
3. Decide: is Path A (legacy monkeypatch, ~2700 lines, now transformers-5.x-compatible
   after this session's fixes) worth continuing to maintain, given Path B
   (AttentionInterface, ~500 lines, architecturally cleaner, sidesteps the whole
   QK-norm/gating/calling-convention problem class by construction) already exists
   and mostly works? If Path B is the future, Path A's fixes are still correct and
   worth keeping (for `DKV_USE_ATTENTION_INTERFACE=0` fallback / non-5.x transformers),
   but new effort should probably go into hardening Path B against Qwen3.5-2B's hybrid
   layers specifically (validate the linear-attention layers' cache/state survives
   correctly across decode steps under Path B — the exact bug class §3.1.6 found and
   fixed for MLX has not been checked for Path B at all). Also worth checking: does
   Path B have the same §9b-4/§9c partial-recall gap, or does its cleaner
   AttentionInterface design sidestep it too (Path B does not use
   `fused_decode_attention_combined`/Metal at all as far as this session determined,
   so it may not share this exact bug).
4. Re-run B3/B6/B7 alone (no concurrent heavy work), then run the never-reached
   phases (binding_table onward) to complete the Qwen3.5-2B results collection.
5. Work through the parity-audit action items in rough priority order: (a) port the
   force-exact/skip-compression digit-code mechanism from CUDA to MLX (agent 1,
   finding 1 — confirmed CUDA-is-better case); (b) re-investigate whether CUDA's live
   default block router actually drops answer-critical blocks the way the disabled
   `DKV_DECODE_PRUNE_K` flag's comment says it does (agent 2, finding 1 —§9c-5 already
   ruled this out specifically for the §9b-4 symptom via `DKV_TOPK_BLOCKS=0`, but the
   broader claim about other contexts may still hold); (c) the smaller items in §5.
6. Audit Path B (`dkv_backend.py`, `_dkv_decode_forward_impl`/`_dkv_prefill_forward_impl`
   in `dkv_attention.py`) against MLX the same way the 3 agents did for Path A — this
   was explicitly out of scope for the existing audit since it wasn't known to exist
   yet when the agents were dispatched.

---

## §9k — MLX alignment implemented; §9j's premise OVERTURNED; defect relocated (2026-07-28)

### What was built (done, unit-verified)

MLX's residual design is now implemented on the Metal path behind
`DKV_RESIDUAL_EXACT_KEYS=1` (default OFF), as three coordinated changes:

1. **Storage** — `residual_{K,V}_values` hold the anchor-relative EXACT value
   (`anchor + residual == true K/V`) instead of a correction to the low-rank
   reconstruction. Applied to all three producers (`compress_lowrank`,
   `compress_lowrank_batch`, `kv_runtime_manager`'s inline path). K and V now
   share ONE index set, as MLX does — substitution is only well-defined when a
   token's score and value are replaced together.
2. **Twin removal** — the kernel backs the token's low-rank value estimate out
   of the block accumulation (`w * (res_val_V - svd_v)`), so the exact value
   replaces it rather than stacking on top. This is what the first rollout was
   missing, and why score-only substitution regressed: the token double-counted.
   Implemented exactly like the `fact` override (section 5), which has always
   done this correctly; positions that are both fact and residual are handled
   once, not twice.
3. **Rotation** — exact keys rotate at the block ANCHOR, not `anchor + t`. See
   below. `residual_exact_keys` no longer depends on `has_exact_res_rope`.

Verified by `tests/test_decode_attention_metal_isolated.py`: **19/19** against an
independent float64 reference — 15 correction-form + 4 new exact-keys cases (E1–E4,
including GQA/partial-rotary, fact/residual overlap, and realistic Qwen3.5 scale).
The suite re-execs itself in a subprocess for the exact-keys half, because the
flag is cached on first kernel launch.

### PTA is position-exact — so `DKV_RESIDUAL_EXACT_ROPE` DOUBLE-ROTATES

`_preprocess_block_for_compression` RoPE-rotates every token of a block by its
WITHIN-BLOCK offset *before* subtracting the anchor (`kv_runtime_manager.py`
~2927-2966, unconditional). RoPE composes, so the kernel's anchor rotation lands
delta token `t` at absolute `anchor_pos + t + 1` — its true position — on its own.
Project-Then-Attend is therefore position-EXACT, not "inherently lossy for
position-sensitive tokens" as previously recorded.

Consequence: rotating residual/fact values at `anchor_pos + rpos`
(`DKV_RESIDUAL_EXACT_ROPE`, **default ON**) applies the offset a SECOND time on
this path. Left as-is for the correction form only because that flag was
validated on the Triton/CUDA path at Qwen2.5 and is not this session's to flip
without a GPU measurement — but it is very likely wrong here.

### §9j WAS WRONG — residual storage was exact all along

§9j concluded DKV's residual design was the defect, from a reconstruction probe
reading cos 0.626 at residual positions. **That probe was broken twice over**: it
compared rotated storage against UNROTATED ground truth, and (once rotation was
added) used theta=10000 instead of the model's `rope_theta=1e7`. Corrected:

| positions | cos | rel_err |
|---|---|---|
| WITH a residual | **1.000000** | 0.00034 |
| without a residual | 0.81–0.83 | ~0.57 |

Residual positions reconstruct EXACTLY. The no-residual figure is honest rank-24
truncation error. Nothing about the correction form was broken. Every "residual
reconstruction is broken" conclusion in §9d–§9j is a probe artifact.

### The defect is NOT in compression

Decisive control: put the needle at the END of the prompt, inside the
UNCOMPRESSED dense/recency window (`scratchpad/test_needle_at_end.py`).

- DKV disengaged (`DKV_ENGAGE_THRESHOLD=999999`): `ZEBRA-4471-QUARTZ` ✓
- DKV engaged: `ABC123` / `ABC-123456789` — generic hallucinated codes ✗
- `DKV_DENSE_VALID_LEN=1` (strict dense bound): still ✗

A needle that never enters a compressed block is still lost. **Compression
fidelity is not the bug** — look at dense-window assembly, routing, or the
prefill path.

### BLOCKER: the runtime is nondeterministic

Two runs, identical build, identical env, `temperature=0.0` (greedy), produce
DIFFERENT output. Added `DKV_SYNC_COMPRESS=1` (forces inline compression, since
background SVD means which blocks are compressed by decode time depends on thread
timing) — it pins compression but does **not** restore determinism, so at least
one more source remains. Candidates not yet eliminated: the TieredBlockStore
evictor, BlockPrefetchEngine, and the RSS-threshold MPS pressure monitor, all of
which are time/state dependent.

**Until this is fixed, no A/B on this path means anything** — two runs of the SAME
build disagree by more than most changes under evaluation. This is why the token
stream flip-flopped (`ZEBRA` → `ZEBRA-2024` → `ZEBY`) and why per-layer cosines
moved incoherently across configs. Fix determinism FIRST; re-measure everything
after.

---

## §9l — Pool-write probe: all three checks CLEAN, no storage defect remains (2026-07-28)

Ran the specified probe program (`scratchpad/probe_pool_write.py`, modelled on
`recon_final.py` and keeping its ANCHOR ROUND-TRIP self-check, which read
cos=1.000000 / rel=0.000000 on every block). `DKV_SYNC_COMPRESS=1` so compression
runs inline and the spy sees the real object.

**1. Does the pool write alter residuals?** No.

| block | produced | in pool | positions | values |
|---|---|---|---|---|
| anchor=0 (force-exact, holds the needle) | 256 | 64 | match | cos 0.999725, max\|diff\| 0.204 |
| anchor=257 | 38 | 38 | match | cos 1.000000, max\|diff\| **0** |
| anchor=512 | 38 | 38 | match | cos 1.000000, max\|diff\| **0** |

- 256→64 on the force-exact block is **documented, deliberate** truncation to the
  pool's `max_residual_tokens` (`lowrank.py:437` — "the pool truncates to its
  per-block residual capacity, keeping the earliest positions, where the exact
  content sits"). Verified the assumption holds here: the needle occupies prompt
  tokens 8–19, inside the kept window (absolute 1..64). Not a live bug for this
  test, but it IS a silent cliff for any force-exact block whose exact content
  sits past position 64 — worth a warning at minimum.
- The non-zero diff on that block is the int8-dequant recompute overwriting
  `block.residual_K_values` AFTER `compress_lowrank` returns. Expected — that is
  the §9i fix doing its job.

**2. `pool.scales[slot]` vs `lr_delta.scale`** — PASS. The probe's initial
"MISMATCH" was its own tolerance being tighter than fp16 storage:
`0.451478 → 0.451416`, `0.477771 → 0.477783`, `0.361699 → 0.361816` are the exact
fp16 roundings of the delta values.

**3. `pool.V_K[slot]` vs `lr_delta.V`'s K-half as [R, kv, D]** — PASS. cos=1.000000;
reconstructing one token both ways agrees to 1e-3 (fp16 noise).

### Probe bug found and fixed mid-run (same class as the 0.626 artifact)

The first version read `block.residual_K_values` after `_compress_block_sync`.
That is a **property that falls back to `pool.residual_K_values[pool_idx]`** once
the block-local copy is nulled — which that function does before returning. So it
compared the pool against ITSELF and reported a vacuous cos=1.000000 / diff=0.
Fixed by reading the `LowRankDelta` that `compress_lowrank` actually returned.
Any future probe touching `block.residual_*` must do the same.

### Verdict

No remaining compression/storage defect. This independently confirms §9k from a
different angle: the pool faithfully holds what compression produced, and §9i's
inference ("reconstruction at residual positions is 0.626 ⇒ another storage bug")
rested on the broken-basis probe. Corrected, residual positions in the pool
reconstruct at cos=1.000000.

**Stop looking at compression.** §9k's control test stands: a needle in the
UNCOMPRESSED dense window is still lost. Next target is dense-window assembly /
routing / prefill — and before that, runtime determinism (§9k blocker).

### Needle test with the two §9i compression fixes

`ZEBY` / FOUND=False, against a `ZEBRA-2024` baseline — a regression on its face.
But two runs of the SAME build later produced `ABC123` and `ABC-123456789`, so a
single-run token comparison is not evidence in either direction. Both readings are
draws from an uncharacterised distribution. The §9i fixes remain justified by the
storage measurements; they are **not** demonstrated to help end-to-end, and a
verified storage improvement that does not move end-to-end behaviour is not a fix
for the user's bug.

---

## §9m — Determinism switch + dense-window invariant, both aligned to MLX (2026-07-28)

### `DKV_DETERMINISTIC=1` — no background thread may mutate runtime state

Every background mutator in this runtime fires on a WALL-CLOCK timer, so which
blocks are resident (and therefore routable) when decode starts depends on thread
scheduling:

| thread | site | fires |
|---|---|---|
| async SVD compressor | `async_compressor.py:118` | on submit |
| pager eviction loop | `paged_kv_store.py` `_bg_eviction_loop` | every 2.0 s |
| pager prefetch loop | `paged_kv_store.py` `_bg_prefetch_loop` | on demand |
| BlockPrefetchEngine | `block_prefetch_engine.py` | on demand |
| MPS pressure monitor | `hf_dkv_wrapper.py:246` | every 5.0 s — `gc.collect()` only, harmless |

`DKV_DETERMINISTIC=1` disables the first four at once (and implies
`DKV_SYNC_COMPRESS=1`). MLX — the reference, which IS reproducible — runs a
single background thread and no timed eviction/prefetch at all, so this flag is
the MLX-parity configuration for measurement. Leave it OFF in production, where
the overlap is the point. Expect noticeably slower prefill: SVD no longer
overlaps.

### The dense window violated its own contract

`assemble_dense_window_kv` computes `L_dense` from the BLOCK LIST (step 1,
`sum(1 + active_len)`) but writes the workspace through CACHED per-block offsets
(step 3). Nothing reconciled the two, and it returned the step-1 sum. Its own
docstring promises "positions >= L_dense contain stale/zero data" — an assumption,
not a fact.

Three MLX-parity fixes:

1. **Verify the packed-from-0 invariant.** Before reusing cached offsets, check
   each block's offset equals the running sum of preceding block sizes; repack if
   not. MLX cannot reach this state — it keeps one contiguous buffer and compacts
   on eviction (`dense_keys[..., :remaining] = dense_keys[..., block_size:dense_len]`),
   so the invariant is structural rather than cached.
2. **Return the ACTUAL written extent** (`curr_idx`), not the block-list sum, with
   a one-time warning if they ever disagree. Now "positions past this are stale"
   is a fact.
3. **Zero the workspace tail** past what was written — MLX does exactly this
   (`dense_keys[layer_idx][0, :, dense_len:] = 0.0`, lines 2259/2281/3952) at
   every point its window shrinks. The workspace is reused across steps and
   layouts, so without it the tail holds REAL keys from an earlier layout — and
   the decode kernel bounds its dense loop by the padded ROW STRIDE unless
   `DKV_DENSE_VALID_LEN` is set, i.e. it attends those stale keys as live tokens.
   They win real attention mass and they change run to run.

Zeroing is defence in depth, not the fix: a zero key still scores 0 and takes
`exp(0)` weight in the softmax denominator. Correctness needs the `-inf` mask.

### `DKV_DENSE_VALID_LEN`'s "do not flip" note is retracted

The comment in `metal_runtime.mm` claimed the strict bound was measured worse
('...ZEBRA' -> '...Z'). That A/B predates `DKV_DETERMINISTIC` and is a single-run
token comparison on a runtime later shown to be nondeterministic at temperature 0
— it could not have measured this either way. The note now says so. MLX masks
unconditionally and has no equivalent flag; with the invariant enforced above, the
strict bound should become the default if it holds up under
`DKV_DETERMINISTIC=1`. **Re-measurement owed.**

---

## §9n — TWO REAL BUGS: the 768-token dense cap, and a stale kernel binary (2026-07-28)

### BUG 1 — the decode kernel silently dropped every dense token past row 767

`dkv_decode.metal` bounded its dense loop by `min(L_dense, DKV_MAX_DENSE_SHARED)`
where `DKV_MAX_DENSE_SHARED = 768` — the width of the `dense_w_shared`
threadgroup array. Tokens past row 767 were not down-weighted or approximated:
**they were never scored at all.**

`max_dense_len` is `recency_window + block_size` = **1419** at the default mid
preset, so any prompt with a full recency window loses the NEWEST ~650 tokens of
its own dense window. Measured on the §9k control prompt (`probe_dense_window.py`):

```
workspace=(2, 1419, 256)  L_dense=993  max_dense_len=1419
needle abs 8366/8367/8374/8375 -> rows 943/944/951/952, cos=1.0000
```

The needle sat at rows 943–952. The kernel stopped at 767. The dense window held
it **exactly**, at its **correct RoPE position**, and the model never saw it.

The host-side warning for this (`metal_runtime.mm`) was gated on
`dkv_dense_strict_valid()`, which defaults OFF — so it stayed silent in precisely
the configuration that hit the bug.

**Fix:** PASS 2 now walks the dense window in TILES of `DKV_MAX_DENSE_SHARED`,
rebuilding each tile's weights in the shared buffer. The shared array bounds only
the tile, never how much context is attended. PASS 1 is uncapped (the online
softmax merge needs no storage). Cost: one extra q·k dot per dense token per
step. MLX has no such limit — it runs SDPA over the whole dense buffer with an
`arange < dense_len` mask.

Regression cases added (`15_dense_exceeds_shared` L_dense=993,
`16_dense_exceeds_shared_no_slots` L_dense=1419). Suite: **21/21** (17
correction-form + 4 exact-keys). Both new cases fail against the old kernel.

### BUG 2 — the runtime was loading a kernel binary from 2026-07-23

`import dkv_core` resolves to the first match on `sys.path`. There are several
candidates: a copy at the REPO ROOT, the real build in
`ACTIVE_RUNTIME/native_core/dkv_core/`, and two editable-install `.pth` finders.
Rebuilding in the build directory does **not** update the root copy.

`runtime/dkv_attention.py` does a bare `import dkv_core` → it got the root copy,
dated **2026-07-23** (634 KB, vs 652 KB freshly built). The isolated kernel tests
insert the build directory FIRST, so they exercised the new code and passed —
while every end-to-end run in this session executed the July-23 kernel.

**This invalidates every end-to-end measurement taken before this was found**,
including this session's needle tests, layer differentials, and the
`DKV_DENSE_VALID_LEN` A/B — and, since the root copy dates to 2026-07-23, quite
possibly conclusions recorded in earlier sessions too. Treat any end-to-end
result from that window as describing unknown kernel code.

**Fix:** a stale-binary guard at the import in `dkv_attention.py` compares the
loaded `.so`'s mtime against `dkv_decode.metal` and prints a loud warning with
the rebuild command. Verified silent on a fresh build, loud on a stale one.

### Determinism: NOT solved — earlier claim retracted

`DKV_DETERMINISTIC=1` (§9m) disables all four background mutators. Two runs on
the stale kernel gave identical output, and I recorded that as determinism
achieved. On the fresh kernel two runs gave `ABC` and `ABC1234567879`. **n=2 was
never enough to claim it.** Retracted: the runtime is still nondeterministic at
temperature 0 with every background thread disabled, so the source is NOT thread
timing. Untried: `PYTHONHASHSEED` (set iteration order in routing), MPS kernel
reduction order, and uninitialised reads.

### Still open

The needle is STILL not recovered after the dense fix (`ABC…`, FOUND=False). What
is now excluded, each by direct measurement rather than inference: compression
storage (§9l), dense-window assembly and RoPE positions (§9m/§9n), the sparse LSE
bias (defaults 0.0 in BOTH runtimes — not a divergence), and the in-kernel merge
(21/21 vs a float64 reference). Remaining surface: block ROUTING (which
compressed blocks are selected) and the call site that feeds the kernel. Fix
determinism first — with n=2 samples nothing here is measurable.

---

## §9o — NONDETERMINISM ROOT-CAUSED AND FIXED: an unseeded rSVD sketch (2026-07-28)

### The cause

`native_core/mac_utils.py::mlx_svd_lowrank` built the randomized-SVD projection
with `mx.random.normal(shape=(d, r_proj))` — **unseeded**. Every process sketched
the matrix differently, so the rSVD returned a different `U`/`V`. Because the
factorization is TRUNCATED to `rank`, a near-degenerate spectrum means which
directions survive truncation depends on the sketch — so the reconstruction
genuinely changed run to run, at temperature 0.

(Sign flips alone would have been harmless: they cancel between `U` and `Vh`.
Truncation is what makes it bite.)

Three `torch.randn` sketches in `compression/lowrank.py` (lines ~312, ~948,
~1543) had the same defect, as did the SRL descriptor basis
(`kv_runtime_manager.py:736`). On this Mac the MLX path is the one that runs.

### How it was isolated

A pool fingerprint (`scratchpad/probe_pool_hash.py`) run twice showed a very
specific signature: `anchors_K`, `anchors_V`, `scales`, `U_scale`, `seq_lens`,
the slot mapping and the dense/compressed split were **identical**, while `U`,
`V_K`, `V_V` and the residuals derived from them **differed** — i.e. exactly the
tensors downstream of Omega, and nothing else. That pointed straight at the
sketch. `compress_lowrank` on a fixed input across two processes then reproduced
it with no model involved.

This is why the earlier suspects all came up empty: it was never thread timing
(`DKV_DETERMINISTIC` disables all four background mutators and did NOT fix it),
never routing (`DKV_TOPK_BLOCKS=0` did NOT fix it), never `PYTHONHASHSEED`.

### The fix

Seeded generators at all five sites, sharing `DKV_RSVD_SEED` (default 0). Uses an
explicit key/generator rather than a global `manual_seed` so it neither perturbs
nor depends on global RNG state.

**Verified:**
- `compress_lowrank` on fixed input: bit-identical across 3 processes.
- Pool fingerprint: bit-identical across runs (all 12 tensors).
- Needle test: 3 identical runs — **with background threads ENABLED**
  (`DKV_DETERMINISTIC` not set), confirming they were never the cause.

**A/B on this runtime is now valid for the first time in this investigation.**
Every measurement recorded before this — in this session and earlier — was taken
on a runtime that could not reproduce itself.

### PROBE TRAP (again): `hash()` is salted

`probe_pool_hash.py` first used builtin `hash(bytes)`, which is salted per process
by `PYTHONHASHSEED` — it reported every tensor as different even when the bytes
matched, and briefly looked like the fix had regressed. Use `hashlib`. This is the
third probe defect in this investigation to produce confident, wrong numbers.

### First reproducible baseline (needle at START, so it sits in a compressed block)

| config | L3 | L7 | L11 | L15 | L19 | L23 |
|---|---|---|---|---|---|---|
| default (adaptive rank, top-K) | 0.093 | 0.074 | −0.049 | 0.707 | 0.878 | 0.872 |
| attend-all (`DKV_TOPK_BLOCKS=0`) | 0.185 | **0.721** | 0.057 | **0.903** | 0.914 | 0.929 |
| flat rank + attend-all | 0.177 | 0.185 | 0.079 | 0.831 | 0.889 | 0.894 |

- Attend-all materially helps layers 7/15/23 ⇒ **routing drops useful blocks**.
- Flat rank (MLX's scheme) is WORSE here — flat 32 is below the 48 that layers
  7/11/15 get under the taper. Not an MLX-parity win; leave adaptive rank on.
- Layers 3 and 11 stay broken under every config. They are the target.

### `get_layer_rank` contradicts its own docstring

Documented: 0-15% → base_rank, 15-50% → base_rank, 50-79% → 0.75×, 79%+ → 0.50×.
Actual, with num_layers=24 / base_rank=32:

| layer | depth | doc says | actual |
|---|---|---|---|
| 3 | 12% | 32 | **24** |
| 7 | 29% | 32 | **48** |
| 11 | 46% | 32 | **48** |
| 15 | 62% | 24 | **48** |
| 19 | 79% | 16 | 16 |
| 23 | 96% | 16 | 16 |

It never returns base_rank for any full_attention layer, and it exceeds the
configured rank (48 > 32) for three of them — the docstring explicitly promises
the configured rank is a ceiling ("no silent VRAM inflation beyond --rank").
Layer 3, the worst-performing layer, is also the only early layer pushed BELOW
base_rank. Unresolved: whether the schedule or the docstring is wrong.

---

## §9p — Storage EXONERATED per layer; the defect is at DECODE, layer-specific (2026-07-28)

`scratchpad/recon_all_layers.py` measures pool→ground-truth reconstruction for
every full_attention layer in one run (correct rotation basis, real rope_theta,
per-layer anchor round-trip check).

| layer | rank | anchor | SVD-only | +resid | res-pos | attn vs dense |
|---|---|---|---|---|---|---|
| 3 | 24 | 1.00000 | 0.8203 | 0.8591 | 0.99996 | 0.185 |
| 7 | 48 | 1.00000 | 0.9311 | 0.9462 | 0.99988 | 0.721 |
| 11 | 48 | 1.00000 | **0.9349** | 0.9491 | 0.99989 | **0.057** |
| 15 | 48 | 1.00000 | 0.9407 | 0.9536 | 0.99994 | 0.903 |
| 19 | 16 | 1.00000 | 0.8186 | 0.8591 | 0.99997 | 0.914 |
| 23 | 16 | 1.00000 | 0.8177 | 0.8570 | 0.99997 | 0.929 |

Storage is healthy and behaves exactly as it should: fidelity tracks RANK
(48 → 0.93-0.94, 16/24 → 0.82), anchor round-trip is 1.00000 on every layer, and
residual positions are ~0.9999 on every layer.

**Storage does not explain the attention quality at all:**
- Layers 19/23 have the SAME storage fidelity as layer 3 (0.818 vs 0.820) but
  5× the attention quality (0.91/0.93 vs 0.185).
- Layer 11 has the BEST storage of any layer (0.935) and the WORST output (0.057).

So the layer-3/11 defect is at DECODE. Also ruled out there:
- **rank argument** — the call site passes `_layer_active_rank` (the layer's own
  rank), and layer 11's rank 48 == pool_rank 48, so the kernel reads every stored
  component. The kernel's rank-48 shared arrays (`q_proj_cached[32*48]`,
  `red_proj_temp[64*48]`) sit exactly at capacity, not past it.
- **routing selection** — attend-all (`DKV_TOPK_BLOCKS=0`) lifts layer 7 from
  0.074 to 0.721 but leaves layer 11 at 0.057. Selection is not what is wrong
  there.
- **the kernel's own math** — 21/21 against a float64 reference.

### Where to pick this up

The remaining surface is what the call site hands the kernel PER LAYER, other
than rank — the per-layer slot list, the anchor cos/sin slice
(`decode_cos_sliced`/`decode_sin_sliced`, which are cached keyed on
`routing_version`), and the residual/fact tensors. Two of those are CACHES keyed
on a version counter, and this investigation has already found two separate
bugs of exactly that shape (the dense-workspace offset cache in §9m, and
`dense_pos_tensor_cache` which turned out to be dead). Instrument one decode step
at layer 11 and layer 23, dump every tensor passed to
`fused_decode_attention_combined`, and diff their provenance.

Do it with the runtime now reproducible (§9o) — that is what makes such a diff
meaningful.

---

## §9q — MLX vs CUDA on the SAME prompt: premise confirmed, one divergence fixed (2026-07-28)

### MLX SOLVES this prompt on this model. CUDA does not.

Never actually verified for Qwen3.5-2B until now — the "MLX works" premise came
from Qwen2.5. Ran the identical needle prompt (8217 tokens, needle at the START
inside a compressed block, `DKV_ENGAGE_THRESHOLD=4096`) through
`MLXDKVWrapper(model_id="Qwen/Qwen3.5-2B")`:

```
[MLX] rank=48 base_rank=32 layer_adaptive_rank=True
[MLX] prompt tokens=8217
[MLX] ANSWER TAIL: '...ZEBRA-4471-QUARTZ'
[MLX] FOUND: True
```

Same model, same prompt, same settings. **The divergence is real and it is on the
CUDA side, not the model.** Script: `scratchpad/needle_mlx.py`.

### Divergence FOUND AND FIXED: residual twin-exclusion defaulted off

MLX gates the exact-value residual substitution on `DKV_RESIDUAL_EXCLUDE_SVD`,
which **defaults to "1"** (`mlx_dkv_wrapper.py:1763`). The CUDA equivalent
(`DKV_RESIDUAL_EXACT_KEYS`, the storage + twin-removal + rotation work from §9k)
defaulted to **"0"**. Straight divergence from the reference.

Per-layer attention cos vs dense, reproducible runtime, needle prompt:

| | L3 | L7 | L11 | L15 | L19 | L23 |
|---|---|---|---|---|---|---|
| off (old default) | 0.093 | 0.074 | −0.049 | 0.707 | 0.878 | 0.872 |
| **on (MLX default)** | **0.116** | **0.782** | **0.048** | **0.892** | **0.910** | **0.926** |

Every full_attention layer improves. **Flipped to default ON** on both sides
(`lowrank._exact_keys_enabled`, `metal_runtime.dkv_residual_exact_keys` — these
read the same variable and their defaults must match or a build stores one form
and decodes the other). `DKV_RESIDUAL_EXACT_KEYS=0` reverts.

Test harness now runs ALL cases under BOTH conventions: **21/21 with the flag at
0 and 21/21 at 1**.

### Checked and NOT divergent

- **Rank schedule.** CUDA's `get_layer_rank` is a faithful port of MLX's:
  `<0.25 → 0.75×base`, `<0.75 → 1.5×base`, `else → 0.5×base` — identical
  24/48/48/48/16/16 at layers 3/7/11/15/19/23 for base 32. Only CUDA's DOCSTRING
  was stale (it described an older schedule and claimed base_rank was a ceiling,
  which the middle band has long exceeded by design — MLX allocates its pool at
  `base_rank*1.5` precisely to fit it). Docstring corrected. An earlier
  "flat rank" A/B here was misleading: `DKV_LAYER_ADAPTIVE_RANK=0` gives CUDA flat
  **32**, whereas MLX's pool capacity is **48**, so that test compared the wrong
  thing.
- **MLX zero-pads U/V to pool rank and always reads pool_rank components**; CUDA
  passes the layer's own rank. Equivalent — compression only ever produced
  `layer_rank` components.
- **Pool slot allocation** (`scratchpad/probe_slots.py`): each of the 6
  full_attention layers gets 30 slots, cleanly interleaved (stride 12, 2 per
  layer per group). Ranges interleave but **zero shared slots** — no cross-layer
  aliasing.

### Still open — layers 3 and 11

Broken under EVERY CUDA config tried (default, attend-all, flat rank, exact-keys):
L3 ≈ 0.09–0.19, L11 ≈ −0.05–0.08, while L15/19/23 reach 0.89–0.93. Storage for
those layers is healthy and uniform (§9p), rank is correct, slots are clean,
routing selection is not it (attend-all lifts L7 but not L11), and the kernel is
21/21 against a float64 reference.

Next: capture MLX's per-layer attention output for the same prompt and diff it
against CUDA's layer by layer. MLX is now a known-good oracle ON THIS PROMPT, so
that comparison localises the remaining defect directly instead of by elimination.

---

## §9r — MLX vs CUDA, layer by layer: divergence STARTS AT LAYER 3 (2026-07-28)

`scratchpad/hidden_states.py mlx|cuda` captures the residual stream after every
decoder layer at the first decode step, same prompt, same fp16 weights (preset
"mid", no quantization on either side). Residual stream rather than the attention
sublayer: same shape and same meaning in both frameworks, no gating/o_proj
convention to reconcile.

| layer | cos(MLX, CUDA) | rel_err | ‖MLX‖ | ‖CUDA‖ | kind |
|---|---|---|---|---|---|
| 0 | 0.999983 | 0.006 | 2.04 | 2.04 | linear |
| 1 | 0.999979 | 0.007 | 2.50 | 2.50 | linear |
| 2 | 0.999966 | 0.008 | 3.09 | 3.09 | linear |
| **3** | **0.778125** | **0.628** | **3.65** | **4.63** | **FULL_ATTN ← first divergence** |
| 4 | 0.830095 | 0.561 | 3.87 | 4.35 | linear (partial recovery) |
| 5 | 0.858501 | 0.524 | 4.14 | 4.28 | linear |
| 6 | 0.897367 | 0.450 | 4.53 | 4.61 | linear |
| 7 | 0.763002 | 0.697 | 4.60 | 4.49 | FULL_ATTN ← drops again |
| 11 | 0.485737 | 0.928 | 5.73 | 7.19 | FULL_ATTN |
| 15 | 0.645878 | 0.911 | 11.62 | 10.17 | FULL_ATTN |
| 19 | 0.742075 | 0.699 | 20.56 | 21.87 | FULL_ATTN |
| 23 | 0.765708 | 0.644 | 32.89 | 41.30 | FULL_ATTN |

### What this establishes

1. **Layers 0-2 agree to cos 0.99998.** The two runtimes are numerically
   equivalent right up to the first DKV layer — which validates the comparison
   (same weights, same prompt, comparable capture point) and rules out any
   framework-level, tokenizer, RoPE-table or embedding difference.
2. **The divergence begins EXACTLY at layer 3** — the first `full_attention`
   layer, i.e. the first layer DKV compresses. Not before.
3. **The linear layers partially RECOVER** (4→0.830, 5→0.858, 6→0.897) and each
   subsequent full_attention layer knocks it back down. Clean sawtooth: the model
   self-corrects, DKV re-injects the error. Every drop lands on a DKV layer.
4. **CUDA's layer-3 residual is 27% LARGER in norm** (4.63 vs 3.65). A magnitude
   error, not just a direction error — the attention output carries too much
   mass. Note zero-padded dense rows would make it SMALLER (they add nothing to
   the numerator while inflating the denominator), so this points the other way:
   toward something being counted twice, or a normalisation that is not dividing
   by the full denominator.

### Next step, now well-posed

Instrument ONE decode step at layer 3 in both runtimes and compare the attention
output directly (pre-o_proj), then the softmax denominator and the sparse/dense
LSE split. CUDA's own dense window and compressed storage are both verified exact
(§9m, §9p) and the kernel is 21/21 against a float64 reference — so the suspect is
how the two contributions are COMBINED and normalised, which is exactly where a
"too much mass" error would live.

That is a single-layer, single-step comparison against a known-good oracle, which
is a far better position than this investigation has been in at any earlier point.

---

## §9s — ROOT DIVERGENCE: DKV_ENGAGE_THRESHOLD means different things in the two runtimes (2026-07-28)

### The bug

MLX treats `DKV_ENGAGE_THRESHOLD` as a **hard override of the recency window**
(`mlx_dkv_wrapper.py`, MLXKVBlockManager.__init__):

```python
env_engage = os.environ.get("DKV_ENGAGE_THRESHOLD")
if env_engage is not None:
    recency_window = int(env_engage)
else:
    recency_window = max(512, round_up_512(num_layers * head_dim * factor))
```

CUDA used it **only** as an engage gate and took its recency window from a
separate `DKV_RECENCY_WINDOW`, default flat **512**.

So every comparison ever run with `DKV_ENGAGE_THRESHOLD=4096` gave MLX a
**4096-token** dense window and CUDA a **512-token** one. On the 8217-token needle
prompt that is 4122 tokens held exactly by MLX versus 993 by CUDA — 16 compressed
blocks versus 30. **The two runtimes were never running the same experiment.**

### The fix

`KVRuntimeManager._resolve_recency_window()` ports MLX's precedence exactly:
explicit `DKV_RECENCY_WINDOW` → `DKV_ENGAGE_THRESHOLD` → MLX's
`max(512, round_up_512(num_layers*head_dim*DKV_DENSE_WINDOW_FACTOR))` formula
(which also replaces a flat 512 that ignored model shape). Used by both call
sites.

`max_residual_tokens` for the `mid` preset raised 64 → **128**, MLX's flat default
at every preset (the 40/64/128 preset ladder is a CUDA-only invention). Both are
required — see below.

### Measured: per-layer residual-stream cos vs MLX

Mean over full_attention layers, 8.2k needle prompt, reproducible runtime:

| config | mean | L3 | L7 | L11 | L15 | L19 | L23 |
|---|---|---|---|---|---|---|---|
| before (512 window, res 64) | 0.697 | 0.778 | 0.763 | 0.486 | 0.646 | 0.742 | 0.766 |
| window fix only (res 64) | 0.631 | 0.984 | 0.737 | 0.636 | 0.404 | 0.584 | 0.443 |
| **both fixes** | **0.836** | **0.972** | **0.907** | **0.893** | 0.755 | 0.773 | 0.714 |

Layer 3 — the first compressed layer, where §9r localised the divergence — goes
**0.778 → 0.972**, and layer 11 **0.486 → 0.893**.

**Both changes are required.** The window fix ALONE is a net regression (0.697 →
0.631): a larger dense window means the surviving compressed blocks carry more of
the answer, and at `max_residual=64` they cannot. The earlier "res128 vs res40:
identical quality" A/B measured PROSE SYNTHESIS, the one workload the config
comment itself says is insensitive to this dial — it never generalised to recall.

### Still not fixed, and a new problem

The needle is still not retrieved (`ZEBRA`, partial). And at this larger footprint
(4096 window + 2x residual pool) two identical runs diverged again — one produced
a degenerate repetition loop. The rSVD seeding (§9o) removed the sketch as a
source, so this is a SECOND, footprint-dependent source.

**CONFIRMED:** with `DKV_DETERMINISTIC=1` (which disables the pager eviction /
prefetch threads and the BlockPrefetchEngine) both runs are identical (`ZEBRA`
twice). So the pager IS the second source, and it only engages once the larger
dense window + 2x residual pool push the allocation over its eviction threshold —
which is why §9o's seeding looked sufficient at the old, smaller footprint.

The eviction path evicts on a 2.0 s wall-clock timer (`paged_kv_store
._bg_eviction_loop`), so WHICH blocks are resident at decode depends on thread
timing. Fixing it properly means making eviction deterministic — evict on a
counter/watermark checked synchronously at ingest, not on a timer — rather than
just disabling it. Until then, pass `DKV_DETERMINISTIC=1` for any measurement at
these settings.


---

## §9t — §9s's config change: metric up, behaviour down (SUPERSEDED by §9u — alignment KEPT)

§9s changed two defaults to match MLX (DKV_ENGAGE_THRESHOLD as a recency-window
override, mid-preset max_residual 64 -> 128). Per-layer cos vs MLX went
0.697 -> 0.836, layer 3 0.778 -> 0.972.

**End-to-end output got worse and became unstable.** Same prompt, repeated runs:

```
before (512 window, res 64):  'ZEBRA'   stable
after  (4096 window, res 128): 'ZEB' / 'ZHEBENMONG' / a degenerate repetition loop
```

The larger dense window plus a 2x residual pool pushes the allocation into
eviction on an 8 GB machine, and the evicted state is what the model reads. That
also explains the returning nondeterminism: eviction is state- and
timing-dependent.

Halving a genuine workspace over-allocation did NOT rescue it (see below), so
this is not simply slack that could be reclaimed.

**NOT reverted** — see §9u. The regression turned out to be CUDA-side limits
that MLX's configuration exposed, not the configuration being wrong. MLX's configuration works IN MLX, whose memory layout differs;
transplanting it requires solving the memory story first.

**Lesson worth keeping:** per-layer cosine-vs-MLX and end-to-end answer quality
moved in OPPOSITE directions here. A metric that improves while the artifact
degrades is not a fix. Any future config alignment must be validated end-to-end,
not on the layer metric alone.

### Kept from this round (independent, safe)

- **max_dense_len over-allocation fixed.** CUDA computed
  `(ceil(recency/block)+3) * (2*block+1)` = 9747 rows at recency 4096, versus
  MLX's `recency + block_size` = 4352 — 2.24x, ~265 MB of extra K+V workspace
  across 24 layers. It multiplied a per-block worst case by a padded block COUNT,
  which double counts: the window holds ~recency tokens in TOTAL. Now
  `recency + 2*block_size`. Under-allocating is safe —
  `assemble_dense_window_kv` already trims oldest blocks with a warning.
- **Pager timed eviction is now opt-in** (`DKV_PAGER_BG_EVICT=1`, default off).
  It was REDUNDANT: `maybe_evict()` already runs synchronously after every ingest
  (`kv_runtime_manager`), and the loop called the same method on a 2.0 s timer —
  no extra capability, only nondeterministic timing. MLX has no timed eviction.
  (This alone did NOT restore determinism at the larger footprint; the remaining
  source there is the eviction *state*, driven by the oversized allocation.)

### Where this leaves the investigation

Confirmed and unchanged: MLX solves this prompt, CUDA does not (§9q); the
divergence starts at layer 3, the first compressed layer (§9r); storage, dense
window, and the kernel are each verified correct in isolation (§9l, §9m, §9p,
21/21).

The open question is now sharper: CUDA's layer-3 attention can be brought to
cos 0.972 vs MLX by giving it MLX's dense window — i.e. by COMPRESSING LESS. That
says the residual gap is dominated by compression fidelity at this rank, not by a
discrete bug in the decode path. Either the reconstruction has to get better at
the same budget, or the memory cost of MLX's budget has to come down enough to
afford it here. The latter is the more tractable of the two and is where
max_dense_len above is a first (insufficient) step.


---

## §9u — Why MLX's config broke CUDA: three CUDA-side limits it exposed (2026-07-28)

MLX runs a 4096-token dense window and 128 residuals on this same 8 GB Mac
without trouble. So the §9t regression was never "MLX's config is too big" — it
was CUDA-side limits that only that configuration reaches. Alignment KEPT
(`DKV_MLX_RECENCY_SEMANTICS` now defaults ON, mid-preset `max_residual` = 128);
three real bugs found and fixed:

### 1. The decode kernel had a hard 64-residual limit, silently violated

`dkv_decode.metal` declared `res_pos_K_shared[64]` / `res_pos_V_shared[64]`, and
the WRITE loops were guarded at 64 — but the **READ** loops iterated to
`max_residual`. With `max_residual = 128` the kernel read indices 64..127 PAST
the array, picking up whatever threadgroup memory followed (the V positions, the
fact positions, the dense weights) and interpreting those bytes as residual token
POSITIONS. Those bogus positions then matched real tokens and triggered score
replacements and value substitutions for them.

Nothing crashed — the writes were in bounds. The output just became garbage, and
non-reproducibly so, since it depended on adjacent memory contents. **Any pool
configured above 64 residuals was silently corrupt**, which is precisely the
configuration MLX ships by default.

Fixed: `DKV_MAX_RESIDUAL_SHARED = 128`, arrays sized to it, reads clamped to it.
Suite 21/21 in both residual conventions.

### 2. Pool sizing divided the budget by a per-slot cost that excluded residuals

`dynamic_max_blocks = pool_budget_bytes // bytes_per_block`, where
`bytes_per_block` deliberately EXCLUDES the residual arrays — the reporting code
right below it adds them back to print the real figure. So the pool allocated
more slots than the budget covers, and the error scales with `max_residual`:

    kv_heads=2, head_dim=256 -> res_bytes = max_res * 2052
      max_res  64 : true 238 KB/slot vs 110 KB assumed -> 2.2x over-allocated
      max_res 128 : true 366 KB/slot vs 110 KB assumed -> 3.3x over-allocated

Raising max_residual to MLX's 128 therefore did not merely cost 2x residual
memory — it widened this sizing error from 2.2x to 3.3x. Measured effect of the
fix: `max_blocks` 27648 -> 8301, worst-case pool 6.7 GB -> 3.1 GB.

MLX has no equivalent error: it sizes its pool as a flat block count
(`DKV_MAX_BLOCKS`, default 256) and never derives it from a byte budget.

### 3. Dense workspace over-allocated 2.24x

`(ceil(recency/block)+3) * (2*block+1)` = 9747 rows at recency 4096 vs MLX's
`recency + block_size` = 4352 — it multiplied a per-block worst case by a padded
block COUNT, double counting (the window holds ~recency tokens in TOTAL). ~265 MB
of extra K+V workspace across 24 layers. Now `recency + 2*block_size`.

### Also: timed eviction is now opt-in

`DKV_PAGER_BG_EVICT=1`, default off. It was redundant — `maybe_evict()` already
runs synchronously after every ingest and the loop called the same method on a
2.0 s timer, adding no capability, only nondeterministic timing. MLX has no timed
eviction.

### Where it stands

Per-layer residual cos vs MLX, mean over full_attention layers:

| | mean | L3 | L7 | L11 |
|---|---|---|---|---|
| original defaults | 0.697 | 0.778 | 0.763 | 0.486 |
| MLX config + these fixes | **0.795** | **0.975** | 0.870 | 0.834 |

Dense window re-verified at recency 4096: packed from 0, **4065/4065 rows exact**,
needle present at cos=1.0000, RoPE aligned. (It keeps block 0 dense alongside the
recent window — the "never drop block 0" protection — so the window is
non-contiguous in absolute position but correctly rotated.)

**The needle is still not retrieved and output is still unstable at this
configuration.** So at least one more CUDA-side limit is being reached. The
pattern from all three bugs above is the same and worth following: a hardcoded
constant sized for the OLD defaults that nothing validates against the new ones.
Audit remaining fixed-size buffers and derived counts against
`max_residual=128` / `recency=4096` — in the kernel (`weights_shared[256]`,
`scores_anc_cached[32]`, `q_proj_cached[32*48]`, `red_proj_temp[64*48]`) and in
the host-side block/slot bookkeeping.

---

## §9v — Background threads eliminated as a suspect; instability at the MLX config is something else (2026-07-28)

Following §9u's pattern (constants sized for the old defaults), two more
background mutators were made opt-in, on the same reasoning as the eviction
timer — each changes block RESIDENCY on its own thread, and MLX has no equivalent:

- `DKV_PAGER_BG_PREFETCH=1` (default off) — `_bg_prefetch_loop` calls
  `_reload_block`, i.e. CPU→GPU residency changes off-thread. A miss is not a
  correctness failure: `_reload_block` is also reachable synchronously on access,
  so prefetch only ever hides latency.
- `DKV_BLOCK_PREFETCH=1` (default off) — same for `BlockPrefetchEngine`.

**Result: this did NOT restore determinism at the MLX config**, and neither does
`DKV_DETERMINISTIC=1` any longer. With every background mutator off and the rSVD
sketch seeded (§9o), the same prompt still produces `Z` / `ZEBROF` /
`ZEBRA FOCHE MORNING THEODMET committee weathering` across runs.

**So the remaining nondeterminism at recency 4096 / max_residual 128 is NOT
thread timing.** That eliminates the entire class.

### Correction

§9u recorded that `DKV_DETERMINISTIC=1` pinned the output at this config. That was
n=2 and it does not reproduce — the same n=2 trap as the retracted determinism
claim in §9n. Two matching runs are not evidence of determinism when the
distribution is this wide; use >=3 and treat a match as weak evidence.

### The kernel is NOT the problem at these settings

Added `17_production_config` to the isolated suite — the ACTUAL shipped config
(`max_residual=128`, `S_max=257`, `L_dense=4065`, D=256, rank 32 / pool_rank 48,
K_active=16, facts+residuals+dense rope). **22/22 in both residual conventions.**
So the decode kernel computes the right answer at production scale; whatever is
unstable is host-side.

This case exists specifically to pin the shipped configuration: every bug in this
investigation has been a constant sized for the OLD defaults that nothing
validated against the new ones.

### Net position after §9u + §9v

| | mean full_attn cos vs MLX | L3 | L7 | L11 |
|---|---|---|---|---|
| original defaults | 0.697 | 0.778 | 0.763 | 0.486 |
| MLX config + all fixes | 0.795 | **0.975** | 0.870 | 0.834 |

Verified correct at the MLX config: the decode kernel (22/22), the dense window
(4065/4065 rows exact, needle at cos=1.0000, RoPE aligned, packed from 0), pool
slot allocation (no cross-layer aliasing), and compressed storage (§9p).

Still failing: the needle is not retrieved, and output varies run to run.

### Next

The suspect list is now short, because the alternatives are individually
verified. Host-side, per decode step, at this config: the routed block set, the
anchor cos/sin slices (`decode_cos_sliced`/`decode_sin_sliced`, cached on
`routing_version`), and the residual/fact tensors handed to the kernel. Capture
every argument passed to `fused_decode_attention_combined` on two runs of the
same prompt and diff them — the kernel is proven correct, so if its OUTPUT
differs, its INPUTS differ, and that diff names the source directly. That is a
much narrower question than anything asked earlier in this investigation.

---

## §9w — Nondeterminism localised to DKV's DENSE window; MPS and the pool exonerated (2026-07-28)

Ran the argument diff proposed at the end of §9v
(`scratchpad/probe_kernel_args.py`): checksum every argument handed to
`fused_decode_attention_combined`, twice, and diff. Pool-wide tensors are hashed
through `slot_indices` so the checksum covers exactly what the kernel reads.
hashlib, not builtin `hash()` — that one is salted per process and has already
produced one wrong conclusion here.

### Result

**At the first decode step, every argument is bit-identical.** Across 30 decode
steps, the ONLY arguments that ever differ are:

```
dense_k   (1, 2, 4224, 256) float16   <- differs
dense_v   (1, 2, 4224, 256) float16   <- differs
q         (8, 256)          float16   <- differs (follows from the above)
```

Bit-identical in every call: `pool_U`, `pool_U_scale`, `pool_V_K`, `pool_V_V`,
`pool_anchors_K`, `pool_anchors_V`, `pool_seq_lens`, `pool_scales`,
`slot_indices`, `cos_anc`, `sin_anc`, `cos_dense`, `sin_dense`, all four
`res_*`, all three `fact_*`, and every scalar.

**The compressed side is fully reproducible. The dense window is not.**

### MPS is NOT the source

`scratchpad/mps_determinism.py` runs the model with DKV disengaged
(`DKV_ENGAGE_THRESHOLD=999999`) and hashes the final logits:

```
[MPS] logits sha1: 704c53956c6b6358  argmax: 57  (x3, identical)
```

The model is bit-reproducible on this platform. So the nondeterminism is DKV's,
not the framework's — it cannot be waved away as GPU noise.

(The pool staying identical while raw dense K varies is consistent: the pool
stores U as int8, which quantizes tiny fp differences away, whereas the dense
window stores raw fp16 K and preserves them.)

### Narrowed further

`probe_dense_window.py` hashes the assembled workspace and gets identical results
across 3 runs — but that probe sets `DKV_DETERMINISTIC=1` internally, while the
arg diff did not. Since every background mutator is now opt-in and off by
default, and compression is already synchronous on macOS at the `mid` preset,
`DKV_DETERMINISTIC` should be a no-op at this point. It is not, which means one
of the things it disables still has an effect that has not been accounted for.

Also notable: at the first decode step layers 3 and 7 had identical `dense_k`
while **layer 11 differed** — the divergence is per-layer, not global, so it is
in per-layer dense state rather than anything session-wide.

### Next

Two concrete steps, in order:

1. Run the arg diff WITH `DKV_DETERMINISTIC=1` and again WITHOUT, same build.
   If it is reproducible only with the flag, bisect the flag: it now gates only
   `_async` (already False on macOS/mid), the pager's two loops, and
   BlockPrefetchEngine — all nominally off. Whichever still matters is doing
   something not captured by its own opt-in switch.
2. Hash `blk.active_k` / `blk.anchor_kv` per dense block immediately after
   prefill, across two runs. The workspace is assembled FROM those; if they
   already differ, the problem is upstream in ingest/streaming rather than in
   `assemble_dense_window_kv`, and the layer-11-only divergence says to look at
   per-layer block state.

This is now a question about one tensor's provenance, which is as narrow as this
investigation has been.

---

## §9x — Nondeterminism pinned to ONE tensor: the tail block's `active_k` (2026-07-28)

`scratchpad/probe_dense_provenance.py` hashes, per dense block, the SOURCE
tensors (`anchor_kv`, `active_k`) immediately before `assemble_dense_window_kv`
runs and the resulting workspace immediately after. Two runs, diffed.

### Result — it is one tensor, on some layers only

Of 17 dense blocks per layer, exactly one differs, and only one of its two
tensors:

```
blk anchor=8192  state=ACCUMULATING  dirty=True
    anchor_kv = faf9eceb051e            <- IDENTICAL
    active_k  = f0b4aca0df3f
              vs a569cc1dd5a0           <- DIFFERS
```

anchor 8192 is the FINAL, still-ACCUMULATING block — the 25-token tail
(8217 − 8192 = 25). Every completed block matches. And it is layer-selective:

| layer | tail `active_k` |
|---|---|
| 3 | identical |
| 7 | identical |
| **11, 15, 19, 23** | **differ** |

which lines up exactly with the kernel-argument diff (§9w): calls 0-1 identical,
calls 2-5 differ, at the same decode step.

### What that implies

`assemble_dense_window_kv` is exonerated — it is fed different data, it does not
create it. The workspace difference is downstream of `active_k`.

Layers 3 and 7 having a bit-identical tail means their INGESTED K is
deterministic. Layer 11's input hidden states are produced by layer 7's attention
output (layers 8-10 are linear-attention, and the same class of layer is provably
deterministic here — layers 0-2 are what make layer 3's K reproducible). So the
first nondeterministic step is **the attention OUTPUT of an early full_attention
layer during PREFILL** — layer 7 is the prime suspect, layer 3 the fallback —
for the tokens of the final partial chunk specifically.

That the completed blocks are all identical while only the in-progress tail
differs points at the last, partial prefill chunk rather than at steady-state
prefill.

### Next

Hash each full_attention layer's ATTENTION OUTPUT during prefill, per chunk,
across two runs; find the first (layer, chunk) that differs. Given the tail-only
signature, look hardest at how the final partial chunk is handled — sparse
prefill (`DKV_SPARSE_PREFILL`, default ON) routes per layer and is the obvious
candidate for a path that behaves differently on a short trailing chunk.

Everything else in the decode path is now verified reproducible: all pool
tensors, slot_indices, every cos/sin table, all residual and fact tensors, the
completed dense blocks, and the kernel itself (22/22).

---

## §9y — First divergence isolated to ONE forward: layer 7, first decode step (2026-07-28)

`scratchpad/probe_prefill_chunks.py` hashes every full_attention layer's
`self_attn` INPUT and OUTPUT on every forward, tagged with layer and q_len. Two
runs, diffed.

### Result

```
 100 layer=19 q_len=  25  in=3d425cb04229 out=ee99d2de73f4   identical
 101 layer=23 q_len=  25  in=ddb1fb2b5307 out=9384654d7f80   identical
 102 layer= 3 q_len=   1  in=80f106e63373 out=6c3da063a00a   identical
 103 layer= 7 q_len=   1  in=264aa9fd5f71 out=8c36b04982da   <-- run A
 103 layer= 7 q_len=   1  in=264aa9fd5f71 out=aa3975b0537b   <-- run B
                          ^^^ INPUT IDENTICAL   ^^^ OUTPUT DIFFERS
```

- **Every prefill forward is identical** (records 0-101, q_len 25..chunk size),
  on every full_attention layer. Prefill is fully reproducible — which retires
  §9x's hypothesis that the final partial prefill chunk was responsible.
- **Layer 3 at decode is identical** (record 102).
- **Layer 7 at the FIRST DECODE STEP is the first divergence**, with a
  bit-identical input hidden state. Records 104+ all cascade from it.

Note this also re-explains §9x: the tail block's `active_k` differed on layers
11+ but not 3/7 because that tensor is written from each layer's decode-step K,
and layer 7's output is what first goes wrong.

### The kernel is bitwise deterministic — verified

`/tmp/kern_det.py` invokes `decode_attention_metal` six times on identical inputs
at the shipped production config (max_residual=128, S_max=257, L_dense=4065):
**all six outputs hash identically**. The unit suite compares against a float64
reference with tolerance and would not have caught run-to-run variation, so this
was worth checking separately.

### The contradiction, which is the finding

At layer 7, first decode step:

| | status |
|---|---|
| input hidden states | bit-identical (§9y) |
| every kernel argument | bit-identical (§9w) |
| the kernel itself | bitwise deterministic (above) |
| model weights | identical |
| **`self_attn` output** | **DIFFERS** |

A deterministic function of identical inputs cannot produce different output. So
the decode path at layer 7 is reading state NOT captured by the kernel arguments,
or taking a DIFFERENT BRANCH between the two runs. `dkv_forward` has several
(combined kernel / separate dense-SDPA + fused_decode_mps merged by LSE / bypass),
selected on `has_dense`, `has_comp`, `block_indices.numel()`, engage thresholds
and cached routing state.

Both runs recorded exactly 24 combined-kernel calls, so the branch counts match at
that granularity — but that does not rule out a different branch being taken on a
call the arg probe never saw, nor post-kernel state (facts store, CAD, LSE merge)
being applied differently.

### Next — this is now a single forward to instrument

Log, for layer 7 at the first decode step only: which branch `dkv_forward` takes,
every predicate that selects it (`has_dense`, `has_comp`,
`block_indices.numel()`, `dense_len`, engage/bypass thresholds,
`routing_version`), and any post-kernel transform applied to `attn_out_b`.
Diff across two runs. One layer, one step, a handful of scalars — the divergence
is provably inside that block of code.

---

## §9z — Divergence is POST-kernel at layer 7, and it is INTERMITTENT (2026-07-28)

Extended the argument probe to also hash the kernel's RETURN value.

### Bisection result

At layer 7, first decode step, across two runs:

| | |
|---|---|
| `self_attn` INPUT hidden states | bit-identical |
| every kernel ARGUMENT | bit-identical |
| **kernel RETURN** | **bit-identical** (`0071cfa5fd3f` both runs) |
| kernel re-invoked 6x on fixed input | all 6 hashes equal (bitwise deterministic) |
| `self_attn` OUTPUT | **DIFFERS** |

So the divergence is strictly **between the kernel returning and the module
returning** — a short stretch of code. `attn_out_b = out_val.unsqueeze(0).unsqueeze(2)`
is a pure reshape, so if that branch ran, `o_proj`'s input must match. There are
EIGHT distinct `attn_out_b` assignments in `dkv_forward` (lines 1846, 1990, 2060,
2066, 2163, 2296, 2544, 2568); a different branch being taken for layer 7 between
runs would explain the whole picture.

### The o_proj bisection was INCONCLUSIVE — because the fault is intermittent

`scratchpad/probe_oproj.py` hooks `o_proj` (which every branch feeds) plus the
module output. That run-pair came out **identical** — the divergence simply did
not occur in either run.

Layer 7's decode output across the runs observed so far:

    8c36b04982da     aa3975b0537b     470d9ae3979d

Three distinct values. **The failure is intermittent, so a 2-run diff can miss
it.** This is the same n=2 trap that produced two earlier retracted claims in
this investigation (§9n, §9u). Any probe for it must run >=3 times and compare
all of them, and a single matching pair must NOT be read as "fixed".

### Next

Re-run `probe_oproj.py` 4-6 times, keep every output, and compare the whole set —
the question it answers is unchanged and still decisive:

- `o_proj` INPUT already differs -> the divergence is in `attn_out_b` assembly,
  i.e. a different branch was taken; log which of the eight assignments ran.
- `o_proj` INPUT identical, module OUTPUT differs -> the divergence is in
  `o_proj`/the gate, which given identical weights would mean a mutable-state
  read there.

Everything else is now verified reproducible or deterministic: all prefill
forwards, layer 3 at decode, every pool tensor, every kernel argument, the kernel
return, the kernel itself, and the model with DKV disengaged.

---

## §10a — Confirmed: a BRANCH difference in `attn_out_b` assembly at layer 7 (2026-07-28)

Ran the o_proj bisection repeatedly, per §9z's own rule (never diff only two runs
when the fault is intermittent). Layer 7, first decode step:

| run | `o_proj` INPUT | `self_attn` OUTPUT |
|---|---|---|
| r1 | `1750e3ab660d` | `8c36b04982da` |
| r2 | `5843ad9f7c32` | `aa3975b0537b` |
| a  | `28eb6d663e36` | `470d9ae3979d` |
| b  | `28eb6d663e36` | `470d9ae3979d` |

**`o_proj`'s INPUT already differs** — three distinct values across four runs, and
the output follows deterministically from it (a and b share both). So `o_proj` and
the gate are exonerated: the divergence is upstream, in `attn_out_b` assembly.

### The conclusion this forces

Established, each measured:

- layer 7's input hidden states are bit-identical across runs (§9y)
- every argument to the decode kernel is bit-identical (§9w)
- the kernel's RETURN is bit-identical (§9z, `0071cfa5fd3f`)
- the kernel is bitwise deterministic on fixed input (§9z, 6 invocations)

If the kernel returns the same value but the assembled attention output differs,
then in at least some runs **layer 7's `attn_out_b` did not come from that kernel
call**. `dkv_forward` has EIGHT `attn_out_b` assignments (lines 1846, 1990, 2060,
2066, 2163, 2296, 2544, 2568) and is selecting a different one between runs.

That also explains why the arg probe looked clean: it only ever saw the calls that
DID reach `fused_decode_attention_combined`.

### Cross-runtime check: the dense maths is NOT the problem

`scratchpad/xruntime_dense.py` feeds IDENTICAL values to MLX's
`_dense_only_attention_static` and to `decode_attention_metal` (zero compressed
slots, dense RoPE off so both consume pre-rotated keys), against a float64
reference:

| case | MLX vs ref64 | CUDA vs ref64 | MLX vs CUDA |
|---|---|---|---|
| H=4/1 D=16 L=8 | 0.99999986 | 0.99999998 | 0.99999985 |
| GQA 8/2 D=32 L=16 | 0.99999976 | 0.99999998 | 0.99999971 |
| 8/2 D=256 L=64 | 0.99999971 | 0.99999998 | 0.99999970 |
| 8/2 D=256 L=993 | 0.99999966 | 0.99999998 | 0.99999963 |
| 8/2 D=256 **L=4065** | 0.99999951 | 0.99999998 | **0.99999950** |

The two implementations agree to ~1e-7 at every size including production scale,
and the Metal kernel is marginally CLOSER to the float64 reference than MLX is.
The dense attention maths is equivalent — the divergence is in control flow around
it, not in the arithmetic.

### Next — one print statement

Log which of the eight `attn_out_b` assignments executes for layer 7 at the first
decode step, plus the predicates that select it (`has_dense`, `has_comp`,
`block_indices.numel()`, `dense_len`, the engage/bypass thresholds,
`routing_version`, `_triton_batch_queue` membership). Run >=3 times. The branch
that varies is the bug.

---

## §10b — Branch hypothesis REFUTED; a claim retracted; sync is the live lead (2026-07-28)

### Intermittency, properly sampled

Layer 7's `o_proj` input across SEVEN runs: **4 distinct values**
(`28eb6d663e36` x3, `1750e3ab660d` x2, `5613c6f5f082`, `5843ad9f7c32`). The
`self_attn` output follows deterministically from it. So o_proj and the gate are
exonerated; the variation is upstream.

### The branch hypothesis is WRONG

`scratchpad/probe_branch.py` wraps every function a branch calls and tags each
call with the executing layer (via a self_attn pre-hook). Three runs:

```
layer=3  fused_decode_attention_combined
layer=7  fused_decode_attention_combined
layer=11 fused_decode_attention_combined
layer=15 fused_decode_attention_combined
layer=19 fused_decode_attention_combined
layer=23 fused_decode_attention_combined      <- ALL THREE RUNS IDENTICAL
```

Every full_attention layer takes the SAME branch (line 1990,
`attn_out_b = out_val.unsqueeze(0).unsqueeze(2)`), exactly once, every run.
**§10a's conclusion that a different `attn_out_b` assignment was being selected is
refuted.**

### RETRACTION

§9z/§10a asserted the kernel's RETURN is bit-identical across runs. That was an
**n=2** comparison on a fault already known to be intermittent — the same error
this document has now flagged three times (§9n, §9u, §9z). It does not support the
conclusion drawn from it.

Worse, that probe hashes the return via `.cpu()`, which **forces a
synchronisation**. A probe that syncs cannot detect a missing-synchronisation
race: it would both hide the symptom and, by completing the work, prevent it.
Treat "kernel return is stable" as UNVERIFIED.

### Live lead: missing synchronisation after the kernel

Everything is consistent with the kernel's output being consumed before the GPU
has finished writing it: identical inputs, identical branch, deterministic in
isolation (where the harness's `.cpu()` syncs), variable in production.

`scratchpad/probe_sync.py` forces `torch.mps.synchronize()` after every kernel
call and hashes layer 7's `o_proj` input:

| | distinct values |
|---|---|
| sync OFF | 3 distinct in 3 runs |
| sync ON | 2 distinct in 4 runs (`1750e3ab660d` x3, `5613c6f5f082` x1) |

Sync concentrates the distribution but does NOT eliminate the variation. That is
**suggestive, not conclusive** — the sample is small and the difference could be
chance. It is explicitly NOT a fix.

### Next

1. Repeat the sync A/B at n>=10 per arm before drawing any conclusion. If sync ON
   is genuinely tighter but still varies, there is a second write not covered by
   that barrier.
2. Read `metal_runtime.mm`'s command-buffer handling directly: it obtains the
   ACTIVE PyTorch MPS command buffer, so ordering against subsequent torch ops
   should hold — verify it actually encodes onto that buffer and does not commit
   or dispatch on a private queue, and that the output tensor's storage is not
   read before the encoder ends.
3. Check whether the pool tensors the kernel reads are written by torch ops on the
   same queue earlier in the step; an unsynchronised WRITE upstream would produce
   exactly this signature too, and the argument probe could not see it (it hashes
   through `.cpu()`, which syncs).

### Cross-runtime result worth keeping

`scratchpad/xruntime_dense.py` (MLX `_dense_only_attention_static` vs
`decode_attention_metal`, identical inputs, zero compressed slots, no RoPE):
agreement **cos >= 0.9999995** at every size through L=4065, with the Metal kernel
marginally closer to a float64 reference than MLX. The dense arithmetic is
equivalent in the two runtimes. Reusable harness for any MLX-vs-CUDA function
pair; the compressed path is the obvious next candidate.

---

## §10c — A real lifetime bug found and fixed. It did NOT fix the nondeterminism. (2026-07-28)

### The bug (real, fixed, worth keeping)

`decode_attention_metal` enqueues the kernel and deliberately does not
commit/wait, to stay pipeline-aligned with PyTorch. But every input was prepared
as a local inside the `@autoreleasepool` block:

```objc
auto Q_c = Q.is_contiguous() ? Q : Q.contiguous();   // allocates when strided
...
[encoder dispatchThreadgroups:...]; [encoder endEncoding];
// no commit, no wait
}   // <- Q_c and all other _c locals destroyed HERE
```

At the closing brace every `.contiguous()` copy is freed while the kernel has only
been ENQUEUED. The block returns to PyTorch's MPS caching allocator, which can
hand it to a later op that overwrites it before our kernel runs.

Not hypothetical: `Q` at the decode call site is `query_states[b_idx, :, 0, :]`, a
strided slice, so it ALWAYS takes the copy path. The isolated tests never expose
it — they call `.cpu()` immediately and nothing allocates in between, so the block
is never recycled.

**Fixed** by retaining all inputs (and the outputs) on the command buffer's
`addCompletedHandler`, so they stay referenced until the GPU signals completion.
No commit added, pipelining unchanged. Suite: **22/22 in both residual modes**.

This is a genuine latent hazard — handing the GPU a pointer and then freeing it is
indefensible regardless of whether it is the bug being hunted — so the fix stays.

### It did NOT fix the symptom

Layer 7's `o_proj` input, after the fix, default path:

| | distinct values |
|---|---|
| before fix | 4 distinct in 7 runs |
| **after fix** | **3 distinct in 5 runs** (`1750e3ab660d` x3, `28eb6d663e36`, `5843ad9f7c32`) |

No meaningful change. **The lifetime bug is not the cause of this
nondeterminism.** Recording that plainly rather than claiming the fix worked.

### Hypotheses now eliminated

- MPS/platform nondeterminism (model with DKV disengaged is bit-reproducible)
- the decode kernel's arithmetic (bitwise deterministic in isolation, 22/22, and
  MLX-vs-CUDA agreement at 1e-7 on identical dense inputs)
- a branch difference in `attn_out_b` assembly (3 runs, identical traces)
- `o_proj` / the gate (output follows deterministically from their input)
- background threads (all opt-in and off; instability persists)
- the unseeded rSVD sketch (fixed in §9o; was a real and separate cause)
- input-tensor lifetime (this section)

### What is left

The remaining untested item from §10b is the one to run next: **an
unsynchronised upstream WRITE to the tensors the kernel reads.** Every probe so
far hashes through `.cpu()`, which synchronises — so a pool tensor still being
written by a queued torch op when the kernel reads it would look identical to
every probe and still feed the kernel different bytes on the GPU. That is
consistent with everything above, including why forcing a post-kernel sync
changed little (the race would be BEFORE the kernel, not after).

Concrete test: insert `torch.mps.synchronize()` immediately BEFORE the kernel call
(not after) and re-run the stability check.

**Done — it does not collapse either.** Layer 7's `o_proj` input:

| arm | distinct values |
|---|---|
| no sync (post lifetime fix) | 3 in 5 runs |
| sync AFTER the kernel | 2 in 4 runs |
| **sync BEFORE the kernel** | **2 in 5 runs** (`1750e3ab660d` x4, `28eb6d663e36`) |

Both barriers narrow the distribution slightly; neither removes it, and the
differences are within noise at these sample sizes. **A GPU-ordering race around
the kernel is not established, and is no longer the leading hypothesis.**

### Standing conclusion for whoever picks this up

The nondeterminism source is still unidentified, but the candidate list is now
short and everything below has been eliminated BY MEASUREMENT, not by argument:

| eliminated | how |
|---|---|
| platform / MPS | model with DKV disengaged is bit-reproducible (logits sha1 x3) |
| kernel arithmetic | bitwise deterministic in isolation; 22/22; MLX-vs-CUDA 1e-7 |
| `attn_out_b` branch selection | 3 runs, identical traces |
| `o_proj` / gate | output follows deterministically from their input |
| background threads | all opt-in and off; instability persists |
| unseeded rSVD sketch | fixed (§9o) — was real, and separate |
| input-tensor lifetime | fixed (§10c) — was real, and separate |
| GPU ordering around the kernel | sync before AND after; neither collapses it |

What has NOT been ruled out: state internal to `dkv_forward` that no probe has
hashed yet (the decode workspace caches keyed on `routing_version`, the
`dense_offsets` map, `_triton_batch_queue`, the tiered store's residency map), and
any Python-level container whose iteration order or contents vary. The next probe
should hash the ENTIRE `kv_manager.decode_workspace` subtree for the session
immediately before layer 7's decode call, across >=5 runs, and diff.

---

## §10d — DECISIVE: identical state produces different output, measured in ONE run (2026-07-29)

`scratchpad/probe_workspace.py` hashes, from a self_attn PRE-hook on layer 7 at
q_len==1, the ENTIRE observable DKV state — every `decode_workspace` key
(recursively: the `routing_version` caches, `dense_offsets`, `dense_block_sig`,
the cos/sin slices, the workspaces), the per-layer block list (anchor, state,
dirty, pool_idx, active length), the pool slots and their tensors, and the input
hidden states. It records the resulting `o_proj` input and `self_attn` output IN
THE SAME RUN, so state and output are paired rather than inferred across run sets.

Four runs:

```
run1  STATE=5ad60a95c27e  OPROJ_IN=28eb6d663e36  ATTN_OUT=470d9ae3979d
run2  STATE=5ad60a95c27e  OPROJ_IN=5843ad9f7c32  ATTN_OUT=aa3975b0537b
run3  STATE=5ad60a95c27e  OPROJ_IN=1750e3ab660d  ATTN_OUT=8c36b04982da
run4  STATE=5ad60a95c27e  OPROJ_IN=1750e3ab660d  ATTN_OUT=8c36b04982da
```

**Byte-identical state. Three distinct outputs.** A separate 5-run batch of the
state dump alone gave a single checksum as well, so the state is stable across at
least nine runs.

This closes out every state-based explanation. Combined with the earlier results
— identical kernel arguments, a bitwise-deterministic kernel in isolation, an
identical branch trace, o_proj/gate exonerated, background threads off, rSVD
seeded, input lifetime fixed, sync before AND after tried — the variation is not a
function of anything the host can observe.

### What that leaves, stated precisely

Every probe in this investigation hashes GPU tensors through `.cpu()`, which
**synchronises**. A probe therefore observes the SETTLED value. If a buffer the
kernel reads is still being written when the kernel runs, every hash still agrees
while the kernel sees something else. **The state hashes cannot exclude a GPU-side
race, and should not be read as doing so.**

So the two live candidates are:

1. **A GPU-side read-before-write** on a buffer the kernel consumes. The
   `torch.mps.synchronize()` A/B did not collapse the distribution, but
   `torch.mps.synchronize()` drains the PyTorch stream and may simply be the wrong
   barrier — the kernel encodes onto the ACTIVE PyTorch command buffer, and
   whether the producing op is on that same buffer at that moment is not
   established. Verify by committing and WAITING inside `decode_attention_metal`
   itself (`mps_stream->synchronize(SyncType::COMMIT_AND_WAIT)`) immediately after
   `endEncoding`. That is slow and not shippable, but it is decisive: if the
   distribution collapses to one value, it is an ordering bug and the fix is a
   correct barrier; if it does not, it is not.

2. **Uninitialised threadgroup memory** read before being written on some path.
   `thread_val` is `= { 0.0f }` and the final `out_buf` write is unconditional and
   covers every (tg_idx, d), so those are clean. The shared arrays
   (`weights_shared`, `dense_w_shared`, `q_proj_cached`, `red_proj_temp`,
   `scores_anc_cached`) each appear to have matching write/read index sets on
   inspection, but that is an eyeball argument, not a measurement — the isolated
   suite passes 22/22 and would not catch a path taken only in production.

Do (1) first: it is one line and it partitions the remaining space cleanly.

---

## §10e — GPU ordering and uninitialised threadgroup memory both ELIMINATED (2026-07-29)

Two decisive negatives, each from a direct intervention rather than an argument.

### 1. Not a GPU ordering race

`DKV_DEBUG_COMMIT_WAIT=1` (added to `metal_runtime.mm`, diagnostic only) commits
the command buffer and blocks until the GPU finishes, immediately after
`endEncoding` — removing all overlap between this kernel and anything around it.

```
OPROJ_IN=5843ad9f7c32   OPROJ_IN=28eb6d663e36
OPROJ_IN=1750e3ab660d   OPROJ_IN=5613c6f5f082
```

**4 runs, 4 distinct values.** Ordering is not the cause. (This also retires the
earlier "sync helps a bit" reading — that was small-sample noise.)

### 2. Not uninitialised threadgroup memory

Metal does not initialise threadgroup memory, and any path reading a shared
element it has not written this dispatch would consume another invocation's
leftovers — which matches the signature exactly. All 15 threadgroup arrays are now
explicitly zeroed at kernel entry behind a barrier (kept: it is cheap, and the
kernel's output must be a function of its inputs alone). Suite **22/22 both
modes**.

Result: **3 runs, 2 distinct values.** Still varies. Not the cause either.

### Elimination table — all by measurement

| candidate | how eliminated |
|---|---|
| platform / MPS | model with DKV disengaged: identical logits sha1 x3 |
| host state into the call | whole `decode_workspace` + blocks + pool + hidden states byte-identical, PAIRED with differing output in the SAME run (§10d) |
| kernel arithmetic | bitwise deterministic on fixed inputs, 6 invocations; 22/22 |
| MLX-vs-CUDA dense maths | agree to 1e-7 through L=4065 |
| `attn_out_b` branch selection | 3 runs, identical traces |
| `o_proj` / gate | output follows deterministically from their input |
| background threads | all opt-in and off; instability persists |
| unseeded rSVD sketch | fixed (§9o) — real, separate cause |
| input-tensor lifetime | fixed (§10c) — real, separate hazard |
| **GPU ordering** | **COMMIT_AND_WAIT: 4 distinct in 4 runs** |
| **uninitialised threadgroup memory** | **explicit zeroing: still 2 distinct in 3** |

### The one measurement never done properly

Every claim that the kernel receives identical ARGUMENTS and returns an identical
VALUE in production rests on **n=2** comparisons (§9w, §9z) — taken before the
fault was known to be intermittent, and retracted once already. They have never
been repeated at n>=5, and never with args and return PAIRED in the same run the
way §10d paired state and output.

Do exactly that next: extend `scratchpad/probe_kernel_args.py` to record the
RETURN alongside the arguments, run it >=5 times, and compare the whole set.

- **args identical, return varies** -> the kernel is nondeterministic IN SITU
  despite being deterministic in isolation. Given ordering and threadgroup
  initialisation are both excluded, look next at device-memory aliasing: whether
  two of the buffers handed to it can overlap, and whether `out`/`lse`
  (`torch::empty`) can alias a live input.
- **args vary** -> something upstream mutates a kernel input between the state
  hash and the dispatch, and the §10d state hash simply does not cover it.

That single result partitions everything that is left.

---

## §10f — PROVEN: the kernel is nondeterministic IN SITU on byte-identical arguments (2026-07-29)

The measurement §10e said had never been done properly, done properly: args and
return recorded in the SAME call, layer 7, first decode step, **n=5**.

```
ARGS=3ecd9a3fe8cf  RETURN=0071cfa5fd3f
ARGS=3ecd9a3fe8cf  RETURN=d2e75c099d49
ARGS=3ecd9a3fe8cf  RETURN=ab614b7c8728
ARGS=3ecd9a3fe8cf  RETURN=d2e75c099d49
ARGS=3ecd9a3fe8cf  RETURN=d2e75c099d49

distinct ARGS: 1        distinct RETURN: 3
```

(The args hash covers every tensor argument — pool tensors through
`slot_indices`, both dense buffers, all cos/sin tables, all residual and fact
tensors, every scalar.)

**Byte-identical inputs, three distinct outputs, in production** — while the very
same kernel, called six times on fixed inputs in isolation, returns one identical
hash every time. Both halves are now measured at n>=5. This supersedes the earlier
n=2 claims (§9w, §9z) that the return was stable; they were wrong.

Also completes the zero-init result: **2 distinct in 5 runs** (full tally).

### Retraction: the buffer-aliasing "finding" is a PROBE ARTIFACT

A first pass appeared to show `out OVERLAPS dense_v` / `dense_k` / `cos_dense`,
varying per run — which would have explained everything. **It is not real.**
Control:

```
a = torch.zeros(1000,256,fp16,mps)   -> [0xc5b759180, 0xc5b7d6180)
b = torch.zeros(1000,256,fp16,mps)   -> [0xc5b759500, 0xc5b7d6500)
a/b "overlap": True
```

Two freshly-allocated, definitely-distinct tensors "overlap": 512 KB allocations
whose reported pointers differ by 0x380. `data_ptr()` on MPS does not linearly
encode the sub-allocation, so range arithmetic over it is meaningless and the test
cannot measure aliasing at all. Aliasing is therefore **untested — neither
confirmed nor refuted.**

(Fourth probe defect in this investigation, after the rotation-basis probe, the
block-property fallback, and salted `hash()`. Every one produced a confident wrong
answer. Validate the instrument on a known case before trusting it.)

### State of the question

Established at n>=5, each by direct measurement:

| | |
|---|---|
| host state into the call | byte-identical, paired with differing output (§10d) |
| kernel ARGUMENTS | byte-identical (§10f) |
| kernel RETURN | **3 distinct values** (§10f) |
| kernel in isolation | bitwise deterministic, 6/6 (§9z) |
| GPU ordering | eliminated — COMMIT_AND_WAIT, 4 distinct in 4 (§10e) |
| threadgroup init | eliminated — explicit zeroing, 2 distinct in 5 (§10e) |

So the kernel computes different results from the same bytes, only in production,
and neither of the two mechanisms that normally explain that is responsible.

### Next: test aliasing PROPERLY, from C++

Python cannot see MPS buffer identity. Add a diagnostic to `metal_runtime.mm`
that, for every bound buffer, logs `getMTLBufferStorage(t)` (the `id<MTLBuffer>`)
together with the storage offset and byte length, and reports any pair whose
[offset, offset+len) ranges overlap WITHIN THE SAME MTLBuffer. That is the real
aliasing test. `out` and `lse` are `torch::empty` from the MPS caching allocator
and are the obvious suspects against the dense buffers.

If no aliasing: the remaining explanation is that some buffer bound to the kernel
is not what the host thinks it is — e.g. a tensor whose storage is mutated by a
queued op the host-side hash (which syncs) reads as settled. Bind-time byte
dumps from inside the encoder would be the way to see that.

---

## §10g — Aliasing and intra-threadgroup racing BOTH eliminated (2026-07-29)

### Buffer aliasing: refuted, properly this time

Added `DKV_DEBUG_ALIAS=1` to `metal_runtime.mm` — the real test, from C++: for
every bound buffer, compare `getMTLBufferStorage()` identity plus
[storage_offset, +nbytes), and report overlaps only WITHIN THE SAME MTLBuffer.

```
[DKV ALIAS] overlaps=0  out.buf=0x7de8e2bc0 off=0  denseK.buf=0x7de52d340 off=0
[DKV ALIAS] overlaps=0  out.buf=0x7de8e39c0 off=0  denseK.buf=0x7ded34700 off=0
... (every call, every run)
```

**overlaps=0 everywhere**, and every tensor gets its OWN MTLBuffer at offset 0 —
PyTorch's MPS allocator is not sub-allocating these at all, so aliasing between
bound buffers is impossible by construction. This is what the §10f Python attempt
should have measured and could not.

### Intra-threadgroup races: eliminated

`DKV_DEBUG_TG1=1` forces ONE thread per threadgroup. Every loop is
`for (i = tid; i < N; i += t_per_tg)` and every reduction strides by
`t_per_tg / 2`, so a single thread is functionally correct — it does all the work
and the reduction loops become no-ops. With one thread there is no
intra-threadgroup concurrency, so a missing `threadgroup_barrier` between a shared
write and a shared read cannot manifest.

```
b6bc38c1f78b, b6bc38c1f78b, 811047f46d8e, b6bc38c1f78b   -> 2 distinct in 4
```

(The value differs from the 64-thread runs, as expected — a different fp reduction
order — but it still VARIES.) The first two runs matched, which at n=2 would have
looked like a confirmed fix; it is not. **A missing barrier is not the cause.**

### Full elimination table

| candidate | verdict | evidence |
|---|---|---|
| platform / MPS | eliminated | DKV disengaged: identical logits sha1 x3 |
| host state into the call | eliminated | byte-identical, PAIRED with differing output, same run |
| kernel ARGUMENTS | eliminated | byte-identical, n=5, paired with 3 distinct returns |
| kernel in isolation | eliminated | bitwise deterministic 6/6 |
| MLX-vs-CUDA dense maths | eliminated | agree to 1e-7 through L=4065 |
| branch selection | eliminated | 3 runs, identical traces |
| o_proj / gate | eliminated | output follows deterministically from input |
| background threads | eliminated | all opt-in and off; persists |
| unseeded rSVD sketch | FIXED | real, separate cause (§9o) |
| input-tensor lifetime | FIXED | real, separate hazard (§10c) |
| GPU ordering | eliminated | COMMIT_AND_WAIT: 4 distinct in 4 |
| uninitialised threadgroup memory | eliminated | explicit zeroing: 2 distinct in 5 |
| buffer aliasing | eliminated | overlaps=0, separate MTLBuffers |
| intra-threadgroup races | eliminated | 1 thread/tg: 2 distinct in 4 |

A **single-threaded** GPU kernel, given byte-identical arguments, with ordering
barriers, zeroed shared memory and no buffer aliasing, still returns different
values. That is the shape of the remaining problem.

### The one hypothesis that survives

`out` and `lse` are `torch::empty` from the MPS caching allocator. That block may
be RECYCLED memory whose previous owner has a GPU write still in flight — freed on
the host, but not yet retired on the device. Our kernel writes the block; the
stale write lands afterwards and corrupts it. This is NOT aliasing between our
bound buffers (measured: none), and barriers around OUR dispatch do not help,
because the racing write belongs to an earlier op on a buffer that was freed and
handed back to us.

It fits every observation, including single-threaded nondeterminism.

**Test:** allocate `out`/`lse` ONCE per (n_q_heads, D) and reuse them across
calls — a persistent buffer that is never freed cannot be recycled memory. If the
output stabilises, that is the mechanism, and the production fix is either a
persistent output arena or forcing the allocator to hand back only retired blocks.

### Shipped state

All three diagnostics (`DKV_DEBUG_ALIAS`, `DKV_DEBUG_TG1`, `DKV_DEBUG_COMMIT_WAIT`)
default OFF. Threadgroup zero-init is KEPT (unconditional, cheap, correct). Suite
**22/22 in both residual conventions.**

---

## §10h — Recycled-output hypothesis eliminated; instrument validated (2026-07-29)

### Persistent output buffer: does not help

`DKV_DEBUG_PERSIST_OUT=1` allocates `out`/`lse` ONCE per (n_q_heads, D) and reuses
them, so they cannot be recycled allocator memory carrying another op's in-flight
write. The kernel fully overwrites both every call, so reuse is safe.

```
28eb6d663e36, 1750e3ab660d, 1750e3ab660d, 28eb6d663e36   -> 2 distinct in 4
```

**Eliminated.** That was the last hypothesis standing from §10g.

### The instrument is sound

Per this document's own rule (four probe defects so far, each producing a
confident wrong answer), the probe was validated before drawing further
conclusions: it records `n_calls=1` — exactly one q_len==1 call at layer 7 — so
the varying hashes are the same decode step across runs, not different steps.

Also checked, and correct: `S_max = U_c.size(1)` and `pool_rank = VK_c.size(1)`
are both derived from the bound tensors, and the pool allocates
`U = zeros(n_blocks, max_seq_len, rank)`, so the kernel's
`slot_id * S_max * pool_rank + t * pool_rank` stride matches the real layout. No
out-of-bounds device read from a stride mismatch.

### Where this ends

Everything nameable about the kernel invocation is now eliminated by measurement:

| | |
|---|---|
| platform / MPS | model with DKV off: identical logits x3 |
| host state | byte-identical, paired with differing output, same run |
| kernel arguments | byte-identical, n=5, paired with 3 distinct returns |
| kernel in isolation | bitwise deterministic 6/6 |
| branch selection | 3 runs, identical traces |
| background threads | all off; persists |
| GPU ordering | COMMIT_AND_WAIT: 4 distinct in 4 |
| uninitialised threadgroup memory | zeroed: 2 distinct in 5 |
| buffer aliasing | C++ MTLBuffer identity: overlaps=0, separate buffers |
| intra-threadgroup races | 1 thread/threadgroup: 2 distinct in 4 |
| recycled output allocation | persistent buffer: 2 distinct in 4 |
| stride / bounds derivation | S_max and pool_rank both from the tensors |

A single-threaded Metal kernel, given byte-identical arguments, with a persistent
output buffer, zeroed shared memory, no aliasing, and a full commit-and-wait
around the dispatch, still returns different values across runs.

**No hypothesis survives.** Something in this stack violates an assumption not yet
articulated, and the honest position is that the mechanism is unknown rather than
"probably X".

### The next probe should observe the GPU, not the host

Every measurement so far is host-side and reads memory through a synchronising
copy. The next step should make the KERNEL report what it actually sees: bind a
debug buffer and have thread 0 of a chosen threadgroup write out the raw bytes it
reads for a few fixed addresses (a slice of `Q`, of `dense_K`, of `U_pool`, and
its own `thread_val` before the final store), then compare those across runs
against the host's hash of the same addresses.

- GPU-visible bytes differ from host-visible bytes -> memory is being changed
  between bind and execute by something outside this code path.
- GPU-visible bytes match but the accumulator diverges -> the arithmetic itself
  is diverging on identical data, which would point at a compiler/runtime issue in
  the Metal toolchain rather than at this code.

That is the first measurement that can distinguish those two, and until it is run,
further host-side probing will keep returning "everything is identical".

---

## §10i — Function-by-function MLX-vs-CUDA sweep: real gaps found and closed (2026-07-29)

Rather than continue the nondeterminism hunt, swept the two runtimes for places
they differ or where CUDA is simply missing behaviour MLX has — small differences
compound across 24 layers and every prefill chunk.

Method: diff the full set of `DKV_*` tunables in each runtime (81 in MLX, 148 in
CUDA), then read the functions behind every name MLX has and CUDA does not.
42 names are MLX-only; most are debug (`DKV_DBG_*`) or known-dead experiments
(`DKV_LEGO_*`, `DKV_EDGE_*`, `DKV_FACTUAL_*`). Four were substantive.

### GAP 1 (FIXED): sparse prefill had no recency window and no min-context gate

`_sparse_prefill_filter_blocks` kept sinks + top-K and dropped everything else.
MLX's equivalent additionally:

| | MLX | CUDA (before) |
|---|---|---|
| `_MIN` min context | **2048** — fully dense below it | **absent**: sparsified from the first chunk |
| `_WINDOW` recency | **1024** tokens ALWAYS attended | **absent**: no recency guarantee at all |
| `_SINK_BLOCKS` | 1 | 1 (hardcoded) |
| `_KMIN` | 8 | 8 |
| `_FRAC` | 0.05 | 0.25 |

The missing recency window is the significant one: MLX guarantees a chunk always
sees its immediate left context, whatever routing decides. Without it, routing
could drop the blocks physically adjacent to the chunk being processed — at every
layer, for every chunk, and the damage carries forward through the whole prefill.
The missing min-context gate meant CUDA also paid routing cost and took routing
risk on short prompts where MLX stays dense.

Both ported (`DKV_SPARSE_PREFILL_MIN`, `DKV_SPARSE_PREFILL_WINDOW`, same defaults),
with `chunk_start` now threaded to both call sites. Kept order-sorted afterwards
because downstream builds positions from `anchor_idx` and assumes monotonic order.

`_FRAC` deliberately left at 0.25: it is LARGER than MLX's 0.05, so this side
attends strictly more blocks, which is the safe direction. Lowering it can only
remove context.

### GAP 2 (FIXED): three knobs are the same concept under different names

A config or benchmark written against MLX silently configured NOTHING here:

| concept | MLX name | CUDA name |
|---|---|---|
| residual budget/block | `DKV_MAX_RESIDUAL` | `DKV_MAX_RESIDUAL_TOKENS` |
| residual twin exclusion | `DKV_RESIDUAL_EXCLUDE_SVD` | `DKV_RESIDUAL_EXACT_KEYS` |
| rSVD sketch seed | `DKV_SVD_SEED` | `DKV_RSVD_SEED` |

All three now accept EITHER name on this side (primary wins when both are set).
Verified: `DKV_MAX_RESIDUAL=40` → 40; both set → primary; `DKV_RESIDUAL_EXCLUDE_SVD=0`
→ exact-keys off.

Note the third: MLX has had a seeded sketch all along (`DKV_SVD_SEED`), which is
why MLX was reproducible and this side was not until §9o.

### Still divergent, deliberately not changed

- `DKV_COMPRESSED_MIN_CTX` (MLX, 16384) gates MLX's adaptive compressed-decode
  mode. This side always uses the combined kernel, so there is no equivalent
  switch to gate; not a defect, but it means "compressed decode" means different
  things in the two runtimes.
- `DKV_DENSE_WINDOW_FACTOR` — MLX's fallback recency formula. This side ports the
  `DKV_ENGAGE_THRESHOLD` override (§9s) but still falls back to a flat 512 rather
  than `num_layers * head_dim * factor`.
- `DKV_RES_V_ONLY`, `DKV_INIT_BLOCKS`, `DKV_PREFILL_CHUNK`, `DKV_SPECULATIVE`,
  `DKV_SP_NO_POOL` — MLX-only, all default-off or MLX-internal.

### Regression caught and fixed during this work

Adding the `DKV_DEBUG_GPUREAD` diagnostic allocated its buffer INSIDE the encoder
scope, which makes PyTorch open a second encoder on the same command buffer:

```
failed assertion 'A command encoder is already encoding to this command buffer'
```

Every kernel test crashed. Fixed by allocating alongside `out`/`lse`, before any
encoder exists — which is precisely why those two are allocated where they are.
Worth knowing for anyone adding a buffer here: **no tensor allocation between
`computeCommandEncoder` and `endEncoding`.**

Suite after all of the above: **22/22 in both residual conventions.**

---

## §10j — More MLX divergences found and closed; needle still NOT retrieved (2026-07-29)

Continued the function-by-function sweep, focusing on hardcoded constants with no
MLX counterpart.

### GAP 3 (FIXED, the biggest): residual budget capped at 0.15*n, not max_residual

MLX (`compress_mlx_block_batched` caller):

```python
b_res = self.max_residual          # 128
if   val < 0.05: b_res = min(8,  b_res)
elif val < 0.15: b_res = min(16, b_res)
```

CUDA (both compress paths):

```python
n_max_residual = int(n * 0.15)     # = 38 at n=256  <-- CUDA-only invention
if   err < 0.05: n_max_residual = min(8,  n_max_residual)
elif err < 0.15: n_max_residual = min(16, n_max_residual)
```

MLX starts from the configured budget and only clamps DOWN for easy blocks. This
side started from `int(0.15 * n)`, so a HARD block — numbers, codes, precisely
the content residuals exist to preserve — could never receive more than **38**
exact tokens no matter how large `max_residual` was.

That silently defeated raising `max_residual` to MLX's 128 (§9u): only
force-exact blocks bypassed the cap, so ordinary factual blocks kept the old
ceiling. Both paths now take the pool's real budget (`compress_lowrank` gained a
`max_residual` parameter; the batched path reads `pool.max_residual_tokens`), with
the fraction retained only as a fallback for callers that pass nothing.

Measured on a synthetic hard block (256x1024, median rel err > 0.15):
**38 -> 128 residuals selected.**

### GAP 4 (FIXED): the router scored only the first 64 residual keys

`query_router.route_blocks_relevance` capped its residual scoring at a hardcoded
`min(R_all, 64)`. Invisible while the pool also held 64 — but with
`max_residual_tokens` now 128, HALF of every block's exact keys were invisible to
routing. MLX takes the max over ALL R with an `res_valid` -inf mask, which this
function already mirrors otherwise. Now defaults to `R_all`;
`DKV_ROUTE_RESIDUALS>0` still caps explicitly for cost control.

Third instance of the same pattern after the kernel's 64-wide residual scratch and
the pool's residual-excluding size math (§9u).

### GAP 5 (FIXED): recency fallback ignored model shape

With neither `DKV_RECENCY_WINDOW` nor `DKV_ENGAGE_THRESHOLD` set, this side used a
flat 512 while MLX derives
`max(512, round_up_512(num_layers * head_dim * DKV_DENSE_WINDOW_FACTOR))` — 1536
for Qwen3.5-2B. A 3x difference in default dense window. Ported; verified 1536.

### Checked, NOT divergent

- **Sparse prefill is default-ON in BOTH** (`!= "0"`, default "1"); nothing in
  `serving/cli.py` disables it. The only difference was the missing 2048 gate and
  1024 window, closed in §10i.
- **Router residual validity masking** — CUDA already masks invalid slots to
  -inf, matching MLX's `res_valid`.
- **The 8/16 error ladder and the 0.05/0.15 thresholds** are identical.
- `serving/cli.py` sets only `DKV_MPS_APPROXIMATE_ATTN`, `DKV_USE_TORCH_COMPILE`,
  `DKV_V_SCALE`; the `DKV_FAST` bundle is opt-in, not a default.

### Answer to "is the needle retrieved now?" — NO

With every fix above applied, 3 runs of the needle-at-start prompt MLX solves:

```
'...ZEBRA FOGG'   FOUND: False
'...ZEB'          FOUND: False
```

Still not retrieved, and still varying run to run. The §10h nondeterminism —
byte-identical kernel arguments producing different returns, with every named
mechanism eliminated — remains unresolved and is the reason results still move
between runs. **None of the parity fixes in §10i/§10j should be read as
addressing it; they are correctness fixes that stand on their own.**

Suite: **22/22 in both residual conventions** after all changes.

---

## §10k — GAP 6: the 0.08 error floor threw away the residual budget (2026-07-29)

MLX picks residuals by pure top-k on the capture score:

```python
top_k_indices = mx.argsort(capture_scores)[-n_res:][::-1]
```

No error threshold anywhere — the string "0.08" does not appear in
`mlx_dkv_wrapper.py` at all.

This side filtered the top-k afterwards, in BOTH compress paths:

```python
mask_K = (top_k_K.values > 0.08) & (error_K[top_k_K.indices] > 1e-4)
```

So a token the SVD got only MODERATELY wrong never received an exact residual —
even though that size of error is exactly what flips a digit. And it capped the
budget from the other end: raising `max_residual` grew the candidate set, then the
filter discarded the extra slots.

**Measured** on a well-reconstructed (low-rank + noise) block, budget 128:

| threshold | residuals kept |
|---|---|
| 0.08 (old) | **0** |
| 0.0 (MLX) | 8 |

On an ordinary prose block this side stored **no exact tokens at all**, where MLX
always spends its budget. Combined with GAP 3 (§10j), the two caps compounded:
the budget was first cut to 38, then whatever survived was filtered to nothing
unless the block reconstructed badly.

Both sites now share `_residual_error_threshold()`, default **0.0** = MLX.
`DKV_RESIDUAL_ERR_THRESHOLD` restores a floor.

### Running tally of MLX divergences closed in §10i–§10k

| # | gap | effect |
|---|---|---|
| 1 | sparse prefill: no recency window, no min-ctx gate | routing could drop a chunk's own left context, every layer |
| 2 | 3 knobs named differently | MLX-written configs silently did nothing |
| 3 | residual budget `int(0.15*n)`=38, not `max_residual` | hard blocks capped at 38 exact tokens regardless of config |
| 4 | router scored only first 64 residual keys | half of every block's exact keys invisible to routing |
| 5 | recency fallback flat 512 vs shape-derived 1536 | 3x smaller default dense window |
| 6 | 0.08 error floor on residual selection | ordinary blocks stored ZERO exact tokens |

Gaps 3, 4 and 6 all point the same way: **the residual mechanism was configured
for 128 exact tokens per block and was actually delivering between 0 and 38.**

### Needle: still NOT retrieved

`ZEB` on the first run with all six closed. Suite 22/22 both conventions. The
§10h nondeterminism is still unresolved, so run-to-run variation persists and no
single run should be read as a verdict.

---

## §10l — GPU-side observation: probe UNSOUND, readings WITHDRAWN (2026-07-29)

Ran the probe queued at the end of §10h: `DKV_DEBUG_GPUREAD=1` binds a debug
buffer (index 29) and has threadgroup 0 / thread 0 write the RAW BYTES it reads
for fixed addresses — a slice of `Q`, of `dense_K`, of `U_pool` — plus its own
accumulator, so the GPU's view can be compared against the host's.

Two runs, layer 7's decode call:

```
Q:      1.26855 -0.858398 -0.437744 -0.852051 0.405762 0.209473 0.879395 -0.103516
denseK: 1.91504  1.78809   3.0957    1.79492 -3.66602 0.760254 4.72656 -1.27246
Upool:  -52 3 30 9 -3 4 4 -9
acc:    0.0201657
```

**Byte-identical across both runs, including the accumulator.** So for that lane
the GPU reads exactly what the host bound and computes exactly the same result —
no evidence of memory changing between bind and execute.

### Why this is NOT the answer yet

1. **n=2.** By this document's own rule (four probe defects, several n=2 mistakes
   already retracted) two runs prove nothing about an intermittent fault.
2. **It only covers one lane.** `thread_val` is thread-PRIVATE and thread 0 owns
   only `d = 0, 64, 128, 192`; `thread_val[1..7]` read as zeros from thread 0,
   which is why `acc` shows a single non-zero value. The other 63 lanes were not
   observed at all.
3. **`DKV_DEBUG_GPUREAD` forces COMMIT_AND_WAIT** to read the buffer back, and
   §10e established that COMMIT_AND_WAIT alone does NOT stop the output varying
   (4 distinct in 4). So this probe measures a configuration already known to
   still be nondeterministic — a stable reading here is consistent with the fault
   living in the lanes it cannot see.

The probe was widened to snapshot `out_buf[0..7]` after the store (eight
DIFFERENT lanes, since thread `d` owns dim `d` for `d < t_per_tg`), but those runs
did not complete: with COMMIT_AND_WAIT serialising every dispatch AND blocks now
compressing a full 128 residuals each (§10j/§10k), a single run takes long enough
that the sample could not be gathered.

### Honest status

**The nondeterminism is NOT fixed.** The instruction was to fix it first; I did
not. What this round adds is one narrow, weak datapoint — thread 0's inputs and
result are stable — and a widened probe that is built and ready but unmeasured.

Next session should run the widened probe (>=5 runs) with the residual budget
temporarily lowered (`DKV_MAX_RESIDUAL_TOKENS=64`) purely to make the runs
tractable, then compare `out_buf[0..7]`:

- all eight lanes stable across runs -> the divergence is in the remaining 248
  dims or in a different threadgroup; widen again rather than theorise.
- some lane varies while its inputs are identical -> that lane's arithmetic is
  the target, and the per-lane difference names which code path.


---

## §10m — the §10l readings are WITHDRAWN: the probe was crashing (2026-07-29)

`DKV_DEBUG_GPUREAD=1` runs exit **139 (SIGSEGV)** / **134 (abort)** and emit only
ONE of the six expected per-layer lines. The abort message:

```
-[AGXG15GFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:...]
  failed assertion `A command encoder is already encoding to this command buffer'
```

That is this document's own documented trap (never create a tensor inside the
encoder region). Two attempts to fix it:

1. Hoisted the `dbg` allocation above `computeCommandEncoder` — still aborted,
   because `torch::zeros` on MPS enqueues a FILL, which opens its own encoder.
2. Moved it up beside `out`/`lse` — still SIGSEGVs, now on the READBACK: the
   print does `synchronize(COMMIT_AND_WAIT)` and then `dbg.to(kCPU)`, which asks
   the stream to encode a blit onto a buffer that was just committed.

**So every number reported in §10l came from a process that crashed immediately
after printing.** They are withdrawn: an aborting run is not evidence that the
GPU's reads or its accumulator are stable. The "byte-identical across two runs"
claim in §10l should not be carried forward.

(That is the fifth probe defect in this investigation, after the rotation basis,
the block-property fallback, salted `hash()`, and MPS `data_ptr()` ranges. The
pattern is consistent enough to be worth stating as a rule: **a probe that
touches the Metal command buffer must be proven to exit 0 before any reading from
it is quoted.**)

### Default path verified unaffected

The diagnostic is opt-in and OFF by default. With no diagnostics set:

- isolated kernel suite: **22/22 in both residual conventions**
- a full end-to-end decode run: **exit 0**

So the scaffolding has not destabilised the shipped path.

### To make this probe usable

Do not read the debug buffer inside `decode_attention_metal`. Bind it, let the
kernel write it, and return it as a THIRD output tensor from the pybind function
— then the caller reads it in Python, after the stream has settled, with no
encoding from inside the encoder region and no readback against a committed
buffer. That also removes the need for COMMIT_AND_WAIT, so the probe would
observe the DEFAULT execution mode rather than a serialised one (which §10e
showed is itself still nondeterministic, and was a confound in §10l anyway).

### Status

**The nondeterminism is not fixed, and this round produced no valid measurement of
it.** What it did produce: the §10l readings are withdrawn rather than left
standing as a false lead, and the crash that invalidated them is diagnosed.

---

## §10n — Preset smoke test: all three RUN, none is healthy (2026-07-29)

Asked whether normal DKV still works across modes/presets after the default
changes (mid `max_residual` 64->128, budget now from the pool, 0.08 floor removed,
MLX recency semantics). Config resolution first:

| preset | max_residual | decode_cache | prefill_chunk | kv_quant |
|---|---|---|---|---|
| low | 40 | False | 256 | q4_0 |
| mid | **128** | True | 512 | q8_0 |
| high | 128 | True | 2048 | f16 |

Correct — only `mid` moved (64 -> 128, matching MLX). Note the ladder is now
40/128/128, so `mid` and `high` share a residual budget; that is what MLX does
(flat 128) but it does collapse this side's three-tier design into two.

End-to-end smoke run (2105-token prompt, `DKV_ENGAGE_THRESHOLD=1024` so
compression genuinely engages, 12 new tokens):

```
[PRESET low ] tokens=2105 time=179s ok=True needle=False tail='ZEEV AND'
[PRESET mid ] tokens=2105 time=640s ok=True needle=False tail='ZEBL000000000'
[PRESET high] tokens=2105 time= 94s ok=True needle=False tail='ZEB. B. A.!!!!!'
```

**Runs: yes — no crashes, no exceptions, all three complete.**
**Healthy: no.** Two problems, both new information:

1. **The corruption reproduces at 2k tokens.** Every preset emits a degenerate
   tail (`ZEBL000000000`, `ZEB. B. A.!!!!!`) rather than the code. Until now this
   was only ever observed on the 8.2k prompt. A 2k repro is far cheaper to
   iterate on — future work on the §10h nondeterminism should use this instead of
   the 8k prompt, and should first confirm the 2k case is also run-to-run
   unstable (not yet checked).

2. **`mid` is anomalously slow: 640s vs `high`'s 94s** on the identical prompt —
   the heavier preset is nearly 7x FASTER. `mid` and `high` now differ only in
   `prefill_chunk_size` (512 vs 2048), `kv_quant` (q8_0 vs f16) and
   `decode_cache_max_tokens`. A 7x gap from those is not explainable by chunk
   count alone and suggests something pathological in the mid path -- most likely
   the q8_0 quantised prefill cache, which `high` (f16) skips entirely. Not
   investigated.

Neither is a regression introduced by this session's changes as far as the smoke
test can tell (no before/after baseline was captured at this size), but both are
now on the record and cheap to chase.

---

## §10o — Staged KV check on a CHEAP 2k repro: prepare/store/retrieve all CLEAN (2026-07-29)

### A much cheaper reproduction

2105-token prompt, `DKV_ENGAGE_THRESHOLD=1024`, needle at the start:

```
MLX  low/mid/high : needle=True  'ZEBRA-4471-QUARTZ'   (all three)
CUDA low/mid/high : needle=False 'ZEEV AND' / 'ZEBL000000000' / 'ZEB. B. A.!!!!!'
```

CUDA `generate` is ~20 s here versus minutes at 8.2k. **Use this prompt for all
further work on this defect.** (Earlier preset timings in §10n included MODEL LOAD
because the harness started its timer before constructing the wrapper — corrected:
CUDA high = load 44 s + gen 20 s.)

### Stage 1 — KV PREPARED: clean

`recon_2k.py`, anchor round-trip (independent of all reconstruction math):
**cos = 1.00000 on every full_attention layer.**

### Stage 2 — KV STORED: clean, and much improved by §10j/§10k

| layer | rank | SVD-only | +resid | res-pos | no-res |
|---|---|---|---|---|---|
| 3 | 24 | 0.9254 | **0.9999** | 0.99993 | nan |
| 7 | 48 | 0.9932 | 0.9963 | 0.99982 | 0.9951 |
| 11 | 48 | 0.9939 | 0.9964 | 0.99983 | 0.9953 |
| 15 | 48 | 0.9948 | 0.9972 | 0.99989 | 0.9964 |
| 19 | 16 | 0.9165 | **1.0000** | 0.99995 | nan |
| 23 | 16 | 0.8976 | **1.0000** | 0.99996 | nan |

`no-res = nan` on layers 3/19/23 means EVERY position now carries an exact
residual — the §10j budget fix and §10k threshold removal working: reconstruction
is exact, not merely close.

### Stage 3 — KV RETRIEVED: clean

The needle's block is **DENSE, not compressed** on this prompt (compressed anchors
195..967; dense anchors 0, 65, 130, 1024..2048 — block size is 65 here, adaptive
micro-blocks). So routing is not involved for it; the dense window is.

`probe_dense_2k.py`:

```
workspace=(2,1152,256)  L_dense=1136  max_dense_len=1152   packed-from-0: True
rows checked=1136  mismatched=0
needle abs 9/10/17/18 -> rows 9/10/17/18, cos=1.0000, RoPE uses abs 9/10/17/18
```

Every row matches the token it claims to be, the needle is present bit-exactly,
and it is rotated at its true position. (The dense set is 20 blocks x 65 = 1300
tokens against max_dense_len 1152, so trimming IS active — but the block-0
protection holds and the needle survives.)

### Stage 4 — DECODED: this is where it breaks

All three upstream stages verify clean, at cos 1.0000, and the model still emits
garbage. That is §10f reproduced cheaply: byte-identical, verifiably-correct
inputs into the decode kernel, wrong output out.

**The staged answer: KV is prepared correctly, stored correctly, and retrieved
correctly. The defect is entirely in decode.**

### Probe caveat recorded

`probe_routed.py` prints "BLOCK 0 NOT ROUTED" — that is a FALSE ALARM on this
prompt (block 0 is dense, so it is correctly absent from the compressed routing
set), and its slot->anchor map leaks the loop variable across layers. Do not
quote it without fixing both.

---

## §10p — GPU-side proof: the divergence ORIGINATES at layer 3, on identical inputs (2026-07-29)

The GPU-read probe finally works (§10m fix: debug buffer returned via
`dkv_core.last_debug_buffer()` and read from Python) and was run against the cheap
2k repro. It reports what the KERNEL ITSELF reads, not what the host believes.

### Layer 3 — first DKV layer — three runs

```
Q       = [1.267578, 0.990723, -1.114258, -0.104248]   IDENTICAL
denseK  = [-0.175781, 0.072632, -0.029861, -0.070007]  IDENTICAL
slots   = [3, 4, 5, 6]                                 IDENTICAL
anchK   = [0.548828, 1.217773, 2.384766, 1.388672]     IDENTICAL
seq_len = 64.0        scale = 0.32373
u_scale = 1.161133    U[0]  = 59.0                     ALL IDENTICAL

out     = [-0.367432, 0.740234, 0.723145, 0.056458]    run 1
          [-0.455078,-0.066895, 0.157104, 0.486816]    run 2
          [-0.002596, 0.344238,-0.328857,-0.010689]    run 3
```

Every input the kernel touches — the query, the dense key it reads, the routed
slot list, the anchor key, the sequence length that bounds its token loop, both
scales, and the first int8 U value — is **byte-identical as observed BY THE GPU**.
The output is not merely perturbed; it is a different computation
(0.31 / -0.39 / 0.01 in the same slot).

### The origin is layer 3, not layer 7

Layer 7's Q DOES vary across runs — but only because it is downstream of layer 3's
output. Layer 3's own Q is identical every run. So the chain starts at the FIRST
compressed layer, and everything after it is consequence.

That also resolves the §9x/§9w picture: the tail block's `active_k` and layer
11+'s `dense_k` differed for the same reason — they are all downstream of layer 3.

### Combined with the earlier eliminations

| checked | verdict |
|---|---|
| every kernel input, GPU-observed | identical |
| host state, kernel args, kernel return | identical args, differing return (§10f, n=5) |
| kernel on fixed inputs in isolation | bitwise deterministic 6/6 |
| GPU ordering (COMMIT_AND_WAIT) | eliminated |
| uninitialised threadgroup memory | eliminated (all 15 arrays zeroed) |
| buffer aliasing (MTLBuffer identity) | eliminated (overlaps=0) |
| intra-threadgroup races (1 thread/tg) | eliminated |
| recycled output allocation | eliminated (persistent buffer) |
| platform/MPS | eliminated (DKV off = bit-reproducible) |

**A Metal kernel, reading provably identical bytes, deterministic in isolation,
with ordering/threadgroup/aliasing/allocation all excluded, produces three
different results.** The mechanism remains unidentified — but it is now pinned to
one kernel invocation at one layer, observable in 20 s, with every input verified
from the GPU side.

### Not yet dumped from the GPU

`VK_pool` / `VV_pool` (the low-rank bases), the residual values, and
`cos_anc`/`sin_anc`. Those are the only kernel inputs not yet read back through
the debug buffer. VK/VV are the largest contributors to the delta term and are
the obvious next dump — one more slot in `dbg_buf`, same 20 s loop.

## §10q — **SOLVED**: `cos_dense`/`sin_dense` were passed as fp16 to a float32 kernel pointer (2026-07-29)

The decode nondeterminism tracked since §9w is fixed. Root cause, one line:

```python
# ACTIVE_RUNTIME/runtime/dkv_attention.py, MPS fused-decode call site
_cos = cos_all[0, dense_positions...].squeeze().unsqueeze(0).unsqueeze(1)   # fp16!
```

`dkv_decode.metal` declares those buffers `device const float*`:

```
device const float* cos_dense [[buffer(24)]],   // [L_dense, D]
device const float* sin_dense [[buffer(25)]],
```

`cos_all`/`sin_all` carry the model's dtype — fp16 on Qwen3.5-2B — and nothing
converted them. So the shader

1. read every **pair of halves as one float32**, i.e. rotated the dense window by
   nonsense angles, and
2. indexed **twice as far as the buffer is long** (1096 rows x 256 floats out of a
   1096 x 256 *half* allocation), walking ~560 KB past the end into whatever the
   MPS allocator had recycled there.

(2) is what made it nondeterministic: the overrun landed in different recycled
memory each process, so a temperature-0 decode on byte-identical inputs returned
a different answer every run.

### Why it presented the way it did

The garbage angles produced a handful of enormous dense scores. Measured at
layer 3 on the 2k repro, across three runs:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| `blocks_max` (compressed side) | 3.39575 | 3.39575 | 3.39575 |
| `dense_max` | 186118 | 33655.7 | 2164.78 |
| `global_d` | 1 | 1 | 1 |

`global_d == 1` **exactly** is the tell: one score exceeded every other by >100
in scaled units, so ~2700 real tokens all underflowed to zero and the softmax
returned a single junk token. The compressed path was never involved — its max
is identical to six digits across runs, which is why every compression,
residual, routing and pool fix in §10i–§10k left the symptom untouched. They
were all real bugs; none of them was *this* bug.

`cos_anc`/`sin_anc` were always built with an explicit `.to(torch.float32)`
(`dkv_attention.py:1405`), which is why only the dense half of PASS 1 was
corrupted and why the anchor tables hashed identical run to run.

### The measurement that found it

Every earlier GPU-side dump sampled 8 elements (`Q[0..7]`, `dense_K[0..7]`) and
was read as "every kernel input is verified identical". It was not: 8 of Q's 256
and 8 of dense_K's ~363k is not a check, and `dense_V`, `anchors_V`,
`res_val_V`, the per-slot scalars and **the RoPE tables** were never read at all.
Replacing the samples with **whole-array FNV-1a checksums over exactly the
slices the kernel reads** put the answer on screen immediately:

```
Q=0x65a6e4 denseK=0xd96f19 denseV=0x7c9261 ancK=0xcfd671 ancV=0xca3a36
U=0xd4f4c3 VK=0x54883d VV=0xcc090c resK=0x917a7c resV=0xebed34   <- all identical
cosdense=0xd15754 / 0x3cf92e / 0x737ed8                          <- DIFFERENT EVERY RUN
```

**Method note.** §10p's "every input verified identical from the GPU side" was
overstated on the strength of that weak instrument, and it steered the next several
sessions toward "the arithmetic itself is nondeterministic" — a conclusion with no
mechanism. The rule already in force ("validate every instrument on a known case
first") needs a companion: *state the coverage of a check, not just its result.*
A checksum's value is the fraction of the input it actually covers.

### Fixes

1. `dkv_attention.py` — `.to(torch.float32)` on `_cos`/`_sin` at the MPS
   fused-decode call site (and the empty-tensor fallback now says float32 too).
2. `metal_runtime.mm` — `rope_f32()` converts any non-float32 RoPE table and
   warns once naming the offender, so a caller cannot silently reintroduce it.
3. `metal_runtime.mm` — `expect_dtype()` asserts the element width of all 20
   remaining buffers against what the shader declares, and throws with an
   explanation. This whole class of bug is now loud instead of silent.

### Also fixed on the way (independent, real, kept)

`thread_val` in `dkv_decode.metal` was declared `float thread_val[256]` — **per
thread**. Every accumulation loop strides `for (d = tid; d < D; d += t_per_tg)`,
so a thread only ever touches `D / t_per_tg` = 4 dims at D=256, but the array
reserved 256 floats in each of the 64 threads: **64 KB of thread-private memory
per threadgroup to hold 4 useful values each**. It is now indexed by the thread's
own slot (`DKV_MAX_VAL_PER_THREAD = 8`, 1 KB), with the host asserting
`tg_threads >= ceil(D / 8)`. This did NOT fix the nondeterminism (verified: 4
runs, 4 different outputs, before the dtype fix) — it is a separate resource bug.

### Verification

* Isolated Metal suite: **22/22 in both residual conventions**.
* 2k repro, `dense_max` 7.34 (was 1.9e5), `global_d` 12.26 (was exactly 1),
  `cosdense` checksum identical across runs, kernel `out` identical across runs.
* Needle, 3 consecutive runs, no instrument:
  `hit=True 'ZEBRA-4471-QUARTZ'` — **exact, 3/3**. Prior best was a corrupted
  `...ZEBRA-2024`, and before that degenerate tails like `ZEB. B. A.!!!!!`.

### Still open

* The Triton/CUDA decode kernel has not been checked for the same dtype
  contract, and nothing here is validated on real NVIDIA hardware.
* `DKV_DENSE_VALID_LEN` is still default-OFF, so the kernel still attends the
  padded tail of the dense workspace as if it were live context; MLX masks
  unconditionally. Now that decode is deterministic, this A/B is finally
  measurable and should be re-run.

## §10r — CUDA/Triton audit: 3 correctness bugs, 1 divergence closed, per-token syncs removed (2026-07-29)

Follow-up to §10q. The Metal fix landed; this pass asks what the SAME sweep finds
on the CUDA/Triton side, since that is where the work is heading next.

### C1. Exact-keys storage format did not match the CUDA decoder — REGRESSION FROM §10i

`_exact_keys_enabled()` was flipped to default ON to match MLX. It is read by
exactly two places: `lowrank.py` (the WRITER) and `metal_runtime.mm` (the Metal
READER). **Triton never read it.** Triton applies residuals as a pure correction:

```python
s = tl.where(offs_s == r_pos_k, s + r_corr, s)   # score: ADDS
O_res_corr += p_at * rv                           # value: ADDS
```

no substitution, no removal of the lossy twin. Feeding it exact-form residuals
(`exact - anchor`) adds almost the whole key a second time on top of the SVD
estimate, for every residual token.

FIX: the default is now derived from the target device, because this is a
STORAGE FORMAT and must match whichever kernel reads it back — MPS ON (Metal
substitutes), CUDA OFF (Triton corrects). Explicit `DKV_RESIDUAL_EXACT_KEYS` /
`DKV_RESIDUAL_EXCLUDE_SVD` still overrides either way.

### C2. K and V halves of `compress_lowrank` used DIFFERENT gates

The K half read `os.environ.get("DKV_RESIDUAL_EXACT_KEYS", "0")` directly
(default OFF) while the V half called `_exact_keys_enabled()` (default ON). So K
was written in CORRECTION form and V in EXACT form **in the same block**, while
the Metal kernel — reading its own flag — substituted both. That is exactly the
mixed-semantics state the batch path's own comment calls "strictly worse than not
enabling the mode at all". Both halves now use one gate.

Note: the §10q needle result (3/3) was obtained UNDER the mixed semantics.
Re-verified after the fix: still 3/3 exact.

### C3. Triton read past the pool's rank dimension — BOTH kernels

```python
offs_r = tl.arange(0, R)      # R = R_pad = next_power_of_2(layer_rank)
vk = tl.load(vk_ptrs)         # no mask on r
u  = tl.load(u_ptrs, mask=s_mask, other=0.0)   # masked on S only
```

`R_pad` is derived from the LAYER's rank; the pool's rank dimension is
`pool_rank` = max rank across layers (48 here). They disagree:

| layer rank | R_pad | pool rank dim | effect |
|---|---|---|---|
| 24 | 32 | 48 | reads 8 columns this layer never wrote |
| 48 | **64** | 48 | **reads 16 rank-rows into the ADJACENT SLOT's basis** |
| 16 | 16 | 48 | correct |

With `get_layer_rank`'s 0.75x/1.5x/0.5x schedule on base rank 32, the middle
band gets exactly 48 — so every 1.5x-boosted layer summed another block's basis
into its own scores and values. This is the Triton counterpart of the Metal
`pool_rank`-vs-`rank` stride bug: different mechanism, same family.

FIX: `R_REAL` threaded through as a constexpr; `r_mask = offs_r < R_REAL` applied
to the `vk`/`vv`/`u` and fact-path `u_val` loads in both kernels.

### C4. Dense-padding mask — Metal was the ONLY backend not masking

* MLX: `where(arange(max_dense_len) < dense_len, 0, -inf)` — unconditional
* Triton: `valid_t = mask_t & (offs_t < L_dense_valid)` — unconditional
* Metal: opt-in via `DKV_DENSE_VALID_LEN`, and the opt-in defaulted **OFF**

so Metal attended the padded workspace tail as live context. The old note kept it
off citing an A/B that predates §10q's determinism fix and therefore measured
nothing. `DKV_DENSE_VALID_LEN` now defaults ON; `=0` restores the old behaviour.

### C5. Per-token device syncs removed (F8 from CUDA_TRITON_AUDIT.md)

Four unconditional guards sat on the per-layer, per-token decode path. Each
converts a GPU tensor to a Python bool or calls `.item()`, both of which force a
device->host sync — a full pipeline stall, every token, to print nothing:

* `fused_decode_mps` entry: `if torch.isnan(Q).any():`
* `fused_decode_mps` exit: `if torch.isnan(O).any() or torch.isinf(O).any() ...`
* `fused_decode_mps` exit: `if lse.max().item() > 100.0:`
* `dkv_attention.py`: `if torch.isnan(attn_output).any():`

All now behind `DKV_DEBUG_NUMERICS` (default 0). Note `fused_decode_mps` is NOT
device-gated despite the name — it is the PyTorch path CUDA falls back to, so
these cost CUDA too.

### Checked and CLEAN (recorded so they are not re-investigated)

* Triton cannot have the §10q dtype bug: `tl.load` infers element type from the
  pointer's torch dtype. There is no hardcoded `float*` to reinterpret against.
* Triton passes REAL tensor strides, so the Metal stride bug cannot occur.
* `triton.next_power_of_2` is already applied to D/R/S_MAX — I suspected a
  missing power-of-2 constraint and was WRONG; it is handled.
* Call sites pass `R=_layer_active_rank` (the layer's rank, not the flat one).
* `MAX_RESIDUAL` is a derived constexpr (`max_res_pad`), not a 4th instance of
  the 64-hardcode. `S_MAX`'s 64 default is dead — all callers pass `session_mbs`.
* `DKV_RESIDUAL_EXACT_ROPE` is honoured on the Triton path (default 1).
* No stale `DIFFKV_` env prefix remains anywhere in ACTIVE_RUNTIME.
* Residual budget reaches BOTH compress paths (param / `pool.max_residual_tokens`).

### Measurement hygiene — two harness defects that corrupted earlier conclusions

1. **Concurrency.** §10n's "`mid` is 640s vs `high` 94s, the heavier preset is 7x
   FASTER" was an artifact of running jobs concurrently on one GPU. Serial:
   mid 56s, high 74s, low 60s — normal ordering. There is no anomaly to explain.
   `low`'s 1625s was likewise contention, not `decode_cache_enabled=False`.
2. **`cp` over an mmap'd `.so` SIGBUSes running processes.** Rebuilding while
   tests were in flight killed three of them with no traceback, which read as
   "incomplete output". Install atomically: `cp` to a temp name, then `mv`.
3. **`preset_smoke.py` used `max_new_tokens=12`**, which truncates
   `ZEBRA-4471-QUARTZ` mid-word once the `<think></think>` preamble is counted —
   producing `ZEBRA-4471-QUART` and scoring it a FAILURE. Any preset conclusion
   drawn from that harness at 12 tokens is unreliable.

### Still open

* **Nothing on the CUDA side is hardware-validated.** All of C1/C2/C3 were fixed
  on a Mac with no NVIDIA GPU. C3 in particular changes kernel numerics and has
  never been compiled by Triton. Run `colab/validate_cuda_dkv.py` FIRST on a
  cloud GPU, with `DKV_TRITON_STRICT=1` so a kernel that fails to compile raises
  instead of silently falling back to the slow PyTorch path.
* **8.2k needle still fails** — `ZEBRA-4471`, losing `-QUARTZ`, with 40 tokens
  available (so not truncation). 2k is exact and deterministic.
* **`low` preset unexplained.** An A/B inverted the obvious hypothesis: res40
  passed and res128 FAILED, with a 10x gen-time swing between the two. n=1 per
  cell; do not draw a conclusion from it.
* `TieredBlockStore` (default ON) does `block_indices.cpu().tolist()` plus a
  Python `update_heat` loop per decode call — a real per-call D2H sync. Left
  alone because it feeds eviction and the tradeoff needs a GPU to measure.

## §10s — First real CUDA throughput numbers, and they are not good (2026-07-29)

Measured through generate() (colab/bench_dkv_tps.py), RTX PRO 4000, Qwen3.5-2B
fp16, 12,870-token prompt, 64 decode tokens, preset mid, fused kernel CONFIRMED
running (`triton fallback count: 0`, "COMBINED path ACTIVE, N_sparse=43,
L_dense=2050"):

| | DKV | dense | 
|---|---|---|
| decode | **11.3 tps** (88.7 ms/tok) | **38.7 tps** (25.8 ms/tok) |
| peak VRAM | 4.45 GB | 4.94 GB |

**DKV is 3.4x SLOWER than dense and saves 10% VRAM.** That is a bad trade as
measured. Two things must be said about it before anyone acts on it:

### 1. This model/context is the wrong showcase, structurally

Qwen3.5-2B is a HYBRID: only 6 of 24 layers are full_attention, with 2 KV heads
and head_dim 256. Its KV cache is therefore tiny:

    12,870 tok -> 0.147 GB   (3.2% of the 4.55 GB weights)
    128,000 tok -> 1.46 GB

DKV's own pool was **491 MB — larger than the 147 MB of KV it replaces.** There
is essentially nothing to compress here at any context this model supports. A
DKV win needs a model whose KV actually dominates (all-full-attention, more KV
heads): Qwen2.5-14B or Llama-3.1-8B.

### 2. The 4.3 tps figure from profile_decode_step.py was an artifact

That harness drives `model(input_ids=...)` directly, bypassing the wrapper's
session setup and routing, so the fused decode never engages (its "dkv" bucket
reads 0.0 ms and the "COMBINED path ACTIVE" banner is absent). It reported 4.3
tps on BOTH Path A and Path B, which is what falsified the "Path B has no fused
kernel" explanation for the slow number -- that was true of Path B but was not
the cause. Benchmark through generate(); bench_dkv_tps.py prints the triton
fallback count next to every number so this cannot recur.

### The MLX gap is real

MLX DKV runs 38-42 tps at 8k flat to 64k (memory: fused_decode_mlx_2x). CUDA DKV
is 11.3 tps at 13k -- and CUDA DENSE is 38.7, i.e. roughly MLX-DKV speed. So
CUDA's DKV decode is ~3.5x off MLX's, on the same algorithm. That gap is the
thing to chase, and it is NOT the fused kernel failing to run.

### Honest disclosure on the workspace fix

§10r raised max_dense_len from 1152 to 2050 rows (the block-size bug). That was
required for correctness -- it was silently trimming live dense blocks -- but it
also means decode now attends ~78% more dense tokens per step than the runs
before it. Some of the 88.7 ms/token is that. The fix is still right; the cost
should be attributed honestly when comparing against older numbers.

### Next

* Long-context sweep (8k/16k/32k/64k) DKV vs dense, both from bench_dkv_tps.py.
* Re-run on a KV-heavy model (Qwen2.5-14B, Llama-3.1-8B) -- the paper's regime.
* Profile the FUSED path (the existing profiler cannot: it never engages it), to
  attribute the ~63 ms/token DKV adds over dense.
* `decode kernel warmup failed (AssertionError)` x4 -- Dynamo on torch 2.11 +
  Blackwell, so decode runs eager. TORCHDYNAMO_VERBOSE=1 for the real stack.
