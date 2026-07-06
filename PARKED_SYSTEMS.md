# PARKED SYSTEMS — built-but-inactive subsystems (idea preservation)

**Created 2026-07-06 (Opus 4.8).** Purpose: a single durable catalog of every DiffKV
subsystem that is **implemented but not active** on the Mac/MLX runtime, so the ideas
are not re-derived or rebuilt from scratch. Per the owner's decision, the code is
**cataloged in place, NOT deleted** — most of these are load-bearing for the
PyTorch/CUDA (Linux) backend and/or the test suite, so removing them would break the
cross-platform build. This file is the map: what each system is, where it lives, why
it's parked, the evidence, and how to revive it.

> This catalog was produced while auditing an external "Antigravity" review. **Two of
> its three "🔥 must-have wins" were already shipped** (see §0), and its one testable
> accuracy claim for the Factual Store was **refuted by an A/B on the real harness**
> (see §1). Treat unverified performance/accuracy claims about this project as
> hypotheses until benchmarked — that has been the repeated pattern.

---

## 0. NOT parked — already active in serving (do not "re-enable")

These were flagged by the audit as "off by default, should enable." **They are already
enabled in every real serving entrypoint** via `setdefault`. The *wrapper-level* default
differs deliberately (see below), which is what made them look disabled.

| Flag | Wrapper default | Serving (CLI + gateway) | Verified this session |
|---|---|---|---|
| `DIFFKV_DECODE_CACHE` | `0` | **`1`** ([cli.py:646](ACTIVE_RUNTIME/serving/cli.py#L646), [gateway:931](ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py#L931)) | NIAH 2/2 @8k+16k, ~21 tps |
| `DIFFKV_COMPRESSED_DECODE` | `1` (always-sparse, to exercise the arch in tests) | **`auto`** + `MIN_CTX=8192` (fast dense <8k, sparse ≥8k) | same run |
| `DIFFKV_SPARSE_BIAS` | `auto` | `auto` | — |

The `setdefault` in [cli.py:643](ACTIVE_RUNTIME/serving/cli.py#L643) and
[gateway:928](ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py#L928) is the source
of truth for the "best config." An explicit env var still wins. **Do not flip the
wrapper-level `COMPRESSED_DECODE` default to `auto`** — it is `1` on purpose so tests
always drive the sparse path.

---

## 1. Factual Store + Eagle Lookback + VSL  — PARKED (net-negative on MLX)

**What it is.** A prefill-built exact-KV store of "factual" spans (entities, digits,
distinctive tokens). At decode it queries the current token against the store, promotes
spans to "prime" entity nodes via **Eagle lookback** (middle-layer key self-similarity ×
key-norm × IDF), and constrains generation with **Verifiable Sequence Locking (VSL)**
logit masking. Meant to fix number/entity corruption when facts are crammed into heavily
compressed blocks.

**Where it lives (fully wired on MLX, gated by `DIFFKV_FACTUAL_STORE=1`):**
- Store/index: [native_core/srl/factual_store.py](ACTIVE_RUNTIME/native_core/srl/factual_store.py) (939 LOC; Eagle lookback ~`:173`, joint salience ~`:265`, prime promotion ~`:495`)
- VSL masking: [native_core/srl/factual_alignment.py](ACTIVE_RUNTIME/native_core/srl/factual_alignment.py) (`get_allowed_tokens_vsl:207`, `update_vsl_state:302`)
- MLX prefill capture: [mlx_diffkv_wrapper.py:1032](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py#L1032)
- MLX store build (prefill→decode boundary): [mlx_diffkv_wrapper.py:1054](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py#L1054)
- MLX decode-time query (layer 0): [mlx_diffkv_wrapper.py:2568](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py#L2568)

**Why parked — the A/B data (this session).** Harness:
[benchmarks/relational_ab.py](benchmarks/relational_ab.py), the adversarial multi-entity
"module registry" (5 modules, shared `BRAVO-` prefix, distinguishing 4-digit tail) — the
*exact* case the store was designed for. Qwen2.5-1.5B-Instruct-4bit.

| Context | `exact` (dense) | `sparse` (no store) | `sparse_factual` (store ON) |
|---|---|---|---|
| 3.5k | 5/5 clean | **5/5 clean** | 5/5 but noisy output (`"B2741- The Wren…"`, repetition) |
| 12k  | 5/5 clean | **5/5 clean** | **4/5** + corrupted output (`"B741-2741.\n BRAVO-…"`) |

**Verdict: the store is a NET NEGATIVE on MLX.** Plain sparse (SVD + exact residuals) is
already at the dense upper bound on the case the store targets; enabling the store
*lowers* exact-key recall (5/5→4/5) and consistently **corrupts output formatting** (stray
`"B"` prefixes, `"The activation key is…"` repetition — VSL over-masking), on top of the
documented **~32% decode speed penalty**. The audit's "table recall 1/5→4/5" claim is the
opposite of what the harness shows. **Keep `DIFFKV_FACTUAL_STORE` off (current default).**

**How to revive (only if a future workload actually needs it).** Requirements before it
could be a net win: (1) find a workload where `sparse` genuinely mis-binds (none found up
to 12k on 5 crammed entities); (2) fix VSL over-masking so it stops corrupting normal
output; (3) port the CPU graph-walk / logit-mask to MLX/Metal to kill the GPU↔CPU stall;
(4) make the store lazy (query only on punctuation/newline/distinctive tokens, not every
step). Re-run `relational_ab.py` all three arms + confirm no output-quality regression
before re-enabling. Raw logs: `scratch/factual_ab/`.

---

## 2. Semantic Index (ANN)  — CUDA/Linux only, cataloged in place

**What it is.** Approximate-nearest-neighbor search over 64-dim block descriptors; one of
the 10 channels of the full query router (concentric-centroid routing).

**Where:** [native_core/srl/semantic_index.py](ACTIVE_RUNTIME/native_core/srl/semantic_index.py);
consumed by [query_router.py](ACTIVE_RUNTIME/native_core/srl/query_router.py),
[session_srl_state.py](ACTIVE_RUNTIME/native_core/srl/session_srl_state.py),
[kv_runtime_manager.py](ACTIVE_RUNTIME/native_core/kv_runtime_manager.py).

**Why parked on Mac.** The MLX wrapper constructs `SessionSRLState(semantic_index=None,
chunk_graph=None, …)` ([mlx_diffkv_wrapper.py:1088](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py#L1088)),
so the ANN channel is never built on Apple Silicon. On Mac the exact **residual-key
router** (linear block scan) is fast enough and more accurate at ≤32–64k. The ANN only
pays off at ~100k+ where linear scans dominate. Still **live on the PyTorch/CUDA path** →
do not delete.

**How to revive on Mac:** build 64-dim descriptors after MLX prefill and pass a non-None
`semantic_index` into `finalize_srl_index`. Only worth it if targeting 100k+ contexts.

---

## 3. Chunk Graph routing  — CUDA/Linux only, cataloged in place

**What it is.** A block-to-block similarity graph built after prefill; query activations
propagate via a 2-hop random walk to pull in neighboring blocks.

**Where:** [native_core/srl/chunk_graph.py](ACTIVE_RUNTIME/native_core/srl/chunk_graph.py)
(594 LOC; `ChunkGraph:30`, `build_chunk_graph:106`); consumed by the same router/manager
files as §2, plus `test_multidim_srl.py`, `test_facter_retention.py`.

**Why parked on Mac.** Same as §2 — passed as `None` on MLX. Graph build cost isn't worth
it below ~100k; the residual-key router covers the tested range. Live on CUDA path.

**How to revive on Mac:** build + pass a non-None `chunk_graph` at prefill→decode. Target
100k+ only.

---

## 4. Stratified 4-bit SVD quantization (`pack_int4`/`unpack_int4`)  — torch/CUDA only

**What it is.** Splits SVD singular vectors `U` into a high-variance "semantic" part
(quantized to int4, packed into int8) and a low-variance "factual" part (kept fp16), to
shrink the `U` pool.

**Where:** [native_core/compression/lowrank.py:25](ACTIVE_RUNTIME/native_core/compression/lowrank.py#L25)
(`pack_int4`), `:61` (`unpack_int4`), `:462` (`compress_layer_blocks_gpu`). Imported only
by `test_facter_retention.py` and within `lowrank.py`.

**Why parked on Mac.** MLX's `compress_mlx_block` stores `comp_U` / `comp_VK` / `comp_VV`
directly in fp16 and never calls the int4 pack/unpack path. The pool is dominated by `V`,
residuals, and anchors, so int4-on-`U` saves little; without GPU bit-unpacking the
reconstruction tax outweighs the memory win below ~100k. Skip unless memory-bound at very
long context.

---

## 5. Speculative decoding  — optional, wired, off by default (NOT dead)

**What it is.** Draft-model speculative decoding: a small draft model proposes tokens,
the main model verifies.

**Where:** [plugins/speculative.py](ACTIVE_RUNTIME/plugins/speculative.py),
[native_core/srl/speculative.py](ACTIVE_RUNTIME/native_core/srl/speculative.py); driven by
[batch_engine.py](ACTIVE_RUNTIME/serving/batch_engine.py). Enabled only via the
`--draft-model` CLI flag on the API gateway.

**Status.** Not dead — it's an opt-in feature that works in the batching server. Left off
because it needs a second model and only helps throughput for predictable outputs. Keep
as-is; enable per-deployment with `--draft-model`.

---

## 6. Marginal flags — documented, leave at defaults

- **`DIFFKV_ROUTE_ONCE`** (default `0`, [mlx_diffkv_wrapper.py:2213](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py#L2213)):
  caches the block-router selection for the whole generation. **Superseded by
  `DIFFKV_DECODE_CACHE`**, which already re-routes + re-materializes every 16 tokens — so
  it captures ~90% of the route-once speed win *without* going stale mid-generation. The
  audit's proposed "adaptive lazy router" is essentially the decode-cache interval that
  already exists. No work needed.
- **`DIFFKV_RESIDUAL_COVERAGE_FRAC`** (default `0`, [mlx_diffkv_wrapper.py:167](ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py#L167)):
  reserves a fraction of the residual budget for evenly-spaced (stratified) tokens.
  Experimental; dilutes residuals for dense numeric spans if set too high. Leave `0`.
- **`DIFFKV_V_SCALE`** (default on): balances K/V energy before the joint SVD. Working;
  don't disable (degrades value reconstruction with no memory benefit).

---

## Summary

| System | Runtime | Status | Action |
|---|---|---|---|
| Decode cache + `COMPRESSED_DECODE=auto` | MLX | **already active in serving**, verified 2/2 NIAH | none (do not re-flip wrapper default) |
| Factual Store + Eagle + VSL | MLX (built) | **net-negative** per A/B (5/5→4/5 + output corruption + ~32% slower) | keep OFF; revive only after VSL fix + GPU port |
| Semantic ANN index | CUDA only | `None` on Mac; fine ≤64k | catalog; revive at 100k+ |
| Chunk graph routing | CUDA only | `None` on Mac; fine ≤64k | catalog; revive at 100k+ |
| Stratified int4 SVD | torch/CUDA only | MLX uses fp16 `U` | catalog; revive if memory-bound >100k |
| Speculative decoding | batch server | opt-in via `--draft-model` | keep as-is |
| `ROUTE_ONCE` | MLX | superseded by decode cache | leave `0` |
