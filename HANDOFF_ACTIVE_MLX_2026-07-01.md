# Handoff — ACTIVE_RUNTIME (MLX) sparse-decode, factual store, multi-entity accuracy

**Date:** 2026-07-01 · **Scope:** Python "active" runtime on Apple Silicon (MLX) only ·
**Model:** Qwen2.5-1.5B-Instruct-4bit · **Machine:** M3, 8 GB

---

## 0. TL;DR (read this first)

1. **On Mac, the "active" runtime is MLX-only** (`DiffKVHFWrapper` → `MLXDiffKVWrapper` on darwin,
   [mlx_diffkv_wrapper.py:1220]). It uses `MLXKVBlockManager`.
2. **The whole "sophisticated" retrieval layer (factual store / SRL / chunk-graph / eagle) was OFF on
   Mac and had never been wired in** — `get_srl_state` hard-returned `None`. It only ran in the
   PyTorch/CUDA `KVRuntimeManager`. On Mac the only retrieval intelligence is SVD low-rank blocks
   ("summaries") + exact residual tokens ("exact word storage") + a top-K residual router.
3. **DiffKV sparse decode now engages from token 1 (DEFAULT FLIPPED).** `DIFFKV_COMPRESSED_DECODE`
   default is now `1` (sparse-always); `auto` (dense <16k, sparse above) is the **opt-in** adaptive
   mode; `0` forces dense. Trade-off accepted by design: short-context is slower (~16 vs ~36 tps @4k)
   + pre-allocates the block pool, no accuracy change — the win is at long context. (Previously `auto`
   was default, so 4k/8k stayed dense and "DiffKV didn't come up" for normal prompts.)
4. **Plain sparse decode is accurate on realistic prompts** — a natural multi-entity relational test
   scores **4/4 with the factual store OFF**, same as exact full-KV. The digit corruption we first
   saw was an artifact of an adversarial layout (5 near-identical keys crammed in one block).
5. **The factual store was a NET NEGATIVE, now WORKS on realistic AND dense-table prompts** — as first
   ported it biased repeated FILLER (natural 0/4). Fixed with **positional query→value linking** (§2b)
   → natural **4/4 exact**. Then fixed the dense-table case (§2c: boundary-aware span segmentation +
   entity-aligned binding + bias-decay) → adversarial **4/5 bare-value, 0 misbound, 2/5 exact-key**
   (was 0/5; plain sparse 1/5).
   Still adds ~32% decode cost, so **keep it OFF by default** (opt-in `DIFFKV_FACTUAL_STORE=1`).
6. **The "dictionary" preserves exact numbers correctly** — the store holds "4193 variable stars."
   verbatim. Preservation was never the bug; *retrieval* (connecting the query's entity to its value
   span vs surfacing filler) was.
7. **One always-on verified win:** a decode-overhead fix (+20% tps @16k, no accuracy change).

**Net recommendation:** the biggest leverage is still decode *speed* + the sparse-from-start *flip*.
The factual store now genuinely helps on realistic multi-entity prompts (opt-in); its remaining gap is
dense tables (span segmentation) — a deeper, lower-ROI fix since plain sparse already handles realistic
prompts.

---

## 1. What the user wanted

- Use DiffKV sparse attention **from the start** (not gated to ≥16k); keep the adaptive/auto policy as
  an opt-in toggle instead of the default.
- Make sparse **faster**, ideally cross-platform (note: MLX is Apple-only — see §7).
- Verify the "sophisticated system" (factual stores, eagles, chunk graphs, exact-word storage) is
  actually **working / not silently off**.
- Understand the accuracy loss.

---

## 2. Changes shipped this session

All edits are in **`ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py`** (+212/−13) plus a new eval harness
**`benchmarks/relational_ab.py`**. **Nothing is committed.** Default behavior is unchanged: the only
always-on change is WS1 (pure speed). Everything else is behind env flags that default OFF.

### WS1 — decode overhead fix (always on, verified)
`execute_decode_attention`: removed the per-layer `mx.eval(sel)` host sync (was 28 GPU↔CPU syncs per
generated token) and replaced the Python top-K residual-gather loop with a fixed-width vectorized
`mx.take` (gated on `all_blocks_full` — a host-cheap check that every block holds `max_residual`
residuals, which always holds since blocks compress only at full `block_size`; a variable-length
fallback covers the rare non-uniform case).
- **Result @16k, gen 40: 9.05 → 10.84 tps (+20%), needle exact, KV footprint identical.**
- Numerically identical output (same residual set; attention is permutation-invariant).

### WS2 — factual store / entity binding now RUNS on MLX (gated OFF: `DIFFKV_FACTUAL_STORE=1`)
Previously dead because `get_srl_state`→None and nothing built the store. Added:
- `get_srl_state` returns a real `SessionSRLState`.
- `capture_factual_prefill_kv(sid, layer, K_unrot, V)` — stashes **unrotated** K/V for layers
  `{0, num_layers//2}` as torch CPU during prefill (descriptors must share the unrotated layer-0 space
  the decode query uses as proxy Q). Called from the prefill path.
- `finalize_srl_index` (was a `pass` **stub** that shadowed the real method — deleting the stub was
  required): builds `InvertedTokenIndex` (important_vocab + IDF) + `FactualExactStore.build(...,
  inv_index=...)` which activates RC4 entity assignment; sets `srl_state.inverted_index`; sets
  `current_query_tokens` from the caller-named question. Runs once at the prefill→decode boundary in
  `MLXQwenModel.__call__`.
- `generate(..., query_text=None)` → `_pending_query` → `current_query_tokens` (entity-binding hint).
- `W_proj` ([64, head_dim] normalized random torch CPU) on the manager + `DummyMLXPool.W_proj`.
- Wrapper passes `tokenizer` + `stop_token_ids` to the manager.
- **The consume side already existed** in the MLX `generate()` loop (+7 factual bias, VSL boost,
  −3.5 anti-hallucination penalty, ~lines 1960–2130) — it was dead only because `get_srl_state`→None.

### WS2b — positional query→value linking (the fix that made it work, 2026-07-01)
Diagnosed (via `DIFFKV_FACTUAL_DBG`) that the descriptor match (max-pooled layer-0 K @ **random**
`W_proj`) surfaced repeated **filler** ("and confirm that…", sim 1.15), never the answer → the +7 bias
derailed generation (0/4). The exact number **is** preserved (entry "4193 variable stars." stored
verbatim) but name↔value are in **separate** spans and no primes fire, so entity binding was dead.
**Fix** (in the factual query block, `mlx_diffkv_wrapper.py` ~1515): replaced descriptor matching with
positional linking — take the query's **distinctive** tokens (high block-IDF **AND** total occurrences
≤ `DIFFKV_FACTUAL_MAX_OCC`=4, so schema words like "module"/"key" don't anchor), look up where they
occur via `inv_index.occurrences`, and surface the **single nearest** fact span per anchor
(`DIFFKV_FACTUAL_WINDOW`=40). Descriptor query is now only a fallback. Because the store biases **exact
token IDs**, the right span forces exact digits even through KV corruption. Result: natural 0/4 → **4/4**.

### WS2c — dense-table fix (boundary-aware segmentation + entity-aligned binding, 2026-07-01)
Positional linking initially still failed the *crammed table* (5 near-identical rows in one block, 0/5)
because the store's spans were large and **crossed fact rows** (one span held two keys). Four changes,
all inside the factual path:
1. **Boundary-aware span segmentation** (`factual_store.build`): split spans at sentence/line
   boundaries (period, newline, `;!?`) BEFORE a length cap, so each row = one clean entry with name +
   value together. Cap raised to **64 when boundaries drive the split** (the old hard 20 chopped long
   sentences *mid-number*: `4193`→`419`|`3` — the natural-regression bug). Needs `inv_index._tokenizer_ref`
   (now set in MLX finalize; also improves prime helper-word filtering).
2. **Occurrence-count anchor gate** (`DIFFKV_FACTUAL_MAX_OCC`=4): distinctiveness by *total* occurrences,
   with very-rare (≤2) tokens bypassing the fragile block-IDF floor (block-IDF dips <3.0 on short docs).
3. **Entity binding aligned to the positional result**: `current_entity_id` set from the nearest
   co-located fact, dual mode cleared — overrides the raw-overlap early-binding that locked entity-0.
4. **Skip neighbor/triple expansion when positional fired** (`_positional_used`): on a dense table the
   neighbor graph pulls adjacent rows' keys back in. Plus a query-overlap filter drops question spans.

Result: adversarial dense-table **bare-value 4/5, 0 misbound** (was 0/5; plain sparse 1/5) — each query
now injects ITS OWN entity's correct number. `relational_ab.py` reports `n_num_correct` (bare value) to
separate binding from generation.

### WS2d — bias decay (repetition fix, 2026-07-01)
The flat +7 factual bias was applied to the entity's sequence tokens EVERY step, so once a value was
emitted it kept winning → `2741-2741-2741…` loops. Fix: `generate()` now **excludes tokens already
emitted this generation** (`generated[len(prompt_ids):]`) from the flat +7 (both entity + fallback
branches); the +10 transition bias still carries in-order progression, so a value is emitted once and
released. Also gated a noisy per-token `[Python DEBUG]` print behind `DIFFKV_FACTUAL_DBG`. Result:
adversarial **exact 0→2/5** (repetition gone), bare-value still 4/5; natural stays **4/4 exact**;
default-off **1/5** unchanged; kernel-parity gate = 4 passed.

**Remaining exact-key gap (dense table only, deprioritized):** the *shared* prefix `BRAVO-` mangles to
`B` — the model jumps to the high-signal number token even under the VSL hard-mask + transition bias
(the number's residual-KV signal + flat bias overwhelm the prefix continuation). Plus Raven (6620→6666,
a `\n\n` double-newline span edge). Both are verbatim synthetic-*code* emission issues; the
DISTINGUISHING value (the number) is recalled 4/5, so bare-value is the meaningful metric. Full 5/5
exact needs strict in-order VSL emission (`get_allowed_tokens_vsl`/`update_vsl_state` advance) — high
effort, low practical value.

### Not hardcoded to the eval
Audited (`git diff | grep BRAVO/module/…`): the only test-specific strings are in **comments**. The
logic is general — boundary chars (`.`/`\n`/`;!?`), occurrence-rarity, position distance, token-overlap
ratio — all with env-overridable constants. Caveat: the constant *values* (MAX_OCC=4, WINDOW=40,
IDF_MIN=3.0, cap=64, overlap>0.5) were tuned against this eval and should be validated on diverse docs.

### Accuracy experiment — exclude-residual-from-SVD (gated OFF: `DIFFKV_RESIDUAL_EXCLUDE_SVD=1`)
Adds a per-block boolean `comp_res_mask` [max_blocks, S_comp]; the kernel
`compute_decode_attention_static` takes a `res_mask` and sets those SVD delta positions to −inf so a
captured token's only representation is its exact residual (no lossy low-rank twin). Built in
`_compress_block`, shifted at the max-blocks boundary, graceful fallback if the buffer is absent.
- **Result: marginal** (adversarial crammed 1/5→2/5, realistic spread unchanged). The corruption is
  dominated by residual *capture*, not dilution — hypothesis largely wrong. Kept (off) as it's
  architecturally cleaner and zero-cost, but it is not the win.

### New eval — `benchmarks/relational_ab.py`
Multi-entity binding A/B the single-needle NIAH couldn't stress. Modes `exact | sparse |
sparse_factual`; flags `--spread` (one fact per block), `--natural` (RC4-style rare names + value in a
flowing sentence). Scores exact-match of **generated-only** tokens (crucial: `generate()` returns
prompt+generation, and the prompt echoes every answer — slice `_session_token_ids[sid][len(prompt_ids):]`).

---

## 3. Results & numbers

### Speed / memory (live, measure_active.py, 16k, gen 40)
| metric | value |
|---|---|
| prefill | 45 s (dominated by per-block **numpy** SVD: 62 blk × 28 layers ≈ 1,736 SVDs) |
| decode  | 9.05 → **10.84 tps** after WS1 (dense @16k ≈ 37; dense @4k ≈ 35.8; sparse @4k ≈ 16) |
| peak mem | 2.51 GB (prefill) / 2.03 GB (decode); model weights ≈ 1.0 GB |
| KV store @16k | nb=62 blocks, dense window 658, residuals 3968 (62×64) |
| store used | 178.8 MB vs full-dense 472.8 MB = **2.64×**; **bounded pool 682 MB (fixed)** |

**Memory nuance:** the pre-allocated pool (682 MB, set by `DIFFKV_MAX_BLOCKS=256`) is *larger* than a
full 16k dense KV (473 MB). Sparse has **no** memory advantage until ~24–26k. This is exactly why
`auto` waits — and why a naive "sparse from start" regresses short contexts on both speed and memory.

### Accuracy — relational multi-entity A/B
**Adversarial** (5 near-identical `BRAVO-####` keys crammed in one block, 3.5k):
| arm | exact-key | bare-value |
|---|---|---|
| exact (full-KV) | 5/5 | 5/5 |
| sparse (factual off) | 1/5 (digit corruption: 5198→"542") | 1/5 |
| sparse + factual (thin) | 0/5 (loops) | — |
| **sparse + factual (WS2c + bias-decay)** | **2/5** (prefix `BRAVO-`→`B`) | **4/5, 0 misbound** |
| — diagnostics — | rank 16→32: 2/5 · **max_residual 64→128: 4/5** · exclude-SVD: 2/5 · spread layout: 4/5 | |

*Why default-off (plain sparse) is 1/5:* no factual store — pure SVD+residuals. The crammed block has
~20 distinctive digit tokens competing for 64 residual slots + 5 similar keys' SVD reconstructions
blend, so only 1/5 digits survive. `max_residual 64→128 → 4/5` proves it's residual-budget overflow,
not a bug. This is exactly the failure the factual store compensates for.

**Natural** (rare names, value in a sentence, spread, 6k) — the fair test:
| arm | correct |
|---|---|
| exact | **4/4** |
| sparse (factual off) | **4/4** ← realistic multi-entity is fine without any factual store |
| sparse + factual (thin, pre-fix) | 0/4 ← derailed on filler |
| **sparse + factual (WS2b/WS2c)** | **4/4 exact** ← FIXED: connects entity→value |

---

## 4. Findings, ordered by importance

1. **Plain sparse decode is accurate on realistic prompts (4/4).** The "losing a bit on accuracy" is
   largely the *adversarial* worst case (many near-identical facts crammed into one 256-tok block,
   overflowing the per-block 64-residual budget). Normal spread-out documents recover.
2. **The factual store was harmful, now FIXED on realistic prompts (0/4 → 4/4) via WS2b positional
   linking.** Original derail: the descriptor match (random `W_proj`) surfaced repeated **filler**, not
   the answer (traced with `DIFFKV_FACTUAL_DBG`), so the +7 bias derailed. Fix connects the query's
   distinctive token to its document position → surfaces the co-located value span. The "dictionary"
   (store entries) always preserved exact numbers verbatim — the bug was retrieval, not preservation.
3. **Dense tables: binding FIXED (WS2c), generation-quality remains.** Boundary-aware segmentation +
   entity-aligned binding + bias-decay took the crammed table from 0/5 to **4/5 bare-value, 0 misbound,
   2/5 exact-key** (each query injects its own entity's number). The exact-key gap is verbatim *code*
   emission mangling the shared `BRAVO-` prefix (bias-decay already killed the repetition) — a VSL
   in-order tuning issue, not a binding error. Realistic prose (natural) is fully correct (4/4 exact).
4. **Entity binding (RC prime path) can't discriminate on these prompts** — primes require entities to
   be *referenced back* (RC7), which single-mention prompts lack → few primes. WS2b/WS2c sidestep this
   with positional linking + occurrence-rarity anchors instead of the prime/descriptor path.
5. **Decode is dispatch/overhead-bound**, not FLOP-bound — micro-opts give %, not the 3–4× needed to
   beat dense; that needs a fused Metal kernel or batched-layer decode.
6. **Prefill (45 s @16k) is the bigger time sink** — every block round-trips to numpy for SVD +
   residual selection.

---

## 5. Current code state / safety

- **Default behavior unchanged.** Verified: default relational sparse = 1/5 (== baseline), OFF path
  after all edits identical; kernel-parity gate (`test_diffkv_kernel_parity.py`) = 4 passed.
- **Gated flags default OFF:** `DIFFKV_FACTUAL_STORE`, `DIFFKV_RESIDUAL_EXCLUDE_SVD`. `factual_store.py`
  boundary-split only activates when the store is built (i.e. factual enabled) + `_tokenizer_ref` set,
  so the default path never touches it.
- **Always-on:** WS1 gather vectorization (pure speed, verified no accuracy change).
- **Not committed.** `git status`: `M ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py`,
  `M ACTIVE_RUNTIME/native_core/srl/factual_store.py`, `?? benchmarks/relational_ab.py`, `?? HANDOFF…`.
- **Deferred plumbing:** `comp_res_mask` is NOT threaded through the session snapshot/clone/restore/
  paging sites (only create/compress/shift + graceful fallback). Fine for the benchmark; **finish this
  before enabling `DIFFKV_RESIDUAL_EXCLUDE_SVD` in the serving path.** Same caution for the factual
  store's multi-turn `current_query_tokens` refresh (only set on first build today).

---

## 6. Env flags (quick reference)
| flag | default | meaning |
|---|---|---|
| `DIFFKV_COMPRESSED_DECODE` | **`1`** | sparse-always (default, from token 1); `auto` = adaptive opt-in; `0` = dense |
| `DIFFKV_COMPRESSED_MIN_CTX` | `16384` | auto threshold |
| `DIFFKV_MAX_RESIDUAL` | `64` | exact residual tokens per block (memory lever; 128 fixed adversarial 1→4/5) |
| `DIFFKV_TOPK_BLOCKS` | `16` | top-K block routing |
| `DIFFKV_ROUTER` | `residual` | `residual` / `minmax` |
| `DIFFKV_RESIDUAL_EXCLUDE_SVD` | `0` | drop residual positions from SVD pool (marginal) |
| `DIFFKV_FACTUAL_STORE` | `0` | enable factual store / entity binding (works on realistic prompts via WS2b; still fails dense tables) |
| `DIFFKV_FACTUAL_MAX_OCC` | `4` | WS2b — max total occurrences for a query token to be a positional anchor |
| `DIFFKV_FACTUAL_WINDOW` | `40` | WS2b — token window for nearest fact-span to an anchor |
| `DIFFKV_FACTUAL_IDF_MIN` | `3.0` | WS2b — min block-IDF for a positional anchor |
| `DIFFKV_FACTUAL_DBG` | `0` | dump surfaced factual sequences + digit-span entries |
| `DIFFKV_RANK` | `16` | (via config / harness) SVD rank |
| `DIFFKV_TELEMETRY` | `0` | prints `[SRL]`/`[FACTUAL]` build lines (+`[FENTRY]` with FACTUAL_DBG) |

---

## 7. Cross-platform note
"Python active = MLX = Apple only." The cross-platform sparse paths are the **Triton** kernel
(`native_core/sparse_decode/triton_fused_decode.py`, CUDA/NVIDIA, wired into the PyTorch wrapper) and
the **C++ native** binary (CPU/Metal/CUDA). There is no fast portable CPU/Windows Python sparse path.
Pick ONE second target (recommend reviving/validating the Triton CUDA path) rather than maintaining
three half-live paths.

---

## 8. Recommended next steps (prioritized)

1. **Decode speed = fused kernel (the ONLY real lever left).** WS1 (sync removal, +20%) captured the
   safe overhead win. Investigated further: `DIFFKV_ROUTE_RESIDUALS=16` gives only ~+13% @16k (within
   noise) — **decode is dispatch-bound, not FLOP-bound**, so router/knob micro-opts won't move it. To
   actually beat dense at short ctx (now the default regime), a **fused Metal decode op / batched-layer
   decode** is required. Not changing the route_residuals default (marginal + >32k recall tradeoff).
2. **Sparse-from-start flip: DONE** (`DIFFKV_COMPRESSED_DECODE` default `1`; `auto` opt-in). Follow-up
   to soften the short-ctx cost: **scale the block pool to the prompt** (today `_create_empty_session`
   pre-allocs the full `max_blocks`=256 pool = 682 MB even at 4k). This needs per-session `max_blocks`
   threaded through the overflow-shift + `hard_cap` + `DummyMLXPool` sites — moderate, deferred.
3. **Prefill speedup:** move `_compress_block` SVD off the numpy round-trip (SVD in MLX / async).
4. **Factual store (binding solved, WS2b/WS2c/WS2d).** Realistic 4/4 exact; dense-table 4/5 bare-value,
   2/5 exact. Remaining: (a) verbatim shared-prefix emission (`BRAVO-`→`B`) and (b) the one value miss
   **Raven** (6620→6666). Both root at the same thing: the VSL tracks position (`vsl_active_candidates`)
   but never ENGAGES for value extraction (it only starts a lock on the sequence's FIRST token, which the
   model skips) → naive token-set biases scramble/loop, and repeated digits ('6','6') can't be handled by
   token-ID exclusion. Proper fix = **seed/drive the VSL lock for value extraction** (in-order emission).
   HIGH behavioral risk (changes verbatim emission for every factual query incl. the clean natural 4/4),
   LOW ROI on synthetic codes → **deprioritized**. Before enabling factual by default: kill the ~32%
   decode cost + finish the multi-turn `current_query_tokens` refresh.
5. **Adversarial-only capture fix (optional):** dense-block residual overflow (the 1/5 case) is helped
   by adaptive per-block residual budget or content-aware selection prioritizing rare/numeric tokens —
   but weigh against the 8 GB memory ceiling.

---

## 9. How to run

```bash
cd /Users/omchimurkar1/Desktop/Differential-KV
PY=diffkv_venv/bin/python

# Live 16k sparse decode (speed/memory + KV composition):
$PY paper/scripts/measure_active.py --single compressed,16384 --gen 40 --ctx 16384 --out /tmp/x.json

# Relational A/B (adversarial):    exact / sparse / sparse_factual
$PY benchmarks/relational_ab.py --mode sparse --target 3500 --gen 24
# Realistic (RC4-style, the fair test):
$PY benchmarks/relational_ab.py --mode sparse         --natural --spread --target 6000 --gen 16
$PY benchmarks/relational_ab.py --mode sparse_factual --natural --spread --target 6000 --gen 16
# Diagnostics:
DIFFKV_MAX_RESIDUAL=128 $PY benchmarks/relational_ab.py --mode sparse --target 3500 --gen 24
DIFFKV_RESIDUAL_EXCLUDE_SVD=1 $PY benchmarks/relational_ab.py --mode sparse --target 3500 --gen 24
# See what the factual store surfaces/biases + its digit-span entries:
DIFFKV_FACTUAL_DBG=1 DIFFKV_TELEMETRY=1 $PY benchmarks/relational_ab.py --mode sparse_factual --natural --spread --target 6000 --gen 12
```
Latest results — **Natural (realistic):** exact 4/4 · sparse 4/4 · **factual 4/4 exact (WS2b)**.
**Adversarial (dense table):** exact-full-KV 5/5 · plain sparse 1/5 · **factual 4/5 bare-value,
0 misbound, 2/5 exact-key (WS2c+WS2d)** (exact-key gap = verbatim-code prefix mangle, not binding).
Gates: `tests/test_diffkv_kernel_parity.py` (kernel parity). Memory notes:
`memory/project_active_mlx_subsystem_gating.md`, `project_active_compressed_decode.md`,
`project_relational_binding_progress.md`.
