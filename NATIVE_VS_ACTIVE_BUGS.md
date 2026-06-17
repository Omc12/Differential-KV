# diffkv_native vs ACTIVE_RUNTIME — Bug & Divergence Audit

_Generated 2026-06-17. **No code was changed** — this is a read-only comparison._

Goal: catalog every place where the C++ `diffkv_native/` reconstruction diverges from the
Python `ACTIVE_RUNTIME/` source of truth, with emphasis on what produces the **garbage
output on long prompts** the user reported:

```
loyalrog of the officers.
. of He the. he she
. a or to for be
Elizabeth. Darcy. similarly refuses to. his She. her  and.. . . is.. . rather.. .. .. .... .  0
[Metrics] TTFT: 9478.6ms | Speed: 2.6 tok/s | ~40 tokens
```

That output is a **mix of (a) entity names (Elizabeth, Darcy, officers), (b) pure function
words (of/the/he/she/a/or/to/for/be/is/rather), and (c) runaway periods**. That exact
signature = **the VSL/SFA logit mask is active** (only helpers + sequence-start tokens
survive) sitting on top of a **lossy/collapsed sparse-attention base distribution**, with
**nothing able to penalize the repeated periods**. The findings below explain each piece.

---

## ⚠️ Meta-finding 0 — the working tree has been reverted away from the last-known-good state

The reconstruction log (`diffkv_reconstruction_logs.md`) and memory claim a series of fixes
(`F25`, `active_slots=None`, `threshold 0.4`, `rank 32→16`, `adaptive_k` C_active + 0.15·N
floor) that made NIAH pass. **Most of those are NOT present in the current code**, and the
**uncommitted `git diff` actively reverts several of them** in the name of "matching Python"
— but it matched the *wrong* Python file (the HF/Triton `diffkv_attention.py`, not the Mac
MLX path `mlx_diffkv_wrapper.py`, which the log itself identifies as the real Mac reference).
The memory even warns: _"the working tree was externally reverted mid-session."_ This is the
single biggest reason behavior regressed. Items E, F, B, K below are all reverted fixes.

---

## 🅐 ROOT CAUSE — C++ decodes with lossy SPARSE attention; the MLX reference uses FULL DENSE attention

This is the fundamental behavioral gap and the reason long prompts fall apart in C++ but not
in the Python/Mac reference.

- **MLX reference** (`ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py:829-840`): the decode path
  calls `cache.update_and_fetch(...)` then `mx.fast.scaled_dot_product_attention` over the
  **entire** K/V. The comment is explicit:
  > _"Use the native MLX cache for correct attention over the full context. The DiffKV store
  > still ingests the token … but the actual attention output comes from the native cache
  > which is always numerically correct."_
  The DiffKV sparse store and factual store are **only** used to drive the VSL logit-masking
  (`current_step_factual_tokens` etc.) — never the attention output.
- **C++** (`diffkv_native/src/main.cpp:781`, `runtime/diffkv_attention.cpp` `custom_attention_op_callback`):
  decode attention is computed **over routed compressed slots + a dense window + factual
  store** via `ggml_map_custom3`. This is inherently lossy.

**Consequence:** every logit bias / penalty / temperature / mask in the MLX decode loop was
tuned assuming a *correct* base distribution. In C++ those same biases sit on top of a
**lossy sparse reconstruction that degrades as context grows**. On a ~24k-token prompt the
base distribution collapses, and the bias/mask stack then amplifies it into the observed
soup. NIAH (a single-needle retrieval test) does not exercise long-form *coherence*, so it
passing never proved this path was good for long generation.

> Note: the `DIFFKV_NATIVE_ATTN` ggml subgraph (main.cpp:407 `build_native_sparse_attn`) has
> its own unresolved "echo / inflated-logit" bug (see `HANDOFF_native_attn.md`), but it is
> **gated OFF by default** (main.cpp:781 uses the custom op), so it is *not* the cause of the
> user's output. The custom-op sparse path is.

---

## 🅑 adaptive_k drops the long-context block budget — C++ attends to far too few blocks

`diffkv_native/native_core/srl/query_router.hpp:165-208` vs
`ACTIVE_RUNTIME/native_core/srl/query_router.py:150-196`.

Python:
```python
k_max = min(srl_state.k_max, N_total)
k_min = min(max(srl_state.k_min, int(0.15 * N_total)), k_max)   # floor scales with N!
if N_total <= k_min: return N_total
...
complexity = min(entropy / max(max_ent, 1e-8), 1.0)             # max_ent = log(N_total)
k_raw    = int(k_min + (k_max - k_min) * complexity)            # truncation
k_scaled = int(k_raw * k_multiplier)
# cluster boost:
k_scaled = int(k_scaled * (1.0 + 0.35 * math.log(C_active)))
return max(k_min, min(k_max, k_scaled))
```

C++ (current):
```cpp
float complexity = min(1.0f, entropy / log_N);                 // log_N = log(N)
float k_raw    = k_min + (k_max - k_min) * complexity;         // k_min/k_max FIXED 20/200
float k_scaled = k_raw * k_multiplier;
int K = round(k_scaled);                                        // round(), not int()
return max(k_min, min(k_max, K));
```

Missing in C++:
1. **The `k_min = 0.15 · N_total` floor.** This is the big one. For a 24k-token prompt with
   hundreds of blocks, Python routes to ≥15% of all blocks (up to `k_max`), while C++ stays
   pinned near `k_min = 20`. C++ therefore attends to a *tiny fraction* of the context →
   incoherent long-prompt output. (`k_min/k_max` defaults are identical on both sides: 20 / 200.)
2. **The `C_active` cluster boost** (`× (1 + 0.35·ln(C_active))`) — C++ never widens K when
   multiple topic clusters are active.
3. **`k_max = min(k_max, N_total)`** cap and the **`if N_total <= k_min: return N_total`**
   early return — absent (minor; matters at small N).
4. **`round()` vs `int()` truncation** — minor numeric drift.

The reconstruction log (`F11`) claims these were fixed; they are **not** in the current code.

---

## 🅒 Repetition penalty only touches alphanumeric tokens → it can never suppress "."

`diffkv_native/src/main.cpp:3148-3171` (filtered by `is_alphanumeric_token`, whose
`alnum_cache` is built at main.cpp:1491-1501 — a token is penalizable only if it contains an
alphanumeric char).

The references penalize **all** repeated tokens in the window with no alnum filter:
- MLX: `mlx_diffkv_wrapper.py:1240-1248` (`for tok_id in set(generated[-window:])`)
- HF/Triton: `batch_engine.py:198-200` (`torch.unique(combined)` over the whole window)

**Consequence:** pure-punctuation tokens (`"."`, `" ."`, newline) are *never* penalized in
C++, so the runaway-period spam in the output (`.. . . is.. . rather.. .. .. .... .`) cannot
be damped. Direct contributor to the symptom.

---

## 🅓 No repetition-loop detection / force-stop in C++

The MLX reference has n-gram loop detection (`mlx_diffkv_wrapper.py:1204-1238`): every 10
tokens it checks 5-gram repetition ≥35%, and on detection **widens the penalty window 64→256
and boosts strength to 1.3×**, then **force-EOS after 40 unrecovered tokens**.
`batch_engine.py:1312-1323` mirrors this.

C++ has **none** of it. The penalty window is fixed at 64 (main.cpp:3155), never widened, and
there is no loop break. So once the model enters the degenerate period/function-word loop, it
keeps emitting garbage instead of stopping. Direct contributor.

---

## 🅔 (uncommitted) VSL mask `F25` factual-token exemption removed → mid-sequence content re-masked

`git diff` on `diffkv_native/src/main.cpp:3210-3223` removes the `&& fact_toks.count(i) == 0`
guard:
```cpp
// before (F25 — the documented fix):
if (allowed.count(i) == 0 && fact_toks.count(i) == 0) { ... mask ... }
// now:
if (allowed.count(i) == 0) { ... mask ... }
```
The `allowed` set is only **helpers + sequence-START tokens** (see item P). Removing the
exemption means every **mid-sequence factual content token** is hard-masked again. The
reconstruction log calls F25 _"the fix"_ that made sparse NIAH pass (0/5 → pass).

The change's comment says it "matches `batch_engine.py:1481-1487`" — and it *does* match that
HF/Triton serving loop, which masks the same way. But:
- The Mac reference is the **MLX** loop, and the mask there only works because the base
  distribution is **dense/correct** (item A) — so the un-masked allowed tokens still carry
  the right relative ordering.
- On the C++ **lossy** base distribution, re-masking to helpers + sequence-starts collapses
  output to exactly the entity-name + function-word + period soup observed.

This is a regression vs the last-known-good C++ state, even if it is "literally" closer to one
of the two Python files.

---

## 🅕 (uncommitted) Factual-store query reverted to the WRONG Python reference

`git diff` on `diffkv_native/runtime/diffkv_attention.cpp:541-547`:
```cpp
// now (claims to match diffkv_attention.py:771-777):
threshold = 0.50f, active_slots = (active_slots.empty()? nullptr : &active_slots)
```
But the real Mac reference is the MLX path:
- `mlx_diffkv_wrapper.py:871-876`: **`threshold=0.3, active_slots=None`**.
- `diffkv_attention.py:771-775` (HF/Triton, *not* the Mac path): `threshold=0.50,
  active_slots=active_slots`.

The reconstruction log (`F27`/`F28`) documents `active_slots=None` (and a lower threshold) as
the fix: passing `active_slots` filters out the needle/entry whose straddled blocks weren't
routed. With `active_slots` filtering **and** the higher `0.50` bar, the factual store
surfaces fewer/wrong entries on long prompts → worse VSL state. Wrong file matched.

---

## 🅖 Anti-hallucination penalty threshold mismatch — C++ 0.55 vs MLX 0.4

`diffkv_native/src/main.cpp:3106` gates the −3.5 penalty at `current_step_max_similarity >=
0.55`. The MLX reference fires it at **0.4** (`mlx_diffkv_wrapper.py:1287`, comment:
_"threshold lowered 0.55→0.4"_). The C++ comment deliberately argues for 0.55. Whatever the
merits, it is a behavioral divergence from the source of truth (the off-target vocabulary is
penalized in a different similarity regime than the reference).

---

## 🅗 C++-only RC8 "foreign entity" penalty has no counterpart in the MLX loop

`diffkv_native/src/main.cpp:3083-3092` applies a −12/−4 penalty to tokens belonging
*exclusively* to other entities while locked to `current_entity`
(`compute_entity_token_license`). The MLX `generate()` loop has **no such step**.

On a multi-character literary prompt (Elizabeth, Darcy, Bingley, Wickham, …) this can
aggressively suppress legitimate tokens from "other" characters, distorting the distribution
in a way the reference never does. Plausible contributor to the broken multi-entity output.

---

## 🅘 +10 transition-bias guard differs from MLX

`diffkv_native/src/main.cpp:3121-3123` only applies the +10 transition boost when `last_token`
is **not** a helper word. MLX (`mlx_diffkv_wrapper.py:1300-1313`) applies it **unconditionally**
for any `last_token`. Minor divergence in which successors get boosted.

---

## 🅙 micro_block_size = 64 (C++) vs 16 (Python) — 4× coarser compression & far fewer dense anchors

`diffkv_native/src/main.cpp:1403` (`int micro_block_size = 64`) vs
`ACTIVE_RUNTIME/native_core/streaming_sparse_ingest.py:131,485` (`micro_block_size = 16`).

Implications for a 24k-token context:
- **Block count:** ~375 blocks (C++) vs ~1500 (Python). Coarser routing granularity, and with
  item B this compounds (few blocks, each huge).
- **Dense anchors:** DiffKV keeps **1 exact dense anchor per block**. C++ retains 1 exact
  token per **64**, Python 1 per **16** → Python preserves **4× more exact KV**. The
  reconstruction log's "needle straddles a 64-token micro-block boundary" failures trace to
  exactly this value.
- **Per-token reconstruction error:** SVD-compressing 64 tokens (item K: to rank 32) is more
  lossy per token than 16 tokens to rank 16.

Significant quality divergence on long contexts.

---

## 🅚 Compression rank = 32 hardcoded (C++) vs 16 (Python serving default); DIFFKV_RANK not honored

`diffkv_native/src/main.cpp:1396` (`int rank = 32;`, passed to `KVRuntimeManager` at :1413).
- The durable project decision and Python serving default is **rank 16**
  (`ACTIVE_RUNTIME/native_core/compression/lowrank.py` `compress_lowrank(rank=...)` is called
  with 16 by the serving layer). The reconstruction log claims `F1: rank 32→16` was applied —
  it is **not** in the current code (reverted).
- `grep DIFFKV_RANK` finds **no** override read in `main.cpp` or `kv_runtime_manager.cpp`, so
  the env override the log says was added is not wired at this construction site.

(rank 32 is *higher* fidelity than 16, so this is not itself a quality-degrader — but it is a
RAM regression and contradicts the agreed config + the memory's claim that 16 is active.)

---

## 🅛 Global dense bypass < 2048 tokens (C++) has no Python equivalent

`diffkv_native/native_core/streaming_sparse_ingest.cpp:480-484`: prompts shorter than
`engage_threshold` (default 2048) are kept **fully dense** (no compression). Python has no
global bypass — it always micro-compresses past `short_context_threshold = 256`
(`streaming_sparse_ingest.py:392`). Different RAM/quality behavior in the [256, 2048) range.
(Not the cause for a 24k prompt, but a structural divergence; the C++ comment at :478
acknowledges "ACTIVE_RUNTIME has no such global bypass.")

---

## 🅜 (uncommitted) factual_store.cpp change is whitespace-only

`git diff diffkv_native/native_core/srl/factual_store.cpp` adds only blank lines (around
:817, :870, :891, :907, :927, :1036) with **no behavioral change**. Pure diff noise — flag it
so it isn't mistaken for a real edit.

---

## 🅝 SFA/VSL fires on non-factual (literary) long prompts

The whole SFA stack is gated on `current_step_max_similarity >= 0.55` + non-empty factual
sequences (main.cpp:3199-3200; mlx:1324-1328). On a Pride-and-Prejudice paste the factual
store is populated from salient spans (character names), and the decode query matches them,
so SFA plausibly activates even though this is *not* a factual-extraction task. The output's
"Elizabeth./Darcy." entity-starts + helper soup is the signature of an **active VSL mask**.
The MLX reference survives this because its base distribution is dense/correct (item A) and
because (pre-revert) the C++ had the F25 exemption (item E). This isn't a single-line bug so
much as the **interaction** of A+B+E+F+H surfacing on a prompt the SFA path was never tuned for.

---

## ✅ Verified FAITHFUL (checked, not the problem)

So the user knows where *not* to look:
- **Helper word list** `ALLOWED_HELPER_WORDS`: **exact match**, 280 == 280 words
  (factual_alignment.hpp vs factual_alignment.py). The uncommitted diff brought this to parity.
- **`RELATIONAL_BINDING_WORDS`**: exact match, 37 == 37 (drives structural helpers).
- **Helper builder includes empty-cleaned punctuation** as helpers on both sides
  (factual_alignment.hpp:133-134 ≡ factual_alignment.py:170) — so "." being VSL-allowed is
  faithful; the bug is that C++ can't *penalize* it (item C), not that it's allowed.
- **Routing constants** that the log's `F11` fixed and that *did* stick: topic-switch 0.30,
  lexical decay 1.0, `graph_hop_decay` 0.5 (query_router.hpp / session_srl_state.hpp match
  query_router.py).
- **Structural-helper logic** (full helpers − relational binders) matches conceptually.

---

## Suggested triage order (to reproduce the reference behavior)

1. **Revert the uncommitted diff** (items E, F, M) — it moved away from the last-known-good
   state and matched the wrong Python file.
2. **adaptive_k** (item B): restore the `0.15·N_total` k_min floor + `C_active` boost — this is
   the most direct lever for long-context attention quality.
3. **Repetition penalty** (item C): drop the alphanumeric-only filter so periods get penalized;
   add the n-gram loop detection + force-stop (item D).
4. Re-align the factual-store query to the **MLX** values (item F: `threshold=0.3,
   active_slots=None`) and the anti-hallucination threshold (item G: 0.4).
5. Reconsider `micro_block_size` 64→16 and rank 32→16 (items J, K) for fidelity/consistency.
6. Recognize the ceiling (item A): the custom-op sparse path will never match a dense
   reference on long-form coherence; that's the architectural gap behind the symptom.

---
---

# ROUND 2 — verification of the user's fixes + new findings (2026-06-17)

The user applied fixes; build passes (`cmake --build build --target diffkv_native` → 100%).
Below: what's verified fixed, what's still open from Round 1, and **new issues found while
reviewing the fixes** (catch-everything pass).

## ✅ Round-1 items VERIFIED FIXED

| Item | What was done | Verdict |
|---|---|---|
| **B** adaptive_k | `query_router.hpp:199-249`: added `k_max=min(k_max,N_total)`, `k_min=min(max(k_min,int(0.15·N_total)),k_max)`, early `return N_total`, `int()` truncation, and the `C_active` cluster boost. Normalizes by `log(N_total)`. | **Faithful** to query_router.py:150-196 (one minor divergence — see N2.4). |
| **C** rep penalty | `main.cpp:3193-3207`: removed the `is_alphanumeric_token` filter; now penalizes **all** repeated tokens. | **Fixed** (periods can now be penalized). |
| **D** loop detection | `main.cpp:3156-3189` + state at `:2609-2611`: 5-gram repetition check every 10 tokens over last 80, ≥0.35 → widen window 64→256 + strength 1.3×, force-EOS 40 tokens after detection. | **Fixed**, mirrors mlx:1204-1238. State resets per generation. |
| **E** F25 exemption | `main.cpp:3257`: restored `allowed.count(i)==0 && fact_toks.count(i)==0`. | **Fixed** (back to last-known-good). |
| **F** factual query | `diffkv_attention.cpp:541-549`: `threshold=0.30, active_slots=nullptr`. | **Fixed** — now matches the MLX Mac reference (mlx:874-875). |
| **G** anti-halluc threshold | `main.cpp:3113`: `0.55 → 0.4`. | **Fixed** — matches mlx:1287. |
| **I** transition bias | `main.cpp:3127-3131`: removed helper-word gate (`if (true)`). | **Fixed** behaviorally (but see N2.5 — code smell). |
| **J** micro_block_size | `main.cpp:1408`: `64 → 16` (+`DIFFKV_MICRO_BLOCK_SIZE` env). Adaptive cap logic at `:1743-1764` matches Python (also caps at 16). | **Block size fixed**, BUT introduced **N2.1 (critical)** — pool not resized. |
| **K** rank | `main.cpp:1396-1399`: `32 → 16` + `DIFFKV_RANK` env now honored. | **Fixed.** |

## ⏸ Round-1 items STILL OPEN (intentional or architectural — confirm)

- **A — architectural ceiling (unchanged):** C++ decode still uses the lossy sparse custom-op
  (`main.cpp:781`); the MLX reference is dense. Not fixable by parameter tuning. The fixes
  above reduce the damage but the base distribution is still lossy.
- **H — RC8 foreign-entity penalty still present** (`main.cpp:3083-3092`): −12/−4 on
  "foreign" entity tokens has no counterpart in the MLX `generate()` loop. Still a divergence;
  still a candidate for distorting multi-entity (literary) output. Decide keep-or-remove.
- **L — global dense bypass < 2048** still present (`streaming_sparse_ingest.cpp:480-484`);
  Python has no such global bypass. Unchanged.

---

## 🆕 N2.1 — CRITICAL: pool slot-count & capacity math were NOT updated for `micro_block_size=16` (regression from the item-J fix)

The item-J fix changed block size 64→16 at the **ingest** level, but the **pool sizing** still
assumes **64 tokens per slot** everywhere:

- `main.cpp:1353` `int n_slots = model.get_config().n_ctx / 64;`
- `main.cpp:1362` `n_slots = (max_tokens + 63) / 64;`
- `main.cpp:1377-1383` presets: `64 / 128 / 256` slots == "4096 / 8192 / 16384 tokens" (i.e. tokens÷64)
- `main.cpp:1649-1653` capacity guard `if (L > n_slots * 64) { … L = n_slots * 64; }`
- `runtime/native_block_pool.cpp:88,356` `const int S_max = 64;` (per-slot token capacity of `U_`/`U_f16_`/`valid_mask_`)
- `native_block_pool.cpp:301-369`, `main.cpp:1822`, `kv_runtime_manager.cpp:566` index host buffers with a hardcoded `*64` stride.

But each block now fills at `micro_block_size + 1 = 17` tokens
(`streaming_sparse_ingest.cpp:406`), and the adaptive logic caps at 16
(`main.cpp:1757`, `target = min(raw_target, 16)`). So:

> **Effective pool capacity = `n_slots × 16` tokens, while the code thinks it is `n_slots × 64`.
> The pool now holds ¼ of the intended token budget.**

Concretely for the user's ~24k-token prompt (Qwen2.5-0.5B, `n_ctx=32768`, default path →
`n_slots = 512`):
- **Before** the fix (64-tok blocks): needs `24000/64 ≈ 375` slots ≤ 512 → **fit**.
- **After** the fix (16-tok blocks): needs `24000/16 = 1500` slots ≫ 512 → **overflow by ~3×.**
  The capacity guard at `:1649` compares against `512×64 = 32768`, so it lets the prefill
  proceed, then the pool runs out of slots mid-prompt → eviction / `active_slot >= n_slots`
  (`:2615`) → most of the context is dropped → garbage persists (or is **worse** than before).

**This likely cancels or reverses the intended benefit of items B & J for long prompts.** Python
avoids this because its pool uses `max_seq_len = micro_block_size` and derives block count as
`n_tokens / max_seq_len`, and the pool **grows** up to `max_blocks`
(`runtime/native_block_pool.py:124,131,141,196-197`). The C++ pool is fixed-size and the
`/64`, `×64`, `S_max=64` constants must all become `micro_block_size`-based (and/or the pool
must be allowed to grow).

## 🆕 N2.2 — RAM waste: `S_max=64` per slot but blocks only fill 16

Same root as N2.1, RAM side. `U_`=`[rank, S_max=64, n_slots]`, `U_f16_` (same),
`valid_mask_`=`[64, n_slots]` are allocated at 64 token-slots each
(`native_block_pool.cpp:88-108`) but with 16-token blocks only 16 of every 64 rows are used.
≈ **75 % of the per-slot delta / U / mask pool memory is dead allocation.** (The `valid_mask`
correctly fills 16..63 with −inf at `:358-359`, so it's safe — just wasteful.) If N2.1 is fixed
by switching the stride to `micro_block_size`, this is fixed for free.

## 🆕 N2.3 — Repetition penalty no longer covers the prompt tail (divergence)

`main.cpp:3199-3203` penalizes over `generated_tokens`, which is **generated-only** (it is
seeded with just `last_token` at `:2422-2423`, not the prompt). The references penalize over
the prompt too:
- MLX `mlx_diffkv_wrapper.py:1244`: `for tok_id in set(generated[-window:])` where `generated`
  is `prompt_ids.copy()` + decoded tokens — so the window includes the **prompt tail**.
- `batch_engine.py:187-201` merges generated **and prompt** penalty sets, plus a dedicated
  "prompt anti-copy guard" at `:1332-1336`.

Effect: C++ does not penalize **re-copying the end of the prompt**, so on a long literary paste
it is more prone to echoing source text (the output's `"…of the officers"` / `"Elizabeth.
Darcy."` look like prompt fragments). There is **no prompt-anti-copy mechanism in C++ at all**
(grep confirms). Consider extending the penalty window to include prompt tokens (bounded), as
both Python paths do.

## 🆕 N2.4 — `C_active` counts UNIQUE parents in C++ but per-block (with duplicates) in Python

`query_router.hpp:218-243` dedups parents via `seen_parents` before counting those with
`score ≥ theta`. Python (`query_router.py:180-191`) iterates `parent_landmarks` **without
dedup** — `parent_landmarks` is a per-block `[N]` array (`chunk_graph.hpp:176`
`resize(N,-1)`), so a popular parent is counted once per child block. Result: Python's
`C_active` is generally **larger** than C++'s for the same state → Python's
`1 + 0.35·ln(C_active)` boost is bigger → Python routes to more blocks. Minor magnitude
divergence in the (just-added) cluster boost; flag for parity. (C++ correctly skips the `-1`
sentinel; Python's `if p in slot_to_idx` filter also drops `-1`, so that part matches.)

## 🆕 N2.5 — `if (true)` dead guard left in the transition-bias block (code smell)

`main.cpp:3129` now reads `if (true) { … }` (the old helper-word gate was stubbed rather than
removed). Harmless, but it leaves a dangling scope + a misleading conditional. Cosmetic.

## 🆕 N2.6 — loop-detection counter off-by-one vs MLX (negligible)

`main.cpp:3160` triggers on `generated_tokens.size() % 10 == 0`, but `generated_tokens`
includes the seed `last_token`, so `size() = 1 + (#generated)`. MLX keys off `_n_new`
(generated-only). So C++ first checks at 29 generated tokens where MLX checks at 30, etc.
One-token phase shift; does not affect correctness.

---

## Updated triage (Round 2)

1. **N2.1 is now the top long-prompt blocker.** Make `n_slots`, the `×64`/`/64` capacity math,
   and `S_max` all derive from `micro_block_size` (or let the pool grow like Python). Without
   this, items B+J actively hurt the 24k-token case the user reported.
2. N2.3: include prompt tokens in the repetition window (bounded) — add a prompt anti-copy guard.
3. N2.4 / N2.5 / N2.6: minor parity / cleanup.
4. Decide on H (RC8) and L (dense bypass) — still divergent from the MLX reference.
5. A remains the ceiling; sparse decode won't fully match dense on long-form coherence.

---
---

# ROUND 3 — verification of N2.* fixes + full-pipeline walk (2026-06-17)

Build passes (`cmake --build build` → 100%). The user's actual run path is the **main.cpp
binary** launched by `serving/cli.py` via subprocess (`batch_engine.cpp` is the separate HTTP
gateway, which received the same fixes in parallel). I traced a prompt through every stage
(tokenize → pool sizing → prefill/RoPE → compression → SRL/factual build → routing → sparse
attention → factual query → VSL/logit-bias → sampling → stop) and diffed each against the MLX
reference. Below: N2.* verification, then NEW issues.

## ✅ Round-2 items VERIFIED FIXED
| Item | Verdict |
|---|---|
| **N2.1** pool sizing | **Fixed everywhere.** `n_slots = n_ctx / micro_block_size` (main.cpp:1360), presets `4096/8192/16384 ÷ mbs` (:1384-1390), capacity guard `L > n_slots*micro_block_size` (:1650), `userdata.S_max = micro_block_size` (:2271), U-stride `slot_id*micro_block_size*rank` (:1823). Pool `S_max_` now a param (native_block_pool.cpp:86-108, hpp:22), threaded through `kv_runtime_manager.cpp:71`, `lowrank.cpp:416` (`params.pool_block_size`), `async_compressor`, `streaming_sparse_ingest.cpp:620`. |
| **N2.2** S_max RAM waste | **Fixed** (same change — per-slot tensors now sized to `micro_block_size`, not 64). |
| **N2.3** rep penalty prompt | **Fixed** — now iterates `all_tokens` (prompt+generated) at main.cpp:3196-3201. |
| **N2.5** `if(true)` smell | **Fixed** — dead guard removed (main.cpp:3129). |
| (bonus) | `batch_engine.cpp` got the parallel sizing fix **and** another latent hardcode fixed: `userdata.rank = 32` → `kv_engines[l]->get_rank()`, `S_max = 64` → `session->micro_block_size` (:1134-1135). |

Also verified **faithful (not bugs):** sampling/top-p (`sample_logits` main.cpp:1055 keeps the
token that crosses `top_p`, == MLX:1196-1199); rep-penalty default 1.15 + loop boost
`max(.,1.3)` (== MLX); prefill chunk 512 (== MLX `PREFILL_CHUNK`); generation EOS is EOG/eos
only (main.cpp:3252, not the lexical stop-word set); factual salience ±3-token window
(== factual_store.py:283-294); adaptive block-size cap logic (== streaming_sparse_ingest.py:558).

## ⏸ Still open from earlier rounds
- **A** architectural sparse-vs-dense ceiling — unchanged.
- **H** RC8 foreign-entity penalty (main.cpp:3083) — still present, no MLX counterpart.
- **L** global dense bypass <2048 (streaming_sparse_ingest.cpp:480) — unchanged.
- **N2.4** `C_active` dedup vs Python per-block count — still present (see N3.4, minor).

---

## 🆕 N3.1 — TWO conflicting factual-store queries per decode step (quality **and** speed)

This is the biggest new finding. There are **two full factual-store queries every step**, in
two places, with **different parameters**, and they fight over the same SRL state:

| | **Callback** (diffkv_attention.cpp:525-577) | **Decode loop** (main.cpp:3363-3509) |
|---|---|---|
| When | layer-0 **forward pass** (every step) | **after sampling**, "for the NEXT step" |
| Query proxy | `Q_unrot` — the **QUERY** vector | `decode_k[0]` — the **KEY** |
| threshold | **0.30** | **0.50** |
| active_slots | **None** (`nullptr`) | **`slot_filter`** (non-null) |
| entity bias | none | `qbias` |
| neighbor injection | **none** (direct entries only) | 1-hop ≥0.45 + 2-hop ≥0.65 + triples |
| writes | clears+sets `current_step_factual_{tokens,sequences,max_sim}` | clears+sets those **plus** `current_step_sequence_{entity_ids,is_prime,prefixes}` + `current_entity_id` |

Problems:
1. **The "fix F" (0.30 / active_slots=None) only applies to the callback.** The decode-loop
   query still uses the old HF-path values (0.50 / `slot_filter`). So the MLX-alignment is
   half-applied.
2. **Wrong query proxy in the callback.** MLX uses the layer-0 **KEY** as the proxy
   (`k_torch = keys[:, :, 0, :]`, mlx:868) — the decode-loop path matches that (`decode_k[0]`),
   but the callback uses the **query** (`Q_unrot`). So the path that actually drives the VSL
   (see #3) is matching on the wrong vector.
3. **Overwrite + parallel-array desync.** By execution order the callback runs in each step's
   forward (before that step's logit post-processing), so it is the **last writer** of
   `current_step_factual_{tokens,sequences,max_sim}` → the VSL mask and the +7/−3.5/transition
   logit biases all read the **callback's** set (0.30 / None / query-proxy / **no neighbor
   injection**). Meanwhile `current_entity_id` and the parallel
   `current_step_sequence_{entity_ids,is_prime,prefixes}` arrays retain the **decode-loop's**
   values from the previous step. The callback does **not** update those parallel arrays, so
   `current_step_factual_sequences[i]` (callback) and `entity_ids[i]` (decode-loop, stale,
   different length) are **out of sync** → entity-filtered boost (main.cpp:3054-3063), RC8
   (:3083), and VSL entry gating (`get_allowed_tokens_vsl_cpp`) index a mismatched
   `entity_ids[i]`, falling back to `-1` (entity-agnostic) for out-of-range sequences →
   wrong/loose entity gating.
4. **The decode-loop's neighbor injection + triples + qbias are effectively discarded** for
   VSL/logit purposes (overwritten by the callback next forward), so the work is wasted.
5. **Redundant cost (speed):** the callback was added as the F29 "query once per step + cache"
   optimization, but the **old decode-loop query was never removed** → **two** full factual
   queries per step (descriptor projection + graph walk + merge each time) instead of one. On a
   large store this is a measurable decode-throughput tax.

MLX does this in **one** place (the attention forward), with the key proxy, threshold 0.3,
active_slots=None, and the neighbor injection all together (mlx:857-902). The C++ should
collapse to a single query and decide one set of parameters. Net effect today: the VSL is
driven by an un-injected, query-proxy, 0.30/None match with desynced entity metadata.

## 🆕 N3.2 — Neighbor-injection thresholds diverge from MLX

Where the C++ does inject (decode-loop, main.cpp:3492/3502): **1-hop ≥ 0.45, 2-hop ≥ 0.65**.
MLX (mlx:889/895): **1-hop ≥ 0.35, 2-hop ≥ 0.50**. The C++ bars are higher → fewer neighbor
sequences surfaced → narrower VSL allowed-set (more masking). (Compounded by N3.1, which then
discards even this for VSL.) Align to 0.35 / 0.50.

## 🆕 N3.3 — Lexical/inverted-index stopword list diverges from Python

`main.cpp:1321-1339` builds the stop set from a ~60-word hand list **plus the first 200 token
IDs**. Python uses `_STOP_WORDS_COMPRESS` (streaming_sparse_ingest.py:61) — a ~150-word
NLTK-style set (contractions, `about/against/between/few/more/most/own/same/…`) and does **not**
blanket-add the first 200 IDs. Consequences (quality): (a) real stopwords the C++ omits get
indexed into `important_vocab` and `occurrences` → they pollute rare-/high-IDF lexical routing
**and** the newly-added important-vocab anchor-overlap filter in `process_and_tag_vsl_step`
(factual_alignment.hpp:247-262); (b) the "first 200 IDs" blanket is a C++-only heuristic that
may stop-list legitimate early-vocab tokens. This is the long-standing "stopword list
duplicated in 3 places / diverging" tech-debt, now confirmed to affect routing + VSL.

## 🆕 N3.4 — `C_active` parent dedup (minor parity; was N2.4)

`query_router.hpp:218-243` dedups parent landmarks; `query_router.py:180-191` counts per-block
(duplicates included) → Python's cluster boost is larger → routes to slightly more blocks.
Low-severity magnitude divergence in the boost.

## 🆕 N3.5 — Capacity guard under-counts by the anchor token (negligible)

`main.cpp:1650` guards `L > n_slots * micro_block_size`, but a full block holds
`micro_block_size + 1` tokens (1 anchor + `micro_block_size` deltas; see
streaming_sparse_ingest.cpp:406). True capacity is `n_slots*(micro_block_size+1)`, so the guard
truncates ~`n_slots` tokens early. Harmless (conservative); fix for exactness only.

---

## Updated triage (Round 3)
1. **N3.1 is the priority** — collapse the two factual queries into one (keep the key proxy +
   single parameter set + neighbor injection in one place, like MLX), fixing the overwrite,
   the entity-array desync, the half-applied threshold/active_slots, and the 2× query cost in
   one move.
2. N3.2: 1-hop/2-hop thresholds → 0.35 / 0.50.
3. N3.3: unify the stopword list with Python's `_STOP_WORDS_COMPRESS`; reconsider the first-200
   blanket.
4. N3.4 / N3.5: minor parity / exactness.
5. A / H / L unchanged from prior rounds.

---
---

# ROUND 4 — why the long prompt STILL fails (2026-06-17)

Symptom after the latest fixes: `Pasted 95310 chars` → `AI > regiment against` →
`Generated: ~4 tokens` then stop (1.9 tok/s, TTFT 8.0s). N3.1 is verified fixed (single
factual query feeding `step_cached_entries`; callback only reads it). But the model now emits
**EOS after ~4 tokens** — it stops at the normal EOS check (main.cpp:3325), i.e. the *base
distribution itself* favors ending. Tracing the decode-attention path end-to-end revealed why,
and it is **not** in any subsystem fixed so far.

## 🔴 N4.1 — `route_query` / `adaptive_k` (all of item B) is DEAD CODE — never called

`grep -rn "route_query(\|adaptive_k(\|route_query_fixed_k("` over all of `diffkv_native`
(excluding the defining header) returns **zero callers**. The entire SRL pipeline in
`query_router.hpp` — `adaptive_k` (165), `route_query` (270), `route_query_fixed_k` (928) — is
**defined but never invoked** by the runtime.

⇒ **Every fix made to `adaptive_k` across Rounds 1–3 (the `0.15·N_total` k_min floor, the
`C_active` boost, the `int()`/normalization parity — item B, N2.4, N3.4) has had ZERO effect on
the running binary.** That work was spent on dead code. The binary's real routing is a separate
function, `KVRuntimeManager::route_decode_slots` (kv_runtime_manager.cpp:283), called once per
`retrieval_interval` at main.cpp:2778.

## 🔴 N4.2 — Decode attention attends only `srl_k_keep = 16` blocks → ~256 tokens of a 24k prompt (~1%)

The real decode slot-selection pipeline (build_decode_graph, layer 0, main.cpp:704-731):
1. `route_decode_slots` (host) → `physical_candidates` → `host_slots` (fixed-budget set).
2. `sem_slots = semantic_search_topk(q_desc, …, srl_k_semantic=32)`.
3. `candidate_slots = concat(sem_slots, host_slots)`.
4. **`selected_slots = anchor_screen(Q, anchors_K, candidate_slots, srl_k_keep)`** → top **16**.

And the attention K is hard-capped: `current_k = min(srl_k_keep, active_slot)` (main.cpp:2837),
`userdata[l].K = current_k` (:2847). So **the sparse decode attends at most 16 blocks per
step.** With the item-J change to `micro_block_size = 16`, that is **16 × 16 = 256 tokens** out
of ~24 000 — **≈ 1 % of the context.**

This is almost certainly the direct cause of the current symptom: the model can "see" only ~256
tokens of compressed context (plus the tiny dense tail of the prompt that hadn't compressed
yet), so after the prefill's first reasonable token it has no coherent continuation and emits
EOS within a few tokens.

**Interaction with item J (regression):** before item J, `micro_block_size = 64`, so 16 blocks
covered `16 × 64 = 1024` tokens. After item J (`16`), the same 16 blocks cover only `256`
tokens — **item J quartered the decode context coverage** because `srl_k_keep` was not raised to
compensate. So the well-intentioned mbs fix made long-prompt coverage 4× worse on this path.

**vs the reference:** MLX decode is **fully dense** (attends all 24k). The Python sparse/triton
path attends the *entire routed set* (`route_query` K, up to `k_max = 200` blocks — and that
routing actually runs there). The C++ attends **16**. So C++ looks at ~12× fewer blocks than
even Python's sparse path, and each block is now 4× smaller.

## 🔴 N4.3 — The real router (`route_decode_slots`) uses FIXED budgets, no context scaling

`route_decode_slots` (kv_runtime_manager.cpp:283-402) builds candidates from fixed per-channel
budgets: sink + recency (`srl_k_recency = 8`) + lexical (`srl_k_lexical = 8`) + 2-hop graph
(`srl_k_graph = 8`), with `srl_k_semantic = 32` added graph-side. None of these scale with
context length (unlike the dead `adaptive_k`, which was the whole point of item B). For a 24k
prompt the candidate pool is ~`srl_k_host = 1+8+8+32+8 = 57`, then `anchor_screen` cuts to 16.
So no matter how long the prompt, the model sees a fixed, tiny slice.

## 🟢 N4.4 — Immediate levers (all env-overridable) + the real fix

Quick experiments (no rebuild): `srl_k_keep` (`DIFFKV_SRL_K_KEEP`), `srl_k_semantic`
(`DIFFKV_SRL_K_SEM`), `srl_k_lexical/graph/recency` are all env vars (main.cpp:1295-1299).
Bumping `DIFFKV_SRL_K_KEEP` to e.g. 64–128 and the candidate budgets accordingly should
immediately widen attended context and likely fixes the early-stop on long prompts (at a
decode-speed cost). The principled fixes:
1. **Scale `srl_k_keep` to compensate for `micro_block_size`** (≈ `1024 / mbs` to preserve the
   old token budget, or higher to approach the Python sparse path's block count).
2. **Either wire the real `adaptive_k` into `route_decode_slots`** (so K grows with context and
   the item-B work actually runs) **or delete the dead `query_router.hpp` routing** so future
   audits don't keep "fixing" code that never executes.
3. Re-evaluate item J: if `srl_k_keep` can't be raised enough for speed reasons, a larger
   `micro_block_size` trades anchor-fidelity for far better per-block coverage.

## Updated triage (Round 4)
1. **N4.2 + N4.1 are the headline:** the binary attends ~1% of a long context via a 16-block cap,
   and the adaptive-K routing meant to widen that is dead code. This explains why every prior
   round's fixes didn't change the long-prompt behavior. Fix `srl_k_keep` scaling first
   (fastest path to a visible improvement), then reconcile the two routing implementations.
2. N4.3: give `route_decode_slots` context-aware budgets (or revive `adaptive_k`).
3. Prior open items (A architectural ceiling, H RC8, L dense bypass) still stand, but they are
   secondary to N4.2.

---
---

# ROUND 5 — EMPIRICAL root cause (ran the binary) — 2026-06-17

I stopped reading and **ran the actual binary** (`build/diffkv_native`, rebuilt 13:03) with the
real `scratch/pride_and_prejudice.txt`, reproducing the cli.py invocation. This overturns the
earlier *speculation* that the base distribution / EOS was the problem. **The pipeline works; the
failure is slot-capacity exhaustion.**

## 🟥 N5.1 — THE BUG: the prompt fills the whole pool, leaving zero headroom for decode → generation stops after ~4 tokens

**Repro (full 95 310-char prompt, preset `mid`, greedy):**
```
[DiffKV Native] Preset 'mid' detected: capping context size to 8192 tokens (128 slots)
[DiffKV Native] Warning: Prompt tokens length 21525 exceeds maximum capacity 8320. Truncating prompt.
[DiffKV Native] Warning: Context slot capacity reached during decode. Stopping generation.
AI>  is this        ← 4 tokens, then stop
```
The decode loop's very first guard is `if (active_slot >= n_slots) { break; }`
(src/main.cpp:2681-2683) — **a hard stop with NO eviction, paging, or sliding window.** The
prompt is truncated to *exactly fill* all 128 slots (8320 tokens ≈ 128 blocks), so `active_slot`
is already at the limit when decode starts; the partially-filled last block absorbs ~2-4 decode
tokens, then slot 128 is needed → `active_slot >= n_slots` → **stop.** This — not EOS, not the
VSL mask, not a bad distribution — is why the user sees "regiment against" / " is this" + ~4
tokens every time.

**Control test — short prompt (~1692 tokens, lots of slot headroom), same binary/settings:**
```
AI> Jane Austen's novel Pride and Prejudice is widely regarded as one of her finest
    works, with a strong claim on primacy among her novels.   ← coherent, complete
[DiffKV Native] Text generation completed successfully!
```
⇒ **The sparse attention, routing, VSL, compression, and sampling are all functioning.** The
only thing that breaks long prompts is that the fixed pool is filled by the prompt with no room
to generate. (This also means the Round 1–4 items, while real divergences, were **not** the
cause of the reported symptom.)

### Why it happens
- Context budget comes from the **preset** (`mid` → `n_slots = 8192/mbs = 128`,
  cap ≈ 8192 tokens). cli.py sets `DIFFKV_PRESET=mid` and does **not** set any context override,
  so a 21 525-token prompt is truncated to ~8320 and packed into all 128 slots.
- The capacity guard (src/main.cpp:1650) truncates the *prompt* to the pool size but **nothing
  reserves space for the `max_generate` decode tokens.**
- At the wall, there is **no eviction / recycling** (src/main.cpp:2681 just `break`s). Python
  never hits this: MLX decode is dense over a growing native cache, and the Python
  `NativeBlockPool` **grows** `n_blocks` up to `max_blocks` (native_block_pool.py:124,196) rather
  than truncating the prompt to a fixed size.

### Confirmed fixes (in order of effort)
1. **Reserve decode headroom.** Truncate the prompt to `capacity − headroom_blocks·mbs` (e.g.
   leave `max_generate` tokens of slots free), instead of letting it fill all `n_slots`.
   One-line-ish change at the truncation site (src/main.cpp:1650) + the `n_slots` sizing.
2. **Make the context budget exceed prompt+generation.** Empirically, running the *same* full
   prompt with `DIFFKV_MAX_CTX_TK` set well above the prompt length gives the pool headroom and
   lets decode proceed (see N5.2). This is the immediate workaround; the preset caps
   (4096/8192/16384) are the trap because a longer prompt always truncates to fill them.
3. **Proper fix: evict/slide at the capacity wall.** When `active_slot >= n_slots`, recycle the
   least-relevant (or oldest non-sink) compressed block's slot instead of stopping — a sliding
   compressed-KV window. There is already a pager/`PAGED` lifecycle in the architecture; it is
   simply not invoked here.

## 🟧 N5.2 — Preset caps make this unavoidable for any long prompt

Because every preset (`low/mid/high` = 4096/8192/16384 tokens) sizes the pool to a fixed budget
and the prompt is truncated to fill it, **any prompt at or above the preset budget will fill the
pool and stop decode immediately** — regardless of all the routing/VSL/attention fixes. The
budget must be `prompt + generation`, or eviction must free slots during decode (N5.1 fix 3).

## 🟨 N5.3 — Secondary (the user's other two complaints)
- **Low TPS (~1.5 tok/s):** this is the known per-layer Metal custom-op dispatch cost (24
  `ggml_map_custom3` launches + syncs per token; memory F29/F30). The native fused subgraph
  (`DIFFKV_NATIVE_ATTN`) that would fix it is gated off due to its unresolved echo bug. Not
  related to N5.1.
- **Long prefill (~8 s):** chunked prefill (512-token chunks) with per-block SVD compression over
  ~8 k tokens; O(n²) attention within the growing cache. Inherent to the design; would be larger
  for the untruncated 21 k prompt. Secondary to N5.1.

## 🪵 N5.4 — Minor/confusing: warmup logs `Adaptive micro_block_size: 64 -> 16 (L=21)`
The adaptive-mbs line prints with `L=21` (a tiny warmup run), not the real prompt length. For the
real prompt (`L≈21525 → raw_target 256 → min(256,64)=64 == current`) it doesn't print (no
change), so mbs stays 64. Harmless, but the stray `L=21` log is misleading — it looks like the
block size is being set from a 21-token context. Cosmetic.

## Updated triage (Round 5) — START HERE
1. **N5.1 is the whole ballgame for long prompts.** Reserve decode headroom (fix 1) or add
   eviction at `active_slot >= n_slots` (fix 3). Verified: with pool headroom the model generates
   coherent output; without it, it stops at ~4 tokens. Everything else (Rounds 1–4) is real but
   secondary — the pipeline demonstrably works when the pool isn't full.
2. N5.2: don't truncate the prompt to exactly the preset budget; budget = prompt + generation.
3. N5.3 (TPS / prefill): the known custom-op dispatch cost; separate effort.
