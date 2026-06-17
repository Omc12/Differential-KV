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
