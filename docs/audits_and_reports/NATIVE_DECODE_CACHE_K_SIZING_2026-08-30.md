# The native decode cache is sized off the context WINDOW, not the blocks that exist

**Status: APPLIED and validated on Apple silicon, 2026-08-30.** See *What
shipped* at the end — including one correction to the fix proposed below, which
was wrong in a way this document's own grid could not show.

**Date:** 2026-08-30
**Files:** `dkv_native/src/main.cpp`, `dkv_native/serving/batch_engine.cpp`
**Reproduce:** `tools/native_k_pipeline/` (two standalone C++ transcriptions,
`g++ -O2 -std=c++17`, no ggml/model/GPU needed)

## The defect

`srl_k_keep` sizes the routed half of the decode cache:

```cpp
// main.cpp:4607
const int cache_routed_cap = srl_k_keep * (micro_block_size + 1);
```

Three gates can raise `srl_k_keep` after prefill, and **all three are raise-only** —
nothing ever clamps it back down to the number of compressed blocks that actually
exist:

| line | gate | keyed off |
|---|---|---|
| 4446 | adaptive-k | `max(20, 0.15 × n_comp_blocks)` — a hard floor of **20** |
| 4469 | short-context dense fallback | `n_slots` ← **the bug** |
| 4484 | short-context dense fallback | `n_comp_blocks` (correct quantity, raise-only) |

`n_slots = n_ctx / micro_block_size` (main.cpp:2239) is the size of the context
**window**, not the block count for this prompt. At the shipped
`micro_block_size = 1024`, `n_slots <= 32` holds for any model with
`n_ctx <= 32k` *and* for all three `DKV_PRESET` values, so the 4469 fallback
fires on essentially every request and pins `K = n_slots` regardless of prompt
length. The 4484 gate uses the right quantity but can only *raise*, so it can
never bring the `n_slots`-derived value back down.

This is the same failure mode measured on the MLX side in `1157f3a2` — "the fp32
decode cache is sized by `k_eff * block_size`, so it held 377.9 MB against a
48.2 MB store."

## Measured by transcription

`micro_block_size=1024`, `n_ctx=32768`, recency 512. `need = n_comp × (mbs+1)`.

| preset | L | n_comp | K | cap | need | waste |
|---|---|---|---|---|---|---|
| (none) | 4096 | 3 | 32 | 32800 | 3075 | **10.7×** |
| (none) | 8192 | 7 | 32 | 32800 | 7175 | **4.6×** |
| (none) | 16384 | 15 | 32 | 32800 | 15375 | 2.1× |
| (none) | 32768 | 31 | 32 | 32800 | 31775 | 1.0× |
| mid | 8192 | 7 | 20 | 20500 | 7175 | 2.9× |
| mid | 32768 | 8 | 20 | 20500 | 8200 | 2.5× |
| high | 16384 | 15 | 20 | 20500 | 15375 | 1.3× |

## The finding that changes the fix

**Re-keying gate 4469 off `n_comp_blocks` is necessary but not sufficient.** It
fixes the no-preset case (10.7× → 6.7×, 4.6× → 2.9×) but does **nothing** for
`DKV_PRESET=mid`/`high`: with a preset, `n_slots` is already ≤ 16, so 4469 never
fired there. The binding overshoot under a preset is gate 4446's
`adaptive_k_min = max(20, …)` — a floor of 20 that applies even when only 3
blocks exist.

Note also the `mid` rows: `n_slots = 8`, so the pool holds at most 8 blocks, yet
`K = 20`. K exceeding the pool size is meaningless on its face.

## Proposed fix (arithmetic verified, behaviour NOT)

Leave all three gates alone and add **one** final clamp after them:

```cpp
const int growth = (max_new + micro_block_size - 1) / micro_block_size;
srl_k_keep = std::min(srl_k_keep, std::min(n_comp_blocks + growth, n_slots));
```

The `growth` term is load-bearing: blocks keep compressing *during* generation as
the dense window flushes, so a bare clamp to the prefill-time `n_comp_blocks`
would under-allocate mid-answer. At `max_new=256`, `mbs=1024` that is one block.

Result across the same grid: **295200 → 135300 rows/layer, 54.2% less** (≈2691 →
≈1233 MB at 28 layers, head_dim 128, 8 KV heads, fp16, K+V). It fixes the
preset and no-preset cases alike and touches one line rather than two gates.

`batch_engine.cpp` carries the identical raise-only pair (~line 1379/1390) and
must receive the same clamp — that file already diverged from `main.cpp` once on
`micro_block_size` (256 vs 1024), fixed in `1157f3a2`.

## Why nothing was changed

There is no built native binary, no `.gguf` model and no Metal device on the
Windows box this was analysed on, and `dkv_native/tests/test_niah_native.sh` is
macOS-only. An end-to-end run needs the Mac. This repo's standing rule is to
verify the instrument before trusting the result, and an untested C++ edit to
cache allocation is exactly the kind of change that rule exists for. The
transcriptions in `tools/native_k_pipeline/` settle the allocation arithmetic and
nothing more — in particular they say nothing about whether a smaller K changes
output quality on the native path, which is the question `main.cpp` would
actually need answered before shipping this.

## Related

* `main.cpp:2153` held a *separate* instance of the same family —
  `std::max(16, 1024 / micro_block_size)`. **Resolved in `92bd7da1`, and the
  conclusion there corrects this one's first reading.** That floor looked like an
  active 16× overshoot against its comment's stated "≥1024 token" budget, but it
  was DEAD code: `1024 / mbs <= 16` for every `mbs >= 64`, so the expression
  collapses to the constant 16 — which is already `srl_k_keep`'s default. The
  comment was wrong; the value it produced never was, at any block size this repo
  has shipped. Removing it changed nothing.

  That distinction matters for the finding above, because it is the opposite
  case: the `n_slots` gate at 4469 is **not** inert. It genuinely raises
  `srl_k_keep` above what the block count justifies, and it feeds an allocation.
  "Same shape as a known defect" was not sufficient evidence in either direction —
  each had to be evaluated on what it actually computes.
* On the CUDA Python path this whole class of concern is currently moot: the
  decode cache is gated on `DKV_DECODE_CACHE_CUDA`, which defaults to `0`.
  Measured 2026-08-30, K=4 vs K=16 gave byte-identical peak VRAM there.


---

# What shipped (Mac session, 2026-08-30)

Applied to `dkv_native/src/main.cpp` and `dkv_native/serving/batch_engine.cpp`,
pinned by three tests in `ACTIVE_RUNTIME/tests/test_routing_k_budget_parity.py`
(each verified to FAIL on the unfixed source).

## Correction to the proposed fix: the growth term

The clamp above is right. The `growth` term was first implemented as the pool's
own `headroom_slots`, on the reasoning that the pool already reserves exactly that
for blocks compressed during generation, so reusing it could not drift from the
pool sizing. **That was wrong, and the grid in this document cannot show it** —
every row uses one block of growth, so both forms agree everywhere in the table.

`headroom_slots` is capped at `headroom_tokens_cap = 512` tokens and does **not**
bound how many blocks generation can create; generation runs until
`active_slot >= n_slots`. At the default `DKV_MAX_TOKENS = 2048` with
`micro_block_size = 1024` that is 2 blocks against a headroom of 1, so K would sit
one block short mid-answer — the precise failure the growth term exists to prevent.

The shipped term is this document's original `ceil(max_new / micro_block_size)`,
with `max_new` = `max_generate` in `main.cpp` and `req->max_tokens` in
`batch_engine.cpp`. It was caught by reading the clamp's own log line, not by any
test: `test_niah_native.sh` sets `DKV_MAX_TOKENS = 40`, where both forms give 1.

## Validation

Qwen2.5-1.5B-Instruct-f16, `micro_block_size = 1024`, needle sweep 4k/8k/16k x
depth 0.5/0.9, baseline binary vs shipped binary, same prompts and env:

| ctx | depth | baseline | clamped | srl_k_keep | output |
|---|---|---|---|---|---|
| 4000 | 0.5 / 0.9 | PASS | PASS | 32 -> 5 | byte-identical |
| 8000 | 0.5 / 0.9 | PASS | PASS | 32 -> 9 | byte-identical |
| 16000 | 0.5 / 0.9 | PASS | PASS | 32 -> 17 | byte-identical |

Byte-identity is not luck, it is forced. Prefill can occupy at most
`n_slots - headroom` blocks and generation can add at most `growth` more, so the
largest block count that can ever exist is exactly `k_ceiling`. The clamp sets K to
that maximum, so it can never prune a block that exists — it only releases capacity
that was provably unusable. The measurement confirms an argument rather than
standing in for one.

Predictions were recorded before the runs and came in one higher (5/9/17 against
4/8/16) because the real prompts exceed their nominal token targets by the chat
template and question, giving one more compressed block.

## What is NOT validated

* **`batch_engine.cpp` was not behaviourally exercised.** `main.cpp` never calls
  `DKVBatchEngine` — it only references it in comments — and there is no test in
  the repo that drives the batch path. Both of its changes are compile-verified and
  source-pinned only.
* **The growth term itself was never exercised behaviourally.** Producing a case
  where the headroom form actually fails needs generation to cross a block
  boundary; this model hit EOS at ~800 tokens on every prompt tried, well short of
  1024. Forcing it with `DKV_MICRO_BLOCK_SIZE=256` was discarded as an instrument
  failure — the **baseline** loses the needle at that block size (it is the
  configuration linkbench scores 9/24 on), so there is no working control to read
  the arms against. The growth correction rests on the arithmetic argument above,
  not on a measurement.
* **The 54.2% figure is `main.cpp` only.** `batch_engine.cpp` allocates no decode
  cache (`cache_routed_cap` does not exist there), so the clamp buys routing work
  rather than bytes on that path.

## Also fixed here: `srl_k_host` sized before the gates mutate it

Found while tracing the clamp's consumers. `batch_engine.cpp` created
`host_slots_decode` — whose length is `srl_k_host` — *before* the gate block, while
the gates raise `srl_k_semantic` to `3 x srl_k_keep` and recompute `srl_k_host`
from it. At `micro_block_size = 1024`, `n_ctx = 32768`: `srl_k_host` starts at 57
and the gates take it to 121, and the decode loop then writes
`srl_k_host * sizeof(int32_t)` — **121 int32s into a 57-int32 tensor**.

Not a merely wasteful size: the length is exact, since `route_decode_slots` both
pads and caps its result to `srl_k_host`. It survived because the decode loop's
re-creation of the tensor on pool growth *does* use the post-gate value, so the
overflow only appears when the pool does **not** grow — the ordinary path.
`main.cpp` has always created the tensor after its gates, so this was a divergence
between two entry points, not a shared design. Fixed by moving the creation below
the gate block, matching `main.cpp`.
