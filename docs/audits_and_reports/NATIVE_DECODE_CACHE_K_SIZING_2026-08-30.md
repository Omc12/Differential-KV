# The native decode cache is sized off the context WINDOW, not the blocks that exist

**Status: analysis only, not fixed.** The arithmetic is settled; the behavioural
change is not, and must not be made from this box — see *Why nothing was changed*.

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

* `main.cpp:2153` has a *separate* instance of the same family —
  `std::max(16, 1024 / micro_block_size)`, which at `mbs=1024` attends 16×1024 =
  16384 tokens against its own comment's stated "≥1024 token" budget.
* On the CUDA Python path this whole class of concern is currently moot: the
  decode cache is gated on `DKV_DECODE_CACHE_CUDA`, which defaults to `0`.
  Measured 2026-08-30, K=4 vs K=16 gave byte-identical peak VRAM there.
