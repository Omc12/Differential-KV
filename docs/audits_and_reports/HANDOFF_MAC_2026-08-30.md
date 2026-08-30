# Handoff to the Mac session — 2026-08-30

Two jobs that need Apple hardware. Both are analysed; neither is applied. Start
from `main` at or after `de476096`.

---

## JOB 1 (higher priority) — MLX's K=4 default rests on a benchmark with no power

`1157f3a2` changed MLX's routing floor so that `K = 4096 // block_size` gives
**K=4** at the shipped `block_size=1024`, down from an effective 16. The evidence
cited was **linkbench 16k, 24 seeds: 24/24, unchanged from dense**, plus a
decode-cache saving of 377.9 → 151.1 MB.

**The retrieval half of that evidence does not support the conclusion.** On CUDA
I measured linkbench with a dense control and **dense scores the same as DKV** —
10/24 at 16k, 6/12 at 32k. When the dense arm cannot beat the routed arm, the
benchmark is at the model's ceiling and has **no power to detect a routing
regression**. "24/24, unchanged" is consistent with K=4 being fine *and* with
K=4 being much worse; the instrument cannot tell those apart.

**Synthesis, which does have power, was never run on MLX.** On CUDA the same
K=4 cost **−16.2 synthesis points** (paired, n=8, CI [−29.1, −3.4]) — and CUDA
subsequently kept its floor at 16 for that reason (`ffad683e`).

**What to run**

```
BLOCK=1024 DKV_TOPK_BLOCKS=4  python colab/synthesis_power.py --arm dkv --reps 16 --ctx 16384 --out mlx_k4.json
BLOCK=1024 DKV_TOPK_BLOCKS=16 python colab/synthesis_power.py --arm dkv --reps 16 --ctx 16384 --out mlx_k16.json
python colab/synthesis_power.py --compare mlx_k4.json mlx_k16.json
```

`--runtime` defaults to `mlx` on darwin. Use **n=16, not 8** — n=8 reversed on
me once in this exact comparison (see "noise floor" below).

**How to read it.** If MLX loses points the way CUDA did, K=4 is buying its
151 MB with quality and the floor should go back up. If MLX holds, the two
runtimes genuinely differ and `_K_MIN = 2` is correct — the asymmetry is already
pinned with its reason in `ACTIVE_RUNTIME/tests/test_routing_k_budget_parity.py`,
so update that test's docstring with whatever you find.

**One trap.** `route_blocks_relevance` early-returns **every** block when
`N <= k_eff`. At 16k with block 1024 there are ~15 blocks, so a K of 16 or more
is **attend-all, not routing**. Check MLX's equivalent guard before reading a
"K=16" arm as routing 16 blocks — on CUDA it was not.

---

## JOB 2 — native decode cache is sized off the context window, not the blocks

Full analysis: `docs/audits_and_reports/NATIVE_DECODE_CACHE_K_SIZING_2026-08-30.md`.
Runnable C++ transcriptions: `tools/native_k_pipeline/` (`g++ -O2 -std=c++17`,
no ggml/model/GPU needed).

`main.cpp:4607` sizes the decode cache as `srl_k_keep * (micro_block_size + 1)`.
Three gates can raise `srl_k_keep` and **all three are raise-only** — nothing
ever clamps it to the number of compressed blocks that exist. The gate at
`main.cpp:4469` keys off `n_slots` (`n_ctx / micro_block_size`, the context
*window*), which at `mbs=1024` satisfies `n_slots <= 32` for essentially every
request. At L=8192 that is **K=32 for 7 blocks — a 4.6× over-allocation**.

**Proposed fix (arithmetic verified, behaviour NOT):** leave the gates alone, add
one final clamp after them.

```cpp
const int growth = (max_new + micro_block_size - 1) / micro_block_size;
srl_k_keep = std::min(srl_k_keep, std::min(n_comp_blocks + growth, n_slots));
```

`growth` is load-bearing — blocks keep compressing *during* generation as the
dense window flushes, so clamping to the prefill-time count under-allocates
mid-answer. Result across the grid: **295200 → 135300 rows/layer, 54.2% less.**

`dkv_native/serving/batch_engine.cpp` has the identical raise-only pair (~1379,
~1390) and must get the same clamp — that file already diverged from `main.cpp`
once on `micro_block_size` (256 vs 1024), fixed in `1157f3a2`.

**Do not just apply it.** Re-keying gate 4469 off `n_comp_blocks` is *necessary
but not sufficient*: under `DKV_PRESET` it changes nothing, because there the
binding overshoot is gate 4446's `adaptive_k_min = max(20, ...)`. The single
final clamp handles both. Validate with `dkv_native/tests/test_niah_native.sh`
(macOS-only) and measure the cache before/after.

---

## Context you will want

**The noise floor is ~3 points** on `synthesis_power` with this model. Fixed K=16
at 32k read 43.8 at n=8 and 40.8 at n=16. Anything smaller than that needs
replicates, and the harness prints how many. I had an n=8 result reverse
completely at n=16 — top-p went from "ties attend-all" to "6.7 behind it".

**CUDA's answer, for reference.** K=16 is a knee in two curves: below it quality
falls (K=8 vs K=16, −6.67, CI [−11.79, −1.54], resolved); above it nothing
improves (grow-only at mean K=20 gives +1.25 unresolved; attend-all −0.21). And
cost is non-linear — a block below 16 costs ~0.3 ms, above 16 ~1.67 ms, so
K=16 vs K=27 is a resolved 22.6% speedup. MLX may differ: its decode cache is
ON, so K there also controls memory, which on CUDA it does not
(`DKV_DECODE_CACHE_CUDA` defaults to 0, and peak VRAM was byte-identical at K=4
and K=16).

**Two opt-in knobs exist and default off**: `DKV_ROUTE_TOPP` (mass-threshold
block selection) and `DKV_ROUTE_SCORE=lse` (logsumexp instead of max when
aggregating a block's keys), plus `DKV_ROUTE_TOPP_STATS=1`, which prints the K
the adaptive rule actually chose. They are CUDA-only; MLX has no equivalent. On
CUDA adaptive K did not beat a fixed K at matched cost, so do not port them
without a reason.

**Verify the instrument before trusting a result** — this session lost two
measurements to instrument defects: a probe hooked above the edge-propagation
step while selection runs below it, and a decode-timing harness that produced
negative times. Both were caught by controls (a self-check arm, and a
"no adaptive routing calls observed" counter), not by inspection.
