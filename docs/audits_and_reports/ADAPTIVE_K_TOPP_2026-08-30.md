# Adaptive K (top-p block routing): experimental, and why it is not the default

**Status: implemented, measured, DEFAULT-OFF. Do not enable without re-reading
the limits at the bottom.**

Knobs: `DKV_ROUTE_TOPP`, `DKV_ROUTE_TOPP_KMIN`, `DKV_ROUTE_SCORE`,
`DKV_ROUTE_TOPP_STATS`. Code: `native_core/srl/query_router.select_by_mass`.
Tests: `ACTIVE_RUNTIME/tests/test_route_topp_selection.py`. CUDA only.

## The idea

A fixed K cannot know whether a query is easy or hard. Take blocks until they
cover a share of the relevance mass instead: an easy lookup costs a few blocks,
a diffuse one takes as many as it needs — including **more** than the fixed K.

## It works mechanically

Measured K, unprompted and without retuning, all on Qwen3.5-2B / block 1024:

| workload | mean K | median | min | max |
|---|---|---|---|---|
| needles (`validate_cuda_dkv --long`) | **4.28** | 4 | 4 | 14 |
| synthesis @16k, floor 4 | 10.53 | 11 | 4 | 15 |
| synthesis @32k, floor 16 | **20.02** | 19 | 16 | 30 |

So the premise is sound: easy queries really do not need 16 blocks, and the rule
really does scale up with context. It also grows **past** the fixed default when
the context is large (20.02 > 16 at 32k).

## It does not convert into quality

Paired synthesis, `colab/synthesis_power.py`.

**Floored at 4 — adaptive is worse per block than fixed.**

| arm | mean K | score | vs fixed K=8 |
|---|---|---|---|
| top-p 0.99 | 10.53 | 37.7 | **+0.00**, CI [±6.99] |
| fixed K=8 | 8 | 37.7 | — |
| fixed K=16 | 16 | 44.4 | −6.67, CI [−11.79, −1.54] **resolved** |

It spends 10.53 blocks to buy exactly what a fixed 8 buys — 32% more cost for the
same result, and the highest variance of any arm (sd 14.8).

**Floored at 16 ("grow-only") — safe, but the headroom is not there.**

Grow-only is a strict *superset* of fixed top-16: at or below the threshold the
floor supplies exactly the fixed top-16 blocks, otherwise those plus more. It can
never route a worse set. At 32k, n=16:

| arm | mean K | score | vs fixed K=16 |
|---|---|---|---|
| fixed K=16 | 16 | 40.8 | — |
| grow-only 0.99 | 20.02 | 42.1 | +1.25, CI [−2.96, +5.46] |
| attend-all | ~29 | 40.6 | −0.21, CI [−4.98, +4.56] |

Nothing above 16 resolves. And the extra blocks are not free: the latency ladder
below prices grow-only's ~4 extra blocks at **≈ +6.7 ms/token, about 11% slower
decode**, for a gain that does not clear noise.

## Why: cost is non-linear, and K=16 is the knee

Paired decode timing, A/A control reads zero at ±3.17 ms:

| comparison | Δ blocks | mean_diff | verdict |
|---|---|---|---|
| K=4 vs K=8 | +4 | −0.30 ms | no effect |
| K=4 vs K=16 | +12 | −3.62 ms | no effect |
| K=16 vs K=27 | +11 | **−18.41 ms** | resolved, 22.6% faster at 16 |
| K=16 vs attend-all | +15 | **−26.10 ms** | resolved, 29.2% faster at 16 |

A block below 16 costs ~0.3 ms; above 16, ~1.67 ms — more than 5×. Not a
code-path artifact: explicit K=27 pays 18.4 ms over K=16 and attend-all adds only
8 ms beyond that. So **K=16 sits exactly where quality stops improving and cost
starts accelerating**, which is why it is the CUDA default.

Memory is unaffected either way on CUDA — peak VRAM was byte-identical at K=4 and
K=16, because `DKV_DECODE_CACHE_CUDA` defaults to `0`.

## The score is the real limit

`DKV_ROUTE_SCORE=lse` exists because the shipped score is a **max** over each
block's anchor + top-R residual keys — an upper bound on its best key, not the
mass it would receive. Softmaxing a max is too peaked to threshold sensibly.

logsumexp helps, but is bounded: `max ≤ logsumexp ≤ max + log(R)`, so it only
reorders blocks whose best logits already sit within ~log(R) of each other
(1.39 nats at R=4, 3.47 at R=32). It moved the median top-1 mass share only
0.673 → 0.577. A block whose relevance is spread thinly across ordinary tokens
stays invisible either way, because the score only ever sees the **top-R** keys.

**Making an adaptive rule genuinely better needs a score that looks past the
top-R keys.** Changing the selection rule is not enough, and neither is changing
the aggregation.

## UNFINISHED — the bigger-model control, now unblocked

**Qwen3.5-4B fp16 could not run at all until `940e6e57` + `fb62f362`.** Two
head-geometry bugs stood in the way (attention output reshaped to `hidden_size`
instead of `num_heads * head_dim`; `head_dim` derived by division instead of
read from config). Both are fixed and the 4B now runs clean — synthesis **53.3**
at 8k, no CPU fallbacks — but the sweep itself was started and **stopped early,
so there is no result yet.**

This is the single most valuable outstanding measurement, because every
"more blocks do not help" conclusion above rests on the 2B, and the 4B is the
clean control: same generation, fp16, no NF4 confound.

To resume (≈1–2 h on a 4070 SUPER, 8.8 GB at 32k so it fits):

```
BLOCK=1024 DKV_TOPK_BLOCKS=<K> python colab/synthesis_power.py \
    --arm dkv --reps 12 --ctx 32768 --model Qwen/Qwen3.5-4B --out c8_k<K>.json
```

for `K` in `4, 8, 16, 0` (0 = attend-all), plus the grow-only arm:

```
DKV_ROUTE_SCORE=lse DKV_ROUTE_TOPP=0.99 DKV_ROUTE_TOPP_KMIN=16 \
DKV_ROUTE_TOPP_STATS=1 DKV_TOPK_BLOCKS=16 BLOCK=1024 \
python colab/synthesis_power.py --arm dkv --reps 12 --ctx 32768 \
    --model Qwen/Qwen3.5-4B --out c8_grow.json
```

then `--compare c8_k0.json c8_k16.json` and friends.

**The decisive pair is attend-all vs K=16.** On the 2B it was −0.21, CI
[−4.98, +4.56] — nothing above 16 helped. If the 4B shows a gradient there,
"more blocks do not help" was a fact about the 2B's ceiling and adaptive-upward
routing deserves another look. If it does not, the conclusion generalises and
fixed K=16 is right for both.

## Limits of this evidence — read before overriding any of it

* **One model.** Qwen3.5-2B caps near 41–44/100 with 5–10 of 15 facts *even when
  shown every block*, so "more blocks do not help" may be its ceiling rather than
  routing's. The bigger-model arm could not be run fairly on a 12 GB card: NF4 is
  forced above ~4B, and both Qwen2.5-7B and Qwen3.5-9B scored 25.0 in NF4 against
  the 2B's 41–44 in fp16 — size and precision are confounded. **Qwen3.5-4B fp16
  is the untested clean comparison.**
* **The noise floor is ~3 points.** Fixed K=16 at 32k read 43.8 (n=8) and 40.8
  (n=16). Every effect measured above 16 blocks is smaller than that, and an n=8
  result in this exact comparison *reversed* at n=16.
* **64k is unexplored.** It runs (38.3) but scores *below* 32k, so long context
  degrades this model on its own; a fixed-K ladder at 64k is needed before any
  adaptive result there means anything.
* **Two guards shape every reading.** `route_blocks_relevance` early-returns
  **all** blocks when `N <= k_eff` — so at the shipped default a 16k context
  (~15 blocks) is *attend-all, not routed*, and top-p cannot engage there at all.
  And `_ROUTE_DIST_HOOK` must stay **below** the edge-propagation step:
  propagation flattens the relevance vector, and observing it from above
  under-predicts K badly (a 0.99 threshold looked capped at 15 of 27 blocks while
  the live rule measured mean 19.65).
