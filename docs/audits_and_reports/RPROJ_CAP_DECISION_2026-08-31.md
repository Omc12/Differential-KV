# `DKV_RSVD_MAX_RPROJ=32` — powered recall decision, 2026-08-31

**Verdict: the cap buys a reproducible −13.4% prefill and −73 MB peak, and did
not flip a single needle in 80 paired trials across two model scales — bounding
its effect at ≤3.7% of prompts (95%). It is the one part of the `DKV_FAST`
bundle still worth having, and the evidence now supports promoting it.**

## The question

`DKV_RSVD_MAX_RPROJ=32` is the last fidelity-affecting knob in `DKV_FAST`. It
caps the randomized-SVD projection width so every cuSOLVER call stays inside the
batched Jacobi 32×32 cliff. The rank boost it replaces existed specifically to
give **digit** blocks extra rank, and `test_niah.py` asserts a 6-digit needle —
so the risk lands exactly where the shipped test looks.

### What the cap actually does (narrower than the docs imply)

With `DKV_RANK_BOOST=off` — which has been the **default** for some time, so the
"two fidelity knobs to validate" is really **one** — the per-block ranks are
already 32. `block_ranks_t.clamp(max=r_proj)` is then a no-op. The cap does
**not** truncate stored rank; it only narrows the projection from
`rank + oversamples` (37) to 32. A subtler numerical change than "caps every
block's rank" suggests.

### Why the existing evidence could not decide

`test_niah.py` runs **n=3** against **one fixed needle** (847291), unpaired, and
any comparison of arm means would be a t-test on a binary outcome. All four are
defects:

| defect | consequence |
|---|---|
| n=3 | Wilson 95% on 3/3 is [0.438, 1.000], **width 0.562** |
| one fixed needle | samples one tokenisation, not the population of digit needles |
| unpaired | prompt-to-prompt variance (depth, digit tokenisation) buries the effect |
| t-test on means | wrong test for a binary outcome; McNemar is the paired form |

Re-running it more times does not fix this. The **instrument** had to change.

## Method

`colab/rproj_cap_decision.py`. One process, one model load. Each replicate draws
a **fresh 6-digit code** and runs **both arms on the same prompt** (paired), so
the comparison is within-pair. Reported per arm: recall with a **Wilson** 95%
interval (not the normal approximation — at p=1.0 that has zero width and would
report 3/3 as perfect certainty). Paired statistic: **exact McNemar** on the
discordant pairs only.

The statistics are validated against known answers before any GPU time:

```
python colab/rproj_cap_decision.py --selftest      # SELFTEST PASSED
  Case A  identical arms, n=32          -> p = 1.0000, 0 discordant
  Case B  cap loses 8, wins 0, n=32     -> p = 0.0078  (flagged SIGNIFICANT)
  Case C  the original n=3 situation    -> p = 1.0000  despite a 33% drop
```

Case C is the point: **the shipped n=3 test could neither clear the cap nor
catch a real regression.**

## What the cap buys

`colab/bench_prefill_arms_peak.py`, arms `rproj_off` / `rproj_32`, two
replicates each (RTX 4070 SUPER, Qwen2.5-1.5B):

```
ctx=16384   rproj_off  6.95, 6.80 s   peak 3953.0 MB
            rproj_32   6.11, 5.82 s   peak 3915.7 MB     -13.2%,  -37 MB
ctx=32768   rproj_off 14.96, 14.83 s  peak 4614.5 MB
            rproj_32  12.91, 12.95 s  peak 4541.4 MB     -13.4%,  -73 MB
```

Peak is **identical across replicates within each arm** at both contexts, and
`rproj_off`'s 32k peak (4614.5 MB) matches the independently measured default
arm exactly — two cross-checks that the instrument is sound.

Timings on this box are **eager-path** (no MSVC, so Inductor never compiles the
decode kernels); they are a pessimistic bound for a box with MSVC.

## What the cap costs

| run | model | ctx | n | cap_off | cap_32 | discordant |
|---|---|---|---|---|---|---|
| 1 | Qwen2.5-1.5B | 16k | 32 | 32/32 | 32/32 | 0 |
| 2 | Qwen2.5-1.5B | 32k | 32 | 32/32 | 32/32 | 0 |
| 3 | **Qwen2.5-7B NF4** | 16k | 16 | **11/16** | **11/16** | **0** |

80 paired trials, 80 distinct 6-digit codes, depths 0.1–0.9, three seeds.
**The cap did not flip a single needle in either direction.**

**Run 3 is the one that matters.** Runs 1–2 are at a 100% ceiling, where zero
discordance is nearly guaranteed and proves little. The 7B is at **0.688** — the
task is hard enough to produce five failures — and *both arms failed on exactly
the same five prompts* (4, 5, 6, 9, 10). That is the shape a fidelity-neutral
change makes: it tracks the baseline through its failures, not just its
successes.

### The statistic that survives the ceiling

Per-arm recall CIs are useless at a ceiling (pinned at 1.0) and wide at n=16
(width 0.414 — the harness flags run 3 as underpowered *on that criterion*).
The paired **disagreement rate** is well defined either way:

```
  n= 16, 0 discordant  ->  95% upper bound  17.1%
  n= 32, 0 discordant  ->                    8.9%
  n= 80, 0 discordant  ->                    3.7%   <- all three runs pooled
```

So the honest claim is: **the cap changes a needle outcome on at most ~3.7% of
prompts (95%)**, not "the cap is identical".

## On the null-instrument guard (a failed instrument, reported as failed)

The harness carries a pool-signature column intended as a positive control:
if the cap never reached compression, both arms would be the same computation
and identical recall would mean nothing.

**That column does not work, and should not be read.** The pool's allocation
lifecycle is not controlled from the harness — slots freed by `clear_session`
are reclaimed lazily during a *later* generate, so the per-generate delta picks
up the previous replicate's deferred frees. It came out negative for the second
arm in **31/31** pairs regardless of the knob. Clearing between arms and
clearing after the pair both fail, in opposite directions; absolute sums fare no
better, since the second arm's pool still holds the first arm's blocks.

**The real positive control is the timing table above.** A knob that never
reached compression could not produce a reproducible −13.4% forward and −73 MB
peak at two contexts. The cap is live, so the recall comparison is not vacuous.

## Limits — read before promoting

1. **Bound, not equivalence.** ≤3.7% of prompts is a bound. A rare effect —
   one prompt in fifty — would not have been seen.
2. **Model scale.** 1.5B and 7B. `test_niah.py:77` notes rank=16 is too low for
   14B models with `RANK_BOOST=off`; nothing here speaks to 14B, though that
   comment is about `rank`, not about this cap.
3. **One box, one task.** NIAH digit recall only, on an eager-path 4070. The cap
   is a global compression change; synthesis / multi-fact behaviour was not
   measured here, and `colab/run_nat_eval.py` already defaults `DKV_FAST=1`, so
   the natural-coherence numbers on record were produced **with the cap on**.

## Recommendation

**Stop describing `DKV_FAST` as carrying two unvalidated fidelity knobs.**
`RANK_BOOST=off` already ships as the default, so there was only ever one; and
that one now has 80 paired trials behind it instead of 3, at two model scales,
one of them off-ceiling.

Promoting `DKV_RSVD_MAX_RPROJ=32` to a default is supported by this evidence: it
is a one-line change at `lowrank.py:1432` (`"0"` → `"32"`) plus the matching
default at `kv_runtime_manager.py:739`. It is **not** applied here, because
changing a global compression default is the maintainer's call and limit (3)
means the synthesis side is untested. If it is promoted, re-run
`colab/eval_natural_coherence.py` on both arms first.
