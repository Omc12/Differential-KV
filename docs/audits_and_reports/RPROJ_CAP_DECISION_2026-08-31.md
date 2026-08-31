# `DKV_RSVD_MAX_RPROJ=32` — powered recall decision, 2026-08-31

**Verdict: PROMOTED to default for configured rank ≤ 64. The cap buys −13.4%
prefill, −73 MB peak, and halves the pool slot (370 → 178 KB); it did not flip a
single needle in 80 paired trials, is identical seed-for-seed on linkbench at
29k, and does not regress multifact synthesis. `high` (rank=128) is left
uncapped — untested at that rank, and capping it would be a 4× cut.**

## The question

`DKV_RSVD_MAX_RPROJ=32` was the last fidelity-affecting knob in `DKV_FAST` (it
is no longer bundled there — see What shipped). It
caps the randomized-SVD projection width so every cuSOLVER call stays inside the
batched Jacobi 32×32 cliff. The rank boost it replaces existed specifically to
give **digit** blocks extra rank, and `test_niah.py` asserts a 6-digit needle —
so the risk lands exactly where the shipped test looks.

### What the cap actually does — and it DEPENDS ON THE CONFIGURED RANK

`DKV_RANK_BOOST=off` has been the **default** for some time, so the "two
fidelity knobs to validate" was really **one**. What the remaining one does then
splits by configured rank, and this distinction matters for reading everything
below:

- **rank = 32** (`low`, and every harness here that pins `rank=32`): block ranks
  are already 32, so `block_ranks_t.clamp(max=r_proj)` is a no-op. The cap only
  trims oversamples, `32+5 = 37` → 32. A mild numerical change.
- **rank = 64** (`mid` — the **default preset** — and `ultra`): the per-layer
  schedule lifts this to 96, and the cap truncates it to 32. A **3× cut**, and
  the pool banner says so: `pool_rank=96` → `pool_rank=32`.
- **rank = 128** (`high`): would be a 4× cut. Left uncapped — see What shipped.

So the needle runs below, which pin `rank=32`, exercise the *mild* form. The
linkbench and multifact runs, which use the default preset, exercise the 3× one.

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

## Synthesis: linkbench at 29k, default preset (where the cap bites hardest)

The needle runs above pin `rank=32`, where the cap only trims oversamples
(37→32). `colab/linkbench_cuda.py` does **not** pin rank, so it runs the default
`mid` preset — `rank=64`, layer schedule to 96 — where the cap is a **3×**
truncation. Multi-hop, distractors, graded on attribution (a confident wrong
answer scores 0, not a substring pass):

```
cap_off  seeds 11..18:  hit hit MISS hit hit MISS MISS MISS   HITS=4/8
cap_32   seeds 11..18:  hit hit MISS hit hit MISS MISS MISS   HITS=4/8
```

ctx=29,339, Qwen3.5-2B. **Identical seed-for-seed** — same hits, same misses,
same wrong answers on the misses. At 50% accuracy this is far off-ceiling, and
the cap tracked the baseline through every failure, under a 3× rank cut.

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

**The real positive control turned out to be the runtime's own log line.**
`kv_runtime_manager.py:743` applies the cap to `pool_rank`, and the pool banner
prints it. One seed of linkbench per arm, default `mid` preset:

```
cap off : rank=64 (pool_rank=96), 370 KB/slot, budget 4.0 GB, 11330 blocks
cap 32  : rank=64 (pool_rank=32), 178 KB/slot, budget 2.6 GB, 15209 blocks
```

That is a **3× rank truncation and 52% less memory per slot**, stated by the
runtime itself — unambiguous, and far better evidence than any checksum. The
timing table (−13.4% forward, −73 MB peak) is the second, independent control.
The cap is live; the comparisons are not vacuous.

### The coherence eval cannot test this at all

`colab/eval_natural_coherence.py` prints generations for a human to read and
produces no verdict, and its prompts are **67–159 tokens** — shorter than a
single 256-token block. Even after forcing `DKV_ENGAGE_THRESHOLD=128`,
`colab/coherence_cap_ab.py` logged *"DKV routing is not engaged for this
context"* on every prompt and both arms were byte-identical 3/3.

**That 3/3 is worth nothing.** Nothing was compressed, so the cap was a no-op
and the eval would have "passed" identically for a knob that broke everything.
It is recorded here so the pass is not mistaken for evidence. The synthesis
gates that *can* see the cap are linkbench and multifact, below.

## Limits — what this evidence does NOT cover

1. **Bound, not equivalence.** ≤3.7% of prompts is a bound. A rare effect —
   one prompt in fifty — would not have been seen.
2. **Model scale.** 1.5B and 7B. `test_niah.py:77` notes rank=16 is too low for
   14B models with `RANK_BOOST=off`; nothing here speaks to 14B, though that
   comment is about `rank`, not about this cap.
3. **`ultra` is capped by inference, not by direct test.** It shares mid's
   `rank=64` and is described in config as "MID PLUS AN UNROTATED POOL, AND
   NOTHING ELSE", so the mid evidence is taken to carry. It was not run.
4. **One box, and synthesis at n=1 per arm.** Everything is an eager-path 4070
   (no MSVC), so timings are a pessimistic bound. Synthesis *was* measured
   (linkbench 8 seeds, multifact 9/9) — but multifact is a single run per arm
   and its score is noisy, so 33.3 → 40.0 supports "no regression" and nothing
   stronger. Note also that `colab/run_nat_eval.py` defaulted `DKV_FAST=1`, so
   the natural-coherence numbers already on record were produced with the cap
   on — they are not an uncapped baseline.

## Synthesis: multifact at 16k (the metric mid's rank=64 was chosen on)

`mid`'s rank was justified by a synthesis score — its own comment reads
"0.9999/64 recovers most of the synthesis that 0.999 gives up (30.0 -> 43.3)".
Capping to 32 could have undone exactly that, so it was measured:

```
cap_off  9/9 checks   synthesis score=33.3  (facts 4/15, links 2/5)
cap_32   9/9 checks   synthesis score=40.0  (facts 6/15, links 2/5)
```

Multi-needle 3/3 and all four relational-binding checks pass identically in both
arms. Synthesis did not regress; it scored higher with the cap (n=1 per arm, so
read that as "no regression", not as an improvement).

## What shipped

`DKV_RSVD_MAX_RPROJ=32` is the **default for configured rank ≤ 96**
(`low`, `mid`, `ultra` — including the default path), at
`compression/lowrank.py` and `kv_runtime_manager.py`. The two sites must agree,
or the pool is sized for a rank the compressor does not produce.

> The threshold reads 96, not 64, because of the rank-ceiling change below.
> It was 64 until the preset ranks were rescaled by 1.5, at which point mid and
> ultra (64 → 96) fell out of the guard and the cap **silently stopped applying
> to the default path** — the pool quietly returned to 370 KB/slot with nothing
> failing. Fixed in `968418b4`, with two tests pinning the coupling.

**`high` (now rank=192) is deliberately left uncapped.** Capping it to 32 would be a
4× cut, taking the explicit fidelity preset *below* what `low` asks for, and
nothing here tested rank 128. The manager keys the guard off `self.rank` (the
configured base rank), **not** off `max_possible_rank` — the per-layer schedule
lifts mid's 64 to 96, so keying off the schedule would have skipped the cap on
the very preset the evidence covers.

An explicit `DKV_RSVD_MAX_RPROJ` always wins, in both directions. Verified:

```
mid,  no env var          -> rank=64  (pool_rank=32)   178 KB/slot   capped
high, no env var          -> rank=128 (pool_rank=192)  835 KB/slot   guard holds
mid,  DKV_RSVD_MAX_RPROJ=0-> rank=64  (pool_rank=96)   370 KB/slot   override wins
```

Suite after the change: 321 passed, 18 skipped — unchanged.

**Stop describing `DKV_FAST` as carrying two unvalidated fidelity knobs.**
`RANK_BOOST=off` already shipped, so there was only ever one; and that one now
has 80 paired needle trials, 8 linkbench seeds and a multifact A/B behind it
instead of 3 fixed-needle runs.


## Postscript: the rank ceiling this exposed (fixed, `ed9b4313`)

Chasing the cap surfaced something bigger: **no preset delivered the rank it
declared, and that predated the cap entirely.** The per-layer schedule's middle
band returned `1.5 * base_rank`, so `mid` declared 64 and stored 32–96, and
`high` declared 128 and stored 64–192 — it had never once delivered its stated
rank. Both wrappers even documented the value as "a CEILING, not a target".

Fixed as a reparameterisation, so nothing moved:

```
schedule multipliers /1.5   0.75 / 1.50 / 0.50  ->  0.50 / 1.00 / 0.333
preset ranks         *1.5   low 32->48, mid/ultra 64->96, high 128->192
MLX default rank     *1.5   32 -> 48
```

`new(1.5*b) == old(b)` at every layer. Pool sizing is unchanged (mid still
370 KB/slot uncapped, 178 capped), and linkbench over 8 seeds at 29k reproduced
`4/8` identically, seed for seed. `tests/test_layer_rank_ceiling.py` pins the
equivalence, the ceiling property, and the guard coupling — 106 cases.

One deliberate change: an **explicit** `config` rank now means what it says.
`rank=32` delivers at most 32 (16/32/11) where it used to deliver up to 48
(24/48/16). Presets are unaffected.

**MLX is edited but UNRUN.** `mlx_dkv_wrapper.get_layer_rank` is mirrored with
an explicit "do not fix one without the other", so it moved in step — but this
box has no `mlx` module. Mirror `test_layer_rank_ceiling.py` on Apple silicon
before trusting the MLX side.
