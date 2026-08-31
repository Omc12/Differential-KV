# Contiguous-prefill flags re-measured against TODAY'S default — 2026-08-31

**Verdict: do NOT promote `DKV_CONTIGUOUS_PREFILL` / `DKV_CONTIG_UNROTATE`.
The July recommendation is superseded. `DKV_SDPA_HISTORY` already took the win.**

## Why this was re-measured

`CUDA_VRAM_PERF_FINDINGS_2026-07-17.md` (§"1× contiguous prefill — CONFIRMED on
A100") recommends promoting both flags on the strength of:

```
                fwd     after_fwd   peak_prefill
2x contiguous   4.85s   15.23 GB    16.74 GB
1x un-rotate    5.03s   12.58 GB    14.09 GB   <- "near dense, fast fwd kept"
dense           6.07s   --          13.27 GB
```

Two defects make that table unable to decide the default:

1. **It never measured the DKV default path.** Every row is either an opt-in
   contiguous arm or dense. The quoted "−2.6 GB" is 1× measured against 2× —
   both opt-in. Against what actually ships, the delta was simply never taken.

2. **The baseline moved afterwards.** `DKV_SDPA_HISTORY` (fused history
   attention) became default-on in `1c293277` on **2026-08-13**, three weeks
   after that doc. Its own comment in `runtime/dkv_attention.py:379` measures it
   as *strictly better* than `CONTIGUOUS_PREFILL`: "reaches 9.28 s but costs
   +0.75 GB because it keeps a persistent rotated K/V buffer of every token."

So the July winner was a candidate to be a *regression* against the new default,
and nothing in the repo had checked.

## Method

`colab/bench_prefill_arms_peak.py`. One box, one model, one prompt; each arm in
its **own subprocess** (`_SDPA_HISTORY` is bound at import time, so flipping it
in-process would silently measure the wrong thing). Peak is
`torch.cuda.max_memory_allocated()` — it captures the rotated buffer even though
that buffer is freed when decode begins, so an after-generate reading would miss
it.

Hardware: **RTX 4070 SUPER, 12 GB** (not the A100 the July numbers came from),
Qwen2.5-1.5B-Instruct, preset mid, rank 32, greedy, 8 new tokens.

## Results

```
=== ctx=8192 — deltas vs TODAY'S DEFAULT (sdpa) ===
      arm    peak MB    d peak   fwd s    d fwd
    eager     3692.1     +66.8    4.44   +18.1%
     sdpa     3625.3      +0.0    3.76    +0.0%
 contig2x     3747.8    +122.5    3.69    -1.8%
 contig1x     3617.3      -7.9    3.81    +1.2%

=== ctx=32768 — deltas vs TODAY'S DEFAULT (sdpa) ===
      arm    peak MB    d peak   fwd s    d fwd
    eager     5857.7   +1243.2   21.25   +45.7%
     sdpa     4614.5      +0.0   14.58    +0.0%
 contig2x     5530.1    +915.5   14.02    -3.9%
 contig1x     4601.0     -13.5   14.52    -0.4%
```

`eager` = `DKV_SDPA_HISTORY=0`, i.e. the pre-2026-08-13 baseline the July table
was implicitly measured against.

**Fidelity: all four arms produced byte-identical text at both contexts.**

## What this means

- **`SDPA_HISTORY` already captured the entire win.** eager → sdpa is
  −1243 MB peak and −31% forward at 32k. That is the lever, and it already
  ships by default.
- **`contig1x` buys nothing.** −13.5 MB at 32k (−0.3%), and 0.4% slower. A wash.
  The July "−2.6 GB" does not exist relative to the shipping path; it was the
  distance between two opt-in arms.
- **`contig2x` is a real regression:** +915 MB peak at 32k to buy 3.9% forward.
  Nobody should set `DKV_CONTIGUOUS_PREFILL=1` alone.

### Instrument validation

This harness reproduces the two independently recorded numbers it can be checked
against, which is why its verdict is trusted here:

| claim | recorded | measured here |
|---|---|---|
| SDPA_HISTORY forward gain | −32% | −31.4% (21.25 → 14.58 s) |
| CONTIGUOUS_PREFILL peak cost | +0.75 GB | +0.92 GB |

Absolute values differ from the A100/7B numbers (different card, different
model); the directions and magnitudes agree.

## The `DKV_FAST` trap

`_apply_fast_mode()` sets `CONTIGUOUS_PREFILL=1` **and** `CONTIG_UNROTATE=1`,
which is the `contig1x` arm — a wash, not a win. But `CONTIG_UNROTATE` silently
downgrades itself to `False` in `dkv_attention.py` when the rotary module cannot
be resolved (`_resolve_rotary_emb` returns None, or raises). When that happens
`DKV_FAST` becomes the `contig2x` arm: **+915 MB peak at 32k**, with no warning.

## Recommendation

1. Leave both flags default-off. (Done — no code change was needed.)
2. Do not repeat the July recommendation without re-running this harness; the
   defaults it compared against no longer exist.
3. The remaining reason to touch `DKV_FAST` is `DKV_RSVD_MAX_RPROJ=32`, which is
   a compress-time (not prefill-peak) lever and is decided separately by
   `colab/rproj_cap_decision.py`.
