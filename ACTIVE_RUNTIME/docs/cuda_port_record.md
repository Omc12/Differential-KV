# What the MLX / Mac runtimes took from the CUDA work — and what they refused

Supersedes `MLX_PORT_FROM_CUDA.md` (deleted). That file was a TODO list written
from the CUDA side, by someone who had not run anything on Apple silicon; every
item was labelled `VERIFIED` (the construct was read in the MLX source) or
`LIKELY` (the mechanism is shared, the code path unconfirmed). This file records
what happened when each was actually measured here.

**The single most reusable thing in it is the method, not the settings.** Those
are collected at the bottom, because they are what stops the next sweep from
producing a number that has to be retracted.

## The adoption bar

**A CUDA change is adopted here only if it is measured to benefit DKV ON MLX.**
Working on CUDA is a reason to *try* it, never a reason to ship it.

This is not pedantry about provenance — the two runtimes have different physics.
CUDA has a split host/device memory hierarchy, a caching allocator that
fragments, PCIe transfers, and a launch overhead high enough that its runtime
measured ~40% GPU-idle waiting on Python. Apple silicon has unified memory: no
host copy, no `expandable_segments` fragmentation to reclaim, no transfer to
amortise, and much cheaper dispatch. A change that buys back a copy or hides a
launch is worth real time on CUDA and can be worth nothing — or worth negative
time — here. `DKV_STREAMING_COMPRESS` is the standing example in the other
direction: CUDA turned it off after measuring it much worse there, and MLX keeps
it ON precisely because the cause was CUDA dispatch overhead.

Applied below, this rule adopted three items (block size, pool sizing, the sampler
guard), refused one on measurement (the unrotated pool: implemented, correct,
working, and worth nothing here), and refused several on structure before any
measurement was needed. The two biggest surprises both came from measuring rather
than porting: CUDA's headline accuracy item does nothing on MLX, and the item CUDA
ranked second closed a 15-point gap to dense that MLX did not know it had.

---

## Implemented

### 1. The greedy sampler had no NaN guard — both runtimes

The MLX sampler ran `int(np.argmax(logits))` on the raw logits while the sampled
path was protected. That is backwards: greedy is the mode callers rely on to be
reproducible, and `np.argmax` over an array containing NaN returns the index of
the **first NaN** rather than raising — so one NaN silently selects a garbage
token, and which token that is moves with wherever the NaN landed.

DKV plainly knows NaN reaches this runtime: the attention combine guards it
(`mx.where(mx.isnan(out_sparse), 0.0, ...)`) and so do the LSE clamps. The
sampler was just never given the same treatment.

The native runtime had the same gap in a worse form. Its linear-scan argmax is
NaN-tolerant by accident (`NaN > x` is false, so NaNs are skipped), but the two
`std::partial_sort` / `std::nth_element` selections over the logits are **not**:
a `a > b` comparator is not a strict weak ordering once NaN is present, and
libc++'s introsort has no bounds check on the partition loop. That is undefined
behaviour, not merely a wrong token.

Fixed in both, with matching constants (`nan=-100`, `±inf=±100`) so MLX, CUDA
(`batch_engine.py`) and native all decode a non-finite logit vector to the same
token. Sanitising does not hide the NaN — the one-shot warning names it — it
makes greedy decode a *function of the logits* again.

* `_sanitize_logits` in `serving/mlx_dkv_wrapper.py`, applied in `sample_logits`
* `dkv_sanitize_logits` in `dkv_native/src/main.cpp`, applied where the prefill
  and decode logits are read back from the backend, so every downstream consumer
  (top-5 print, greedy argmax, nucleus sampler, SRL/factual bias) sees finite data
* `ACTIVE_RUNTIME/tests/test_sampler_nan_guard.py` — 11 tests, including one that
  asserts the *bug* still reproduces on raw numpy, so the guard's test is not vacuous

### 2. The pool was sized by total layers, not attended layers

Confirmed on MLX exactly as CUDA predicted, and worth more here than there.

`mlx-community/Qwen3.5-2B-4bit` is hybrid: **6 of 24 decoder layers have
`self_attn`** (indices 3, 7, 11, 15, 19, 23); the rest are linear-attention and
own no KV cache DKV can compress. Every per-layer slab was still allocated for
all 24.

Measured at 11,407 prompt tokens:

| | before | after |
|---|---|---|
| session pool | 542.5 MB | **135.6 MB** |
| bytes on layers that never hold a block | 406.9 MB (75%) | 0 |
| needle sweep (2k/8k × 3 depths) | 6/6 | 6/6, **byte-identical output** |

`MLXKVBlockManager._per_layer` allocates a zero-row slab of the same rank and
dtype for non-attended layers rather than dropping them, so every existing
`session[key][layer_idx]` index stays valid and in-bounds. Nothing reads them:
those layers' `num_blocks` and `dense_lens` never leave 0.

Dense-attention models (Qwen2.5, Llama, Mistral) are unaffected — every layer is
attended, so the mechanism is a no-op there. `DKV_POOL_ATTENDED_ONLY=0` restores
the old allocation.

### 3. `DKV_ROTATED_POOL=0` — implemented, verified, and NOT adopted

The highest-value item in the port list on CUDA, and the clearest case for the
adoption bar above. **It works here and buys nothing here.**

MLX ingested `keys_rot`, so a block's representation baked in the position it
held at **compression** time. On CUDA, storing keys unrotated took distractor
retrieval from 40/48 to 47/48 over 48 seeds — exact dense parity — while no
routing knob ever moved it at all.

Implemented in full: an unrotated shadow of the dense key window per attended
layer (the rotated window is untouched, since the dense half of decode attention
scores a rotated query against it directly); a new `comp_res_pos` recording each
exact residual's block-relative position, because `comp_res_mask` records *which*
positions were kept but not in *which slot* and read-time rotation needs exactly
that; and a gathered cos/sin table to rotate materialised blocks, their residuals
and the routing anchors to their absolute positions. A table is needed rather
than `mx.fast.rope` because routed blocks and residual tokens both sit at
*scattered* positions, which `mx.fast.rope` cannot express in one call.

It refuses rather than degrading: read-time rotation needs the keys materialised,
which only the decode-cache path does, so `DKV_ROTATED_POOL=0` with
`DKV_DECODE_CACHE=0` raises instead of silently scoring unrotated keys against a
rotated query. An interleaved ("traditional") RoPE layout is refused too, rather
than approximated with the half-split pairing.

**The measurement.** linkbench at 16k over 24 seeds, `QMODE=direct`, with a dense
control in the same configuration:

| block | rotated | unrotated | dense |
|---|---|---|---|
| 256 | 9/24 | **9/24** | 24/24 |
| 1024 | 24/24 | **24/24** | 24/24 |

The same score at both block sizes — and the same predicted answer on all 24
seeds, not merely the same count.

**That pattern is the "my change had no effect, check the code RUNS" signature,
so it was checked rather than assumed.** It runs: the stored anchors differ by
4.31 (pre- vs post-RoPE), the stored SVD basis by 133, and the decode-step-0
logits by 0.081. It changes the numbers and changes no answers.

Why, structurally: anchors and exact residuals round-trip to *exactly* the values
the rotated pool would have stored (that is what the round-trip test asserts), so
the only thing the knob actually changes is the SVD basis of the delta half — and
on this workload the answer is carried by the anchors, the exact residuals and
the dense window.

**Not adopted:** it costs ~39% of decode (41s → 57s per linkbench seed) and a
second dense-window buffer per attended layer, which on unified memory is real
system RAM. It stays an opt-in knob — correct, tested, and available if a
workload ever shows it paying — and it is NOT part of MLX's `ultra`.

**How it was verified, and why the obvious test was not enough.** RoPE is PARTIAL
on this model family — 64 of 256 dims, measured, not assumed — so a wrong position
can only perturb ~25% of each key. It degrades retrieval without ever zeroing a
score, which means a needle sweep passes straight through it. That is not
hypothetical: during development the unrotated dense shadow was not being shifted
in lockstep with the rotated window, so every block after the first was compressed
from stale tokens — **and the 8k needle test still passed and still recalled the
code exactly.**

What caught it was an exactness test on the residual half, where keys are stored
verbatim with no SVD in the way, so `rotate(stored, position)` must reproduce the
rotated key bit-for-bit at fp16. See `ACTIVE_RUNTIME/tests/test_unrotated_pool.py`.
Test the invariant, not the benchmark.

### 4. `ultra` preset — now says that it is `mid` here

On macOS `DKVHFWrapper` *is* `MLXDKVWrapper`, so `--preset ultra` already reached
this runtime and was silently ignored: it behaved as plain `mid` with no
indication. It still behaves as `mid`, because its one distinguishing setting on
CUDA is the unrotated pool and that is measured inert here — but it now prints
so, and names the flag to opt in anyway. A preset that quietly does nothing and a
preset that costs 39% of decode for nothing are both worse than one that explains
itself.

### 5. Block size 256 → 1024 — the lever that actually mattered here

CUDA called this the strongest single lever in the whole effort. On MLX it is
larger still, and it is the item that turned out to carry everything the
unrotated pool was expected to.

Qwen3.5-2B-4bit, linkbench at 16k over 24 seeds with a dense control in the same
configuration, plus the needle sweep and the session pool at 11.4k:

| block | linkbench | needles | pool | KV-side ratio |
|---|---|---|---|---|
| 256 (old default) | 9/24 | 6/6 | 135.6 MB | 0.95x (1.05x smaller) |
| **1024 (new default)** | **24/24 = dense** | **6/6** | **60.0 MB** | **0.28x (3.61x smaller)** |

Four metrics at once, no regression on any. **Default changed to 1024.**

Retrieval tracks the NUMBER OF BLOCKS the context is split into — at 16k, 256
gives ~58 blocks and 1024 gives ~15 — and not fidelity or routing. CUDA
established that separately by measuring rank, residual budget, recency window and
attend-every-block as all inert on this metric; MLX reproduces the mechanism
exactly, including the block counts at which parity appears. Splitting a document
into more pieces destroys cross-piece associations however faithfully each piece
is stored.

**The memory result is the same mechanism from the other side, and it is the more
alarming half.** The residual budget is a FIXED 128 exact tokens per block
regardless of block size. At 256 a block has 255 delta rows, so *half of every
block was being stored verbatim* and the "compressed" pool came to 0.95x the dense
KV it replaced — at the shipped default, DKV was barely compressing at all. At
1024 the same budget covers 4x the tokens and real compression appears.

`DKV_TOPK_BLOCKS`'s derived default is unaffected: `max(16, 4096 // block_size)`
is 16 at both sizes.

Use `BLOCK=512` for synthesis-shaped work, where CUDA measured the finer
granularity helping; 1024 is chosen for retrieval, which is what this system is
for. Note that everything else in the MLX history — the needle sweeps, the
paper's numbers — was taken at 256.

### 6. Decode-cache interval 16 → 4

The old default's justification in the source was "measured @32k: N=8->18,
16->20, 32->23 tps" — a ~15% cost for the shorter interval, which contradicted
CUDA's finding that throughput is flat across the range. Re-measured here rather
than assumed, with `colab/bench_decode_interval_mlx.py` (one process, one model
load, one prefill; arms interleaved with the order alternating every round;
paired statistic; min per round). Qwen3.5-2B-4bit at 9.8k:

| arm | ms/token |
|---|---|
| interval 16 | 22.764 |
| interval 4 | 22.669 |

    paired mean_diff +0.095 ms, 95% CI [-0.486, +0.677]  -> not resolvable

The harness's own A/A control reports +0.023 ms, CI [-0.010, +0.056] — correctly
no effect — so it is calibrated to detect one. A 15% effect would be ~3.4 ms and
could not have hidden inside that interval. **The old number does not reproduce.**

Skipping reconstruction on 3 steps in 4 already captures nearly all of the
saving, so 16 was paying staleness for nothing: the routed block set is FROZEN
for the interval, so a needle whose block is routed late stays invisible that
long. Default is now **4**, a 4x shorter frozen-routing window at no measurable
cost, matching CUDA's `DKV_REMAT_INTERVAL`.

### 7. Benchmark harnesses

Nothing above can be judged without these, and the port list said so.

* `colab/linkbench_mlx.py` (new) — the distractor-retrieval metric `rotated_pool`
  lives or dies on. Imports the document builder from `linkbench_cuda.py` rather
  than copying it, so both runtimes are graded on token-identical prompts.
* `colab/synthesis_power.py` — gained `--runtime {auto,cuda,mlx}` and MLX arms.
  The replicate/pairing/CI machinery is deliberately **shared** with the CUDA
  arms; forking it into a second file is how two "comparable" numbers stop being
  comparable. The one real difference is the seed variable: CUDA reads
  `DKV_RSVD_SEED`, MLX reads `DKV_SVD_SEED`.
* `colab/mlx_needle_parity.py` — gained `--max-new` and the reasoning-model trap
  below, and now prints the configuration next to the score.

### 8. Benchmark trap: a thinking model fails on the ANSWER BUDGET, not on recall

A model that emits a `<think>` preamble never reaches the answer at a 24-token
budget, and scores 0/N at every depth — which reads as total recall failure and
has been mistaken for one. An unclosed `<think>` in a failing case is the
signature. The needle harness now prints `TRUNCATED MID-<think>` with the budget
to re-run at instead of reporting a recall number that is not measuring recall.

### 9. The native runtime did not build

Unrelated to the port, but it blocked testing the Mac path at all: a leftover
`DIFFKV`→`DKV` rename left `GGML_OP_DIFFKV_ATTN` in the vendored
`ggml-metal-device.m` against `GGML_OP_DKV_ATTN` in `ggml.h`. One identifier;
`dkv_native` builds clean now.

### 8. Four structural seams found while porting, all fixed

These are not port items — they are things the port work walked into. Each was
measured before and after.

**The batched prefill compressor never ran on hybrid models, and silently
discarded the context when it was the only compressor.** It opened with
`for l in range(num_layers): if not prefill_K_chunks[l]: return`, so on Qwen3.5
(layer 0 is linear-attention and never stashes) the first iteration returned.
At the default `DKV_STREAMING_COMPRESS=1` that was harmless — the per-layer
compressor inside the forward had already drained every stash. At
`DKV_STREAMING_COMPRESS=0` it is the ONLY compressor, and the result was a
21,019-token prompt ending with **0 compressed blocks and a 2-token dense
window**: the whole context gone, with no error anywhere. Now iterates the
attended layers, indexing the batch by ORDINAL and the session arrays by TRUE
layer index. Post-fix it produces 19 blocks and a 1,565-token dense window,
identical to the streaming path, and passes the needle sweep 6/6.

**The manager ignored its own `recency_window` argument.** It derived the value
from model capacity whenever `DKV_ENGAGE_THRESHOLD` was unset, so
`MLXKVBlockManager(..., recency_window=64)` returned a manager with 512 and no
warning. A test that sized its sequence off the value it passed got a manager
that never compressed, and every assertion after that point passed vacuously —
which is exactly how it was found. Precedence is now env, then explicit
argument, then derived.

**Prefill converted the whole logits sequence to keep one row.** Measured shape
`(1, 512, 248320)` per prefill chunk — 508 MB copied to NumPy and again to
torch, of which 511 of 512 rows were discarded, ~7.6 GB across a 7.6k prompt on
an 8 GB machine. Every consumer takes `logits[..., -1, :]`; slicing on the MLX
side makes it `(1, 1, 248320)`. Decode was measured and deliberately left alone:
the whole mx→np→torch→np round trip is 0.234 ms against a 55 ms/token forward
(0.4%), so removing torch there would be churn against a cost that is not there.

**Entry points shadowed the runtime's own defaults.** `cli.py` and the OpenAI
gateway each carried `--micro-block-size default=256` and passed it EXPLICITLY,
so the measured block-size default could not reach either of them — while every
benchmark, which constructs the wrapper directly, saw the right value. Argparse
now defaults to `None` and forwards only what the user actually passed;
`serving/decode_config.MLX_CONSTRUCTOR_DEFAULTS` owns the numbers and
`resolved_runtime_config()` prints what a process actually resolved to, so the
next such change is a one-line check instead of a four-file read.

### 9. Two more fixed, and one number withdrawn

**The exact-residual budget was not clamped to the block's delta rows.** A block
of `block_size` tokens has S_comp = block_size - 1 delta rows, so at most S_comp
positions can be kept exact; the budget (default 128) was applied without that
bound. The selection returned S_comp rows while `pad_len = max_residual - n_res`
still evaluated to 0, so nothing padded and the store raised
`Shapes (31,2,64) and (1,128,2,64) cannot be broadcast`. Unreachable at
production defaults, but it blocked every small-block configuration — tests had
to pick an unnatural budget to avoid it. Clamped at all six sites across the
three compress paths; `tests/test_residual_budget_clamp.py`.

**Native's two entry points disagreed about block size.** `main.cpp` defaults to
256 and carries the comment *"micro_block_size 64->256 to match MLX reference"*.
The native OpenAI gateway then forced `DKV_MICRO_BLOCK_SIZE=64`, silently undoing
that fix for anyone who started the server rather than the CLI; the native CLI
separately set the variable unconditionally from an argparse default of 256,
shadowing the runtime's own value. Same class as the MLX shadowing above. The
gateway no longer defaults it and the CLI forwards only what the user passed.

**RESTORED WITH EVIDENCE: the MLX linkbench result.** An earlier revision quoted
"9/24 -> 24/24 = dense" as an MLX measurement that could not be evidenced, and it
was withdrawn. It has now actually been run — `colab/linkbench_mlx.py` at 16k,
`qmode=direct`, seeds 1-21, all three arms PAIRED on identical prompts:

| arm | hits |
|---|---|
| dense control | 21/21 |
| **DKV block 1024** | **21/21** — exact parity with dense |
| DKV block 256 | 9/21 |

Fisher p = 5.3e-05 for 1024 against 256; 1024 and dense are identical. Both the
direction and the magnitude match CUDA's independent result, and the dense arm is
what makes the comparison mean anything.

Note the withdrawal was still correct at the time: the numbers quoted then (9/24,
24/24) were not the numbers this run produced (9/21, 21/21), and no run existed to
point at. The lesson stands unchanged —

This is the file's own rule applied to itself: never quote a number without
being able to point at the run that produced it.

### 10. The batch engine returned EMPTY responses on MLX

Found by finally running the serving tests rather than only the wrapper ones.

`ContinuousBatchEngine` is written against the HF signature and calls
`self.wrapper.model(..., past_key_values=req.past_kv, use_cache=True)` on every
step. `MLXQwenModel.__call__` never accepted `past_key_values`, so the call raised
`TypeError`, the engine caught it as `Error in batch step`, and the request came
back with an empty response. Every request served through the engine on MLX.

Pre-existing — `batch_engine.py` is untouched by this work and the signature had
not changed. It stayed hidden because the wrapper tests call `generate()`
directly and never go through the engine.

Accepting and ignoring the argument is the correct fix rather than a workaround:
DKV's attention patch serves history from its own per-session store and never
fills the cache it is handed, which commit 20443474 measured directly — the
handed-in cache reports length ZERO on every step, prefill and decode alike. The
returned `ModelOutput` keeps `past_key_values = None`, so the engine's
`req.past_kv` stays None and the two sides agree.

**The lesson is the one already in this file, earned again:** a fast path that
declines by exception, inside a caller that swallows exceptions, is
indistinguishable from working. `Error in batch step:` printed to stdout and the
suite went green because nothing asserted on response content.

---

## Known-broken, not caused by this work

**RESOLVED: the native sparse-decode recall failure was the block-size default.**
It had been filed as a native-specific defect. It is not. Needle at ~13k on
Qwen2.5-1.5B-Instruct-f16 with sparse decode engaged, same prompt each time:

| micro_block_size | native sparse output |
|---|---|
| 256 (old default) | `ZEUS` — total failure, reproducible |
| **1024 (new default)** | **`ZEBR-4471-QUARTZ.`** — recovered, reproducible |
| dense control | `ZEBR4471QUARTZ` |

The dense arm drops the same `A`, so that character is tokenisation rather than a
DKV defect; at 1024 the sparse path matches its own control. `main.cpp` now
defaults to 1024 with the measurement attached.

This is the same block-count mechanism measured on MLX (linkbench 9/21 -> 21/21 =
dense at 16k). **Two runtimes and two model families — Qwen3.5 hybrid on MLX,
Qwen2.5 dense-attention on native — now agree independently that retrieval tracks
the NUMBER of blocks the context is split into**, not fidelity and not routing.
That is a much stronger claim than either measurement alone, and it was only
reachable because the native binary was actually run.

**The native loader requires `wq/wk/wv/wo` in every layer**, so it cannot load
Qwen3.5 at all ("Missing weights in layer 0"). Native testing needs a
dense-attention GGUF.

---

## Deliberately NOT ported, with the reason

**The `svd_energy` ladder (item 8).** CUDA keeps the smallest rank reaching an
energy target, then clamps to a rank ceiling. MLX's compressor has no energy
selection at all — it truncates at a fixed rank, which is exactly the ceiling
CUDA's own instrumentation says BINDS on real prose (realised rank tracked the
ceiling and barely moved across the entire energy ladder). Adding energy
selection here could therefore only ever lower the rank below today's, on a
fixed-shape pool where the freed columns are zero-padded rather than saved. Cost
with no upside.

**Fused prefill history attention (item 7).** Already fused on MLX.
`_sparse_prefill_attend` runs a single `mx.fast.scaled_dot_product_attention`
over the assembled sparse key set and never materialises a score matrix, so the
CUDA change has no MLX counterpart to make. MLX was ahead here.

**Routing knobs (item 1).** Retracted on CUDA by three independent methods:
48-seed linkbench identical between K=16 and attend-every-block, byte-identical
generated prose, and a paired synthesis measurement at ±4.3 resolution returning
exactly 0.00. Retrieval was never a *selection* problem — which is why the
unrotated pool moved it 40→47 while no routing knob ever moved it at all.
Nothing to change here.

**Zero-block decode (item 8h).** Checked: MLX already dispatches `nb == 0` to
`_dense_only_attention_static` rather than returning an empty tensor.

**Cross-session module globals (item 8h).** Checked: the MLX wrapper has no
module-level mutable state republished per step — no `global` statements at all.

**Dead pool slots (item 9).** Checked: MLX has no stratified-U or fact-anchor
equivalents, so the 31%-of-pool waste CUDA found does not exist here. (`comp_min_k`
/`comp_max_k` go unread on the decode-cache path, but they are 0.5% of the pool and
are read when it is off.)

**Dual-scale storage (item 10d).** CUDA implemented it and all three combining
policies failed. Merging two softmaxes over disjoint key sets is arithmetically
identical to one softmax over their union, so every token appears twice as two
different lossy reconstructions of itself and the exact dense window is diluted in
proportion. Do not build it here.

**CUDA graphs, `DKV_DETERMINISTIC`'s backend pin, fixed-shape routing,
`expandable_segments`, `gc.collect()` removals, occupancy-driven chunking.** No
Metal analogue. The transferable part of that work is the *gating* idea, kept
below.

**`DKV_STREAMING_COMPRESS`.** CUDA turned it off after measuring it much worse
there. Do NOT mirror that default back: the cause is CUDA-specific dispatch
overhead, and MLX's low launch cost is exactly why it is on here.

---

## Method rules that outlive the port

These cost the CUDA effort a full retraction and several afternoons each.

**Temperature-0 replication proves nothing.** It is deterministic, so a number
"reproduced twice" is one sample, not two. The randomised SVD's seed is the axis
that has to move: at a *fixed* config, changing `DKV_SVD_SEED` alone spans ~30
synthesis points. Treat any synthesis difference under ~15 points as no
difference, and never quote a single-seed number. Use `colab/synthesis_power.py`,
which is paired, replicated and interval-bounded, and prints how many replicates a
given effect size would need.

**Record the harness MODE next to the score, not just the harness name.**
linkbench has two question modes and `chain` is the default; `direct` collapses
the multi-hop chain to one lookup and is much easier. Comparing a `direct` score
against a `chain` one reads as a regression that is really two different
benchmarks.

**A score is only meaningful next to a control in the same configuration.** The
dense arm is what turns "DKV regressed" into "these are different tasks".

**State a check's COVERAGE, not just its result.** "Inputs verified identical" on
a dump of 8 of 256 elements is not that claim.

**Probe the OUTPUT before the inputs.** A per-step output trace bounds a
divergence in one run; a clean input probe can be misread indefinitely.

**A decline that silently changes the algorithm is worse than a crash.** Where a
fast path falls back to a differently-rounded one, label the decline — and where a
guard ANDs several conditions together, name the one that failed.

**Compare against the VALID extent, never the padded width.** Dense windows are
fixed-size workspaces whose validity is carried by a mask; both numbers are in
scope and they are not the same number.

**A knob whose payoff depends on a regime should be gated on that regime**, not on
the user remembering.

**When every knob measures as no-effect, stop turning knobs.** The fault is
structural, not configured.

**Report the KV-side ratio, not total device memory.** The pool stands in for the
dense KV of the same tokens; total footprint is dominated by weights, so a real
3–4× KV saving shows up as ~2–5% of the total and reads as failure.
