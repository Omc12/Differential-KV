# DKV — Findings Log

Running record of everything the MLSys evaluation campaign established, with the
evidence for each. Kept so the paper can be written from measurements rather
than from recollection, and so a claim can always be traced back to the run that
produced it.

**Conventions.** Every quality number is LongBench, official protocol, Q4/NF4,
`--max-length 12000`, 20 samples/task, thinking off, on an RTX 4070 SUPER (12 GB).
"Δ vs dense" is a **paired** bootstrap over per-item differences (10k resamples,
95% CI) — paired because every arm answers the same items, which removes item
difficulty from the variance. A row whose interval covers zero is reported as
**not resolved**, never as a win or a loss.

Raw rows: `paper/results/`. Nothing in this file is hand-copied from a log; every
table is reproducible with `--compare`.

---

## 1. Headline results

### 1.1 LongBench, granite-4.2-8b @ 12k

| arm | MACRO | Δ vs dense (95% CI) | KV | compression |
|---|---|---|---|---|
| snapkv | **43.11** | +0.34 [−0.73, +1.62] — not resolved | 0.335 GB | 4.7× |
| dense | 42.77 | — | 1.561 GB | 1.0× |
| kivi4 | 42.70 | (pending) | — | ~3.8× |
| h2o | 41.99 | −0.78 [−2.64, +0.85] — not resolved | 0.340 GB | 4.6× |
| dkv/high | 40.05 | −2.72 [−5.19, −0.39] | 0.743 GB | 2.0× |
| dkv/mid | 38.80 | −3.97 [−6.35, −1.87] | 0.525 GB | 2.9× |
| streamingllm | 33.81 | −8.96 [−12.79, −5.54] | 0.335 GB | 4.7× |
| kivi2 | 14.94 | −27.83 [−34.39, −21.51] | 0.207 GB | 7.5× |

**What this says.** On quality-per-KV-byte at moderate context, **SnapKV and H2O
are the frontier** — both statistically indistinguishable from dense at ~4.7×,
while DKV is measurably below dense at 2.0–2.9×. This must be reported as-is.
The contribution has to be argued on a different axis (§1.2), and reporting the
baselines winning here is what makes that argument credible.

Two structural observations worth keeping:

- The two **attention-observation** methods (SnapKV, H2O) both reach dense; the
  **position-only** method (StreamingLLM) is at −8.96. Knowing *which* tokens
  matter is what buys the quality, not merely keeping recent ones.
- **Quantized KV has a sharp bit cliff**: KIVI-4 is 42.70 (essentially lossless
  against dense's 42.77), KIVI-2 collapses to 14.94.

### 1.2 Context ceiling — the axis DKV wins

Measured ladder, Q4, preset mid (`paper/results/ladder/`):

| Qwen3.5-4B (hybrid, 8/32 attn layers) | GB per 1k | floor | max clean | first spill |
|---|---|---|---|---|
| dense | 0.0902 | 3.19 GB | 49,152 | 65,536 |
| **DKV** | **0.0413** | 5.83 GB | **98,304** (9.89 GB, 77 s) | 131,072 |

| granite-4.2-8b (dense GQA, 40/40) | GB per 1k | floor | max clean | first spill |
|---|---|---|---|---|
| dense | 0.2213 | 5.79 GB | 16,384 | 24,576 |
| DKV | 0.2230 | 7.64 GB | 16,384 | 24,576 |

**DKV halves the marginal cost per 1k tokens and doubles reachable context on the
hybrid model, and does neither on dense-GQA granite.** The claim is therefore
**architecture-conditional** and must be stated that way. Neither blanket
sentence is available: not "DKV does not reduce peak VRAM" (false for the 4B),
nor "DKV doubles context" (false for granite).

The mechanism for the split is **not established**. The obvious candidate — CUDA
defers compression until after prefill, so the dense KV is resident at peak —
predicts no benefit on *both* models, and the 4B refutes it. Recorded as open.

**Eviction cannot follow DKV here — MEASURED, 2026-09-02.** The prediction was
that eviction is a *post-hoc* operation: SnapKV ranks prefix tokens by attention
from an observation window to every prefix position, so the whole prefix KV must
be resident when it ranks, and its ceiling should therefore be dense's ceiling
however small the cache it keeps. Run on the same rungs, Qwen3.5-4B, Q4:

| arm @ 65,536 | final peak | **reserved** | status | wall |
|---|---|---|---|---|
| dense | 9.10 GB | **18.26 GB** | spilled | 122.7 s |
| snapkv | 8.58 GB | **18.27 GB** | spilled | 104.3 s |
| streamingllm | **3.65 GB** | **15.04 GB** | spilled | 78.7 s |
| **DKV** | 8.54 GB | **9.42 GB** | **ok** | **42.7 s** |

| arm | max clean context |
|---|---|
| dense | 49,152 |
| snapkv | 49,152 |
| streamingllm | 49,152 |
| **DKV** | **98,304** — 9.89 GB peak, 11.13 GB reserved, 76.7 s |

`peak_reserved` is the discriminator, and it makes the mechanism visible:
**StreamingLLM ends with the smallest cache of any arm — 3.65 GB — and still
cannot reach 65k**, because *building* the cache it intends to prune transiently
reserves 15 GB. Both eviction methods spill at exactly the rung dense does.

DKV compresses DURING prefill, so it never materialises the full KV. At 65,536
that also makes it **2.9x faster than dense** (42.7 s vs 122.7 s) — not from a
faster kernel, but from not paging.

This is the paper's central systems result, and it is the strongest available
form of the argument: the baseline with the *smallest steady-state footprint*
is still unable to reach the context DKV reaches.

CAVEAT, and it must travel with the claim: this is the hybrid model. On
dense-GQA granite DKV's slope matches dense's and its ceiling does not exceed
dense's (§1.2 table above). The claim is architecture-conditional.

**One near-miss worth recording.** The first run of this experiment returned
`snapkv@32768 error, @49152 error, @65536 error`, which reads exactly like
"SnapKV cannot reach those lengths". It was
`AttributeError: 'LinearAttentionLayer' object has no attribute 'keys'` — the
harness could not read a hybrid cache (Qwen3.5-4B is 8 full-attention layers of
32). A defect in our code, on the model the whole claim rests on, in the
direction that flatters DKV. See §3.9.

### 1.3 Where DKV's remaining quality gap actually lives

| task | dense | dkv/high | gap |
|---|---|---|---|
| passage_retrieval_en | 85.00 | **85.00** | 0.00 |
| narrativeqa | 23.56 | **24.40** | **+0.84** |
| hotpotqa | 30.01 | **30.03** | +0.02 |
| gov_report | 31.60 | 28.76 | −2.84 |
| qasper | 42.60 | 36.00 | **−6.60** |
| multifieldqa_en | 43.84 | 36.10 | **−7.74** |

DKV matches or beats dense on three of six. Essentially the whole deficit is two
**single-document extractive QA** tasks — and both eviction baselines are fine
there (SnapKV 45.23/43.49, H2O 43.08/38.64).

**The architectural reading**: eviction keeps selected tokens **bit-exact** and
discards the rest; DKV keeps **all** tokens **approximately**. For "quote the
right span from one document", exact-but-partial beats approximate-but-complete.
On `passage_retrieval` — find *which* block — DKV is at exact parity, because
that needs location, not verbatim tokens.

That makes **residual budget** (the tokens DKV stores exactly) the lever, not
rank. See §4.1 for the evidence that it is already the lever separating `mid`
from `high`.

### 1.4 SnapKV and H2O see the question; DKV does not

**Status: CONFIRMED CAUSALLY, 2026-09-02.** See the `--query-first` result at
the end of this section.

SnapKV and H2O rank prefix tokens by attention from an **observation window** —
the last 32 prompt tokens. In LongBench's format that window contains the
question verbatim. Measured, granite, qasper:

    "...Do not provide any explanation.  Question: How is the ground truth
     for fake news established?  Answer:"

So they choose which context to keep **while looking at the question**. DKV
compresses during prefill and never sees it — `query_text` is deliberately
withheld from it so the arms have equal information. The asymmetry runs the
opposite way from the one that was being guarded against.

Grouping §1.1 by whether the query falls inside that window:

| task | query in window? | dense | snapkv | dkv/high | **snapkv − dense** |
|---|---|---|---|---|---|
| multifieldqa_en | yes | 43.84 | 45.23 | 36.10 | **+1.39** |
| hotpotqa | yes | 30.01 | 31.34 | 30.03 | **+1.33** |
| qasper | yes | 42.60 | 43.49 | 36.00 | **+0.89** |
| narrativeqa | yes | 23.56 | 24.15 | 24.40 | **+0.59** |
| passage_retrieval_en | **no** (query is the abstract, mid-prompt) | 85.00 | 85.00 | 85.00 | **0.00** |
| gov_report | **no** (summarise — no query at all) | 31.60 | 29.48 | 28.76 | **−2.12** |

**SnapKV beats dense on every task whose window holds the query, ties exactly
where it does not, and loses where there is no query at all.** It also explains
DKV's two worst tasks: `qasper` and `multifieldqa` are precisely where a
query-aware method can pre-select the relevant spans and a query-agnostic one
cannot. Note `narrativeqa` is the one query-in-window task where DKV still
edges SnapKV (24.40 vs 24.15), so the pattern is strong but not absolute.

**The causal test** (`--query-first`, queued) moves the question block ahead of
the context so the window sees only the document tail. LongBench templates are
`\n\n`-separated blocks, so this is a reordering, not a rewrite: same words,
same instructions, same trailing answer cue, and verified **identical token
count** (qasper 4,104 either way). It changes identically for every arm. If
SnapKV's edge over dense collapses, the advantage was placement rather than the
eviction policy.

**What this does NOT license.** Even if the effect is confirmed, SnapKV's
numbers in §1.1 are not invalid — LongBench puts the question last, that is the
benchmark's own format, and SnapKV is being run exactly as its paper specifies.
The correct conclusion would be narrower: *query-aware eviction beats
query-agnostic compression when the query is known at compression time.*

**THE CAUSAL RESULT.** Question moved ahead of the context — same words, same
instructions, identical token count, applied identically to every arm. Deltas
are against the matching dense baseline for each ordering, so the prompt change
itself is controlled:

| arm | Δ vs dense, question LAST | Δ vs dense, question FIRST | shift |
|---|---|---|---|
| snapkv | **+1.14** | **−2.81** | **−3.94** |
| h2o | −2.36 | −4.45 | −2.09 |
| dkv/high | −7.17 | −8.49 | −1.32 |

**SnapKV flips from beating dense to losing to it.** DKV is query-agnostic and
shifts least (−1.32 is the baseline effect of the reordering on any compressor),
so SnapKV's query-awareness is worth roughly **2.6–3.9 points**.

**This does NOT rescue DKV.** With the confound removed the ordering on these
two tasks is dense 44.85 > snapkv 42.05 > h2o 40.40 > dkv 36.36. SnapKV's *lead
over dense* is explained; DKV's own deficit is not.

**Why the narrowing still matters for a systems paper.** SnapKV's advantage
*requires* the query at compression time. Under prefix caching or multi-turn
serving, a context is compressed ONCE and reused across many queries — an
eviction method would have to re-run selection per query, re-materialising the
full KV each time, which is the same cost that caps its context ceiling
(§1.2). DKV compresses once and serves any query. That is a real deployment
distinction and it should be stated whichever way the experiment lands.

### 1.5 What actually buys DKV's quality: residuals, not rank

Sweep on the two tasks DKV loses (qasper, multifieldqa_en), all against the same
dense baseline, n=40 paired:

| arm | score | Δ vs dense | compression | verdict |
|---|---|---|---|---|
| dense | 43.22 | — | 1.0× | |
| `high` — resid 256, rank 32 (**shipped**) | 36.05 | −7.17 [−12.81, −2.45] | 1.7× | worse |
| rank 64 (r_proj cap off), resid 256 | 35.84 | −7.38 [−13.08, −2.71] | 1.6× | worse |
| resid 512 | 38.69 | −4.53 [−8.89, −0.82] | 1.2× | worse |
| resid 512 + rank 64 | 38.42 | −4.80 [−9.00, −1.36] | 1.1× | worse |
| **resid 1024** | **43.30** | **+0.08 [−0.65, +0.76]** | **0.8×** | **not resolved** |

**1. Residual budget is the entire lever.** 256 → 512 → 1024 moves the gap
−7.17 → −4.53 → +0.08, monotonically.

**2. Rank contributes nothing measurable.** Lifting the r_proj cap changes
quality by ~0.2 points at both residual settings — inside the interval — while
costing +13.4% forward time and ~2× pool-slot bytes. Verified the arms really
stored it: the logs read `ceiling rank=64, respected` with `pool_rank=64`,
against `r_proj cap clamps to 32` for the capped arms. **The low-rank
approximation is not what buys DKV's quality on these tasks.**

**3. Dense parity is reachable, and it costs all of the compression.** At
resid 1024 the store is **0.8×** — 25% LARGER than the dense KV it replaces.
At the point where DKV matches dense it has stopped being a compression method.

Taken with §1.3 (DKV's deficit is concentrated in single-document extractive QA,
where eviction keeps its chosen tokens bit-exact), the picture is consistent:
**DKV's quality on these tasks tracks how many tokens it stores EXACTLY, and the
low-rank tail adds little.** That is a finding about the architecture, not a
tuning result, and it belongs in the design section rather than being fixed by a
preset change.

It also means the honest Pareto statement is: DKV's compression/quality curve on
extractive QA runs from 1.7× at −7.2 to 0.8× at parity — i.e. it does not
contain a point that is both compressed and dense-quality on these two tasks.

### 1.6 RULER: parity on 10 of 13 tasks, and a nameable failure mode

**Partial — Qwen3.5-4B, dense complete (1040/1040), dkv/mid at 4k/8k/16k
complete, 32k still running. Five arms not yet started.**

RULER generates the same task at exact context lengths, so a method can be
watched as context grows. 13 task types, scored 0-100 by the authors' own
`string_match_all` / `string_match_part`.

| task | what it tests | 8k dense | 8k DKV | 16k dense | 16k DKV |
|---|---|---|---|---|---|
| niah_single_1 | 1 needle in random noise | 100 | 100 | 100 | 100 |
| niah_single_2 | 1 needle in real essays | 100 | 100 | 100 | 100 |
| niah_single_3 | 1 needle, value is a UUID | 100 | 95 | 100 | 100 |
| niah_multikey_1 | needle + 3 distractor keys, in essays | 100 | 100 | 100 | 100 |
| **niah_multikey_2** | **haystack IS other needles (lookalikes)** | 100 | **30** | 100 | **20** |
| **niah_multikey_3** | **lookalike haystack, keys+values UUIDs** | 100 | **50** | 100 | **25** |
| niah_multivalue | 1 key, 4 values to recall | 100 | 95 | 99 | 100 |
| niah_multiquery | 1 key asked 4 times | 100 | 100 | 100 | 99 |
| vt | chain of variable assignments | 0 | 0 | 0 | 0 |
| cwe | 10 most frequent words | 42 | 43 | 58 | 54 |
| fwe | 3 most frequent words | 95 | 95 | 98 | 95 |
| **qa_1** | **QA quoting an exact span (SQuAD)** | 90 | **65** | 90 | **65** |
| qa_2 | multi-hop QA (HotpotQA) | 60 | 50 | 50 | 50 |

Aggregate at these lengths is −8.34 [−10.21, −6.51], but that number is
misleading on its own: **DKV is at dense parity on 10 of 13 tasks**, including
every basic needle-retrieval task at a perfect 100 even at 16k. The deficit is
three tasks.

**The three that fail share a property.** `niah_multikey_2` and `_3` are the
only tasks whose `type_haystack` is `needle` — the haystack is built out of
*other needles*, hundreds of near-identical sentences, so the job is
discriminating one key among lookalikes. `_3` additionally uses **UUIDs** for
both key and value: maximum-entropy strings with no pattern to reconstruct.
`qa_1` requires quoting an exact span.

The control is `niah_multikey_1`, which also has distractor keys but places them
in ordinary essay text: **100 → 100**. The moment the distractors become
lookalikes, DKV drops to 20.

**So the failure mode is nameable: DKV preserves semantic LOCATION but loses
fine discrimination among near-identical items, and exact reproduction of
high-entropy tokens.** Low-rank compression keeps the gist of a block; it does
not keep the digits.

This is the same conclusion §1.3 reached from LongBench by a different route —
parity on `passage_retrieval` (30 *distinct* paragraphs, 85.00 = dense) and a
deficit on single-document extractive QA. Two independent benchmarks, one
mechanism. It is also what §1.5 predicts, since the residual budget is exactly
the store of bit-exact tokens.

**Two reading notes.** `vt` is 0 for BOTH arms — the model cannot do variable
tracking at all, so that row says nothing about compression and must not be
counted as a DKV failure. And the 4k column (not shown) is identical for both
arms on all 13 tasks, because DKV does not engage below ~4,970 tokens (§4.4) —
a useful built-in control that the harness and scoring are not themselves
introducing a gap.

---

## 2. Correctness bugs found and fixed

Every one of these was found by measuring against a true dense control, and
every one was silent — none raised an error.

### 2.1 The attention scale — the big one

DKV's **decode** kernels hardcoded `1/sqrt(head_dim)`. granite-4.2-8b declares
`attention_multiplier = 0.0078125`; `1/sqrt(128) = 0.0883`. **DKV's decode
softmax ran 11.31× too hot.**

| granite, 16k, token 2 | before | after |
|---|---|---|
| top-1 agreement vs dense | 1/3 | **3/3** |
| KL(dense‖DKV) | 3.62980 | **0.00000** |
| dense-top1 rank | 6.00 | **0.00** |

LongBench effect: MACRO 25.85 → 38.80, paired Δ −16.92 → −3.97.

Commit `7346a492` had fixed the attention **module** and recorded the 11.3×
figure — but not the four decode kernels it calls into (three
`inv_scale = 1.0/math.sqrt(D)` sites, plus `attend_with_remat`'s SDPA with no
`scale=`, so torch applied its own default). **Prefill used the model's scale;
decode did not.**

That single fact explains the entire prior symptom set: token 1 exact, token 2
wrong, text degenerating as error compounds, every config knob inert to five
decimals (none touches the scale), and extra rank barely helping (capacity
cannot fix a softmax temperature).

Scales that differ from the Llama default: **granite 0.0078125 (11.3×),
gemma-4 1.0 (22.6×)**; Qwen3.5 is 0.0625 = 1/sqrt(256) and was never affected.

The fix publishes the model's scale onto the pool (every decode kernel receives
`pool`; none receives the module) and makes the fallback **loud**. The silent
fallback is what let this ship. `tests/test_attention_scale_matches_model.py`
fails for any new model family whose declared scale is not the default.

### 2.2 Reasoning models were scored on their scratchpad

granite-4.2 and Qwen3.5 both open a `<think>` block by default. LongBench allows
128 generated tokens on qasper, so the model was still reasoning when it ran
out — every prediction began *"Okay, let's tackle this question:"* and no answer
was ever reached. qasper scored ~0 instead of 24.80, and each item took twice as
long. Thinking is now off by default, applied identically to every arm, and
recorded in the run config.

### 2.3 The DKV wrapper returns prompt + completion

`DKVHFWrapper.generate()` builds `generated = prompt_ids.copy()` and decodes the
whole list. Scored as-is, a 34,000-character prompt echo drives token-F1 to zero
on every sample and the arm looks catastrophically broken when nothing is wrong.
Harnesses must strip it; the recorded answer is additionally bounded by the
generation budget, since anything longer is echoed context by definition.

### 2.4 Chunked-softmax reduction could produce NaN

Both reduction kernels merged per-chunk partials with
`alpha = exp(m_i - m_new)`. A chunk with every key masked carries `m_c = -inf`;
merged against an accumulator also at `-inf`, both subtractions are
`(-inf) - (-inf) = NaN`, poisoning every head. `O_i / l_i` is 0/0 when all
chunks are empty. Also, the workspaces were `torch.empty` — uninitialised — **and
cached across decode steps**. Both fixed. (Neither was the `noremat` NaN; see
§5.)

### 2.5 transformers 5.x cache API

`to_legacy_cache`/`from_legacy_cache` are gone and iterating a `DynamicCache`
now yields 3-tuples, so `for (k, v) in cache` raises. **None of the competitor
baselines could run on this stack** until the plumbing was rewritten.

---

## 3. Measurement defects — the instruments were wrong before the system was

### 3.1 On Windows, exceeding VRAM does not raise

Under WDDM, CUDA oversubscribes into host RAM rather than failing. granite
dense, Q4, on a 12.28 GB card:

| tokens | peak | wall |
|---|---|---|
| 16,384 | 9.42 GB | 10.6 s |
| 24,576 | 11.23 GB | 65.7 s |
| 32,768 | **13.04 GB** | 380.5 s |

**13.04 GB "succeeded" on a 12.28 GB card.** A harness trusting the absence of
an exception reports an unbounded ceiling and quotes PCIe bandwidth as compute.
Two detectors are needed: allocation against 94% of the physical card, **and** a
latency cliff (>2.5× jump in s/1k against the previous rung) — the cliff is the
one that catches 24,576, where 11.23 GB looks fine but the run is already paging.

### 3.2 Inductor was never compiling

`cl.exe` is not on PATH outside a Developer Command Prompt, so
`_reconstruct_and_score` fell back to eager and **every latency number
understated DKV**. Quality unaffected ("correct but unfused"). Now imported
in-process before the first `torch.compile`; `inductor_fused` is recorded per
row so eager and fused timings cannot be silently averaged.

### 3.3 Two competitor baselines were handicapped ~5×

SnapKV and H2O need real attention weights, so the model was loaded fully
`eager`. But only the 32-token **observation window** needs them — the other
~12,000 tokens do not. Measured: **55 s/item eager vs 11 s under SDPA**, a 5×
gap that is the attention kernel, not the eviction policy, and in exactly the
direction that flatters DKV. Now switched only around the window forward.

Trap worth recording: asking SDPA for `output_attentions` does **not** raise and
does **not** return `None` — it returns an **empty tuple**.

### 3.4 A bare mean at n=20 invites reading noise as a result

Hence the paired bootstrap. On the known-broken DKV arm it returned
−16.92 [−21.55, −12.39], confirming that defect was never a sample-size
artifact; on SnapKV it returns +0.34 [−0.73, +1.62] and is correctly reported as
**unresolved** rather than as a win over dense.

### 3.5 Checkpoint resume merged two experiments

The attention-scale fix changed decode arithmetic but left the run config
byte-identical, so a resume appended post-fix rows onto 81 pre-fix ones:
`gov_report 10.63 → 28.55` beside `hotpotqa 15.12 → 15.12`. Half of one
experiment averaged with half of another, invisibly. Now `dkv_decode_rev`
fingerprints the four decode-critical source files, so a kernel change trips the
same fatal guard a preset change does.

### 3.6 The in-house synthesis metric is gameable

`config.py` records "block 1024 gives synthesis 63.3 (past the dense 60.0)".
`synthesis_eval.py` scores `fact_score = (n_facts / 15) * 50` — **keyword
presence plus co-occurrence, with no fluency term at all**. A decoder degenerating
into on-topic vocabulary scores *well* on it, which is exactly what the broken
decode produced. **Do not use it for quality claims in the paper.** LongBench
`gov_report` (reference-based ROUGE, real dense control) is the summarization
measurement.

### 3.7 Verbosity accounts for a minority of the gap — and was worth checking

DKV emitted 2.5× more words than dense (183.7 vs 73.2), and F1/ROUGE are
precision-weighted. Charitable re-scoring:

| post-processing | dense | dkv/high | gap |
|---|---|---|---|
| official | 42.77 | 24.67 | −18.10 |
| + first line only | 38.64 | 26.03 | −12.61 |
| + truncated to dense's length | 42.77 | 28.16 | −14.61 |

So verbosity was worth ~3.5–5.5 of 18 points. But `passage_retrieval_en` settles
it: with equal truncation, dense 85.00 vs DKV 52.33 on a task whose metric only
asks whether the right `Paragraph N` appears **anywhere** — where a longer answer
can only help. Truncation is also not a legitimate protocol: it drops dense's
gov_report from 31.60 to 9.70. (All pre-scale-fix; retained as the record of how
the question was settled.)

### 3.8 LongBench cannot test long context — measure its lengths before believing it can

Prompt lengths over LongBench's own data, Qwen3.5-4B tokenizer, before any
truncation:

| task | median | p90 | max | > 32k |
|---|---|---|---|---|
| narrativeqa | **31,740** | 62,035 | 67,724 | 85/200 |
| hotpotqa | 15,108 | 17,085 | 17,800 | 0 |
| passage_retrieval_en | 12,770 | 14,414 | 15,549 | 0 |
| gov_report | 8,928 | 18,492 | 52,421 | 3 |
| multifieldqa_en | 7,367 | 12,223 | 16,801 | 0 |
| qasper | 4,806 | 7,616 | 21,831 | 0 |
| **overall** | **11,555** | 31,713 | 67,724 | **7.7%** |

**LongBench is a MODERATE-context benchmark**: median 11.5k, and only 7.7% of
prompts exceed 32k. Raising `--max-length` does not change that — the documents
are simply not longer. `narrativeqa` is the only genuinely long task.

Consequence for this campaign, and it was a planning error worth recording: the
granite @12k table is a **quality control at moderate context**, not a
long-context result. The 12k budget was set by granite's measured ceiling
(16,384 clean, spills at 24,576), and letting that constraint define the whole
campaign put the headline comparison at a length where DKV has no structural
advantage — which is also where the eviction baselines are strongest.

The long-context claim has to come from **RULER**, which generates at any
length, and from the **context ladder**. Both are run on Qwen3.5-4B, the model
that actually reaches 98k on this card.

Note for the 64k RULER run: dense spills at 65,536 and the eviction arms must
materialise the full KV before pruning, so they spill too. Their QUALITY is
still valid there — spilling costs speed, not correctness — so that run is
simultaneously a fair quality comparison and a systems result, because only DKV
runs the length natively (8.54 GB, clean).

### 3.9 The baselines could not run on the hybrid model at all

Every KV baseline died on Qwen3.5-4B with
`AttributeError: 'LinearAttentionLayer' object has no attribute 'keys'`. The
model is 8 full-attention layers out of 32 (`layer_types`: 24 linear_attention,
8 full_attention); the rest carry a recurrent state and no KV cache, and the
cache plumbing assumed every layer has `.keys`/`.values`.

It broke the ladder run that tests whether eviction can reach DKV's contexts —
on the exact model that claim rests on — and produced three `error` rows that
read like a ceiling. Same shape as the eager-attention handicap (§3.3): our
defect, flattering our method.

A second bug surfaced with it: dense-equivalent KV bytes multiplied by
`num_hidden_layers` (32) when only 8 layers hold KV, **inflating the dense
baseline 4x and every compression ratio measured against it on hybrids**.

Both fixed; a pass that compresses zero layers now raises rather than reporting
a footprint.

---

## 4. System / architecture findings

### 4.1 Preset rank is not what the preset says

Two individually correct caps compose badly:

1. `hf_dkv_wrapper.py:947` — `rank >= head_dim → rank = head_dim // 2`. A rank
   cannot exceed the dimension it factorizes.
2. `kv_runtime_manager.py` — `_pool_rproj_cap = 32 if rank <= 96 else 0`.
   cuSOLVER's batched Jacobi solvers cap at 32×32; above it PyTorch loops per
   matrix (eigh **0.006 → 0.812 ms/call**, ~130×, ×2,352 calls per prefill).

| model | preset | head_dim | declared | after cap 1 | **stored** |
|---|---|---|---|---|---|
| granite-4.2-8b | mid | 128 | 96 | 96 | **32** |
| granite-4.2-8b | **high** | 128 | 192 | **64** | **32** |
| Qwen3.5-4B/2B | mid | 256 | 96 | 96 | **32** |
| Qwen3.5-4B/2B | **high** | 256 | 192 | 192 | **192** |

**On head_dim=128 models `high` stores the same rank as `mid`.** The comment at
the cap site says "`high` (rank=128) is deliberately left uncapped" — stale
twice: high is 192, and granite's high is 64 by the time the check runs, so
`<= 96` holds and the exemption never fires.

**This is load-bearing**: granite's `high` beats `mid` (−2.72 vs −3.97) at
*identical stored rank*, so that difference is attributable to **residual budget
alone** (256 vs 128). Residual is the demonstrated capacity lever; rank is an
untested one on this model.

Never quote a preset's declared rank as what ran — read the
`[DKV Memory] Effective per-layer rank actually stored` line.

### 4.2 True preset values

| preset | max_residual_tokens | rank (declared) | residual_quant |
|---|---|---|---|
| low | 64 | 48 | int8 |
| mid | **128** | 96 | int8 |
| high | **256** | 192 | int8 |
| ultra | 128 | 96 | int8 |

The table in `run_a100_paper_experiments.py` (`{"low":40,"mid":64,"high":128}`)
is wrong on all three. Shipped residual format is **int8**, not int4
(`residual_quant_bits = 4` is only the width *if* int4 is selected).

### 4.3 Streaming compression is not the peak-memory lever on CUDA

`DKV_STREAMING_COMPRESS=1`, granite, 16k: **peak 27.69 GB vs 11.30, and 696 s vs
13.6 s.** It does not bound peak on CUDA, it explodes it. `config.py` explains
why it is off by default there (batched deferred SVD is ~20× faster than
layer-by-layer), while MLX defaults it on "for bounding peak VRAM".

### 4.4 DKV engages just above 4,096 tokens

Measured: 3,965 tokens → 0 compressed blocks; 4,970 → 140.
`DKV_COMPRESSED_MIN_CTX = 8192` is a *separate* gate, for sparse **decode**
routing. Below the engagement floor a fidelity comparison measures dense against
dense and means nothing.

---

## 4b. Scope decisions — what was NOT run, and why

Recorded so a reviewer (or a later session) can see these were budget decisions
with stated reasons, not gaps that were quietly left.

### 4b.1 RULER runs 4 arms, not the 9 LongBench uses

RULER is 1,040 items per arm at 4k/8k/16k/32k. The dense arm alone took **~2
hours**; seven arms is 10–12 h at 32k and considerably worse at 64k, where
items run several times slower. The full matrix as originally scoped was **20+
GPU-hours** on a single desktop card.

**Kept**, because each answers something still open:

| arm | why |
|---|---|
| dense | the reference every delta is measured against |
| dkv_mid, dkv_high | the core quality-at-length question, and the preset ladder |
| snapkv | the one competitor that beat DKV at 12k (+0.34, n.s.) |

**Dropped at RULER lengths**, with the reason each is already answered:

| arm | why it was cut |
|---|---|
| h2o | tracks snapkv closely at 12k (−0.78 vs +0.34) and shares its observation-window mechanism, so snapkv already represents attention-observation eviction |
| streamingllm | characterised at 12k (−8.96) and consistently the weakest eviction arm; nothing about length changes that reading |
| kivi2 | catastrophic everywhere (−27.83 at 12k); a second confirmation is not worth 2 h |

**At 64k, only dense and DKV run.** §1.2 already measured that snapkv and
streamingllm spill at 65,536 *exactly where dense does*, because eviction is
post-hoc and must materialise the full KV before pruning. Running them at 64k
would measure paging, not the method — their latency would be PCIe bandwidth
and their peak partly host memory. Dense is still run there (spilled) because
**spilling costs speed, not correctness**, so its quality remains a valid
reference.

Net effect: ~20 h → ~5 h, and nothing the paper claims depends on the cut arms.

`RULER_ARMS` selects the set, so any dropped arm can be added later without
touching the script.

### 4b.2 What this costs

Honest statement of the limitation: the RULER tables carry **one** competitor
(SnapKV) rather than four. If a reviewer asks how H2O or KIVI behaves at 32k,
the answer is that it was not measured and the 12k reading was extrapolated.
That is a defensible trade at 20 GPU-hours, but it is a trade, and the paper
should say so rather than presenting a 4-arm table as if it were the whole
comparison.

---

## 4b.6 RULER @64k carries DKV only -- no baseline can run there

**The result, stated plainly: on a 12 GB card at 64k, DKV is the only arm that
fits.** This is not a gap in the evaluation; it is the evaluation.

The context ladder had already measured it, per arm, on Qwen3.5-4B:

| arm | max clean rung | 65,536 |
|---|---|---|
| dense | 49,152 | spills |
| snapkv | 49,152 | spills |
| streamingllm | 49,152 | spills |
| **dkv** | **98,304** | **fits** |

A RULER table at 64k with a dense or SnapKV column is therefore not something
this hardware can produce. At 64k the honest comparison is DKV against the
context limit itself, not against another method.

### How it was nearly gotten wrong

A dense@64k arm was queued anyway, without consulting the ladder, and it ran
1,046 items before the cost was examined. The per-item cost is the tell, but
only after backing the MARGINAL cost out of the running average the harness
prints -- the average is cumulative over all items and hides the cliff:

    item 1040 (32k):  marginal  11.9 s      avg 11.9 s
    item 1041 (64k):  marginal 532.4 s      avg 12.4 s
    item 1045 (64k):  marginal 849.9 s      avg 14.7 s

A 45-72x cliff at constant 100% GPU utilisation and 11.2 / 12.28 GB -- the WDDM
spill signature (SS 3.x), where the work is PCIe bandwidth rather than compute.
The displayed average still read "14.7s/item, ~62 min left" while the true
remaining cost was 255 x ~530 s = **over 37 hours**, every timing contaminated.

The file was deleted rather than kept: all 1,040 of its sub-64k rows were
already present in the 32k campaign's file, so the only unique content was 6
spill-contaminated rows.

**Two lessons, both cheap and both missed.**
1. The ladder is a PRECONDITION for every long-context arm, not a separate
   result to report at the end. It knew the answer before the arm was queued.
2. A cumulative average is the wrong instrument for detecting a cliff. Any
   progress display that averages over the whole run will show a reassuring
   number while the marginal cost explodes. Consistent with SS 3: the
   instrument was wrong before the system was.

---

## 4c. Limitations the paper must state

Written as limitations rather than left to be discovered in review. Each is a
consequence of the hardware or of a known defect, not of an oversight, and each
has the evidence for why it was handled this way.

### 4c.1 Batch and concurrency are NOT measured

The systems numbers are single-request: TTFT, decode throughput, peak memory,
and multi-query prefix reuse — all at batch size 1. There is no
requests-per-second-under-concurrency figure, which a serving paper would
normally carry.

**Why it was not run.** The DKV wrapper is session-based, and the repo carries
`colab/repro_batch_engine_corruption.py` — a standing reproducer for corruption
on the batched engine path. Numbers from a path with an open correctness bug
are worse than no numbers: they would either flatter DKV (if the corruption
costs accuracy that a throughput table does not show) or understate it (if the
workaround costs speed), and there is no way to tell which from the throughput
figure alone.

**What the paper should say.** That DKV's serving evaluation is single-request,
that batching is unvalidated on this path, and that concurrent throughput is
future work — not that it was measured and found acceptable.

The multi-query experiment (§1.4 / `bench_multiquery_cuda.py`) is the honest
partial substitute: it measures *sequential* reuse of one compressed context
across many queries, which is the property the architecture actually claims,
without asserting anything about parallel requests.

### 4c.2 Sample counts are ~10x below each benchmark's own standard

| benchmark | its standard | used here |
|---|---|---|
| LongBench | ~200 per task | 20 (50 for the four claim-carrying arms) |
| RULER | 500 per task | 20 |

**Consequence, stated precisely.** The paired bootstrap keeps *within-study*
comparisons valid — that is what licenses "SnapKV is not resolved from dense"
and "DKV/high is worse than dense" as claims. What it does NOT license is
placing these absolute numbers beside published LongBench or RULER tables as
though they were comparable; the intervals here are far wider than a
500-sample run's.

Forced by a single 12 GB card and a matrix that already runs ~18 GPU-hours.
The four arms that carry claims were re-run at n=50 to halve their intervals,
because `dkv/high` at −2.72 [−5.19, −0.39] was close enough to zero that the
sample size, not the effect, was the weakest part of the claim.

### 4c.3 The max-context result is architecture-conditional

98,304 vs 49,152 is measured on hybrid models. On dense-GQA granite DKV's
slope matches dense's and its ceiling does not exceed dense's (§1.2). The
cause is identified and is a defect rather than a property: on granite the
dense KV is never released, so peak = dense peak + pool (§4.1 arithmetic).
Fixing it would extend the claim, and granite compresses *better* (4.1x vs
3.4x), so the ceiling is not what limits it.

### 4c.4 One competitor at RULER lengths

RULER carries SnapKV only; H2O, StreamingLLM and KIVI are characterised at 12k
and extrapolated. Reasons and costs in §4b.

### 4c.5 `noremat` returns NaN

The project-then-attend Triton path (`DKV_REMAT_CACHE=0`) produces NaN logits
even with the attention scale corrected. It is not the shipped path — the
default `remat` route is exact against a dense control — but it is an open
correctness defect on a supported configuration and should be named, not
omitted. Three suspects already eliminated by measurement (§5).

---

## 5. Open

- **`noremat` returns NaN.** The project-then-attend Triton path
  (`DKV_REMAT_CACHE=0`) still produces NaN logits with the scale corrected.
  Eliminated by measurement: **not** the gathered pool inputs (all 40 layers
  finite), **not** the combined kernel output (`out_nan=0`), **not** the chunk
  reduction (`num_chunks=1`, so it is skipped). Downstream of all three. Not the
  shipped path — `baseline` uses remat and is exact.
- **Mechanism for the peak-memory split** between hybrid and dense-GQA (§1.2).
- **Can eviction reach DKV's contexts?** Ladder run queued (§1.2).
- **Capacity sweep** — residual 512/1024 and rank 64, on qasper and
  multifieldqa only. Diagnostics; anything that earns its cost becomes a preset
  change with the measurement attached.

---

## 6. Corrections to the campaign's starting assumptions

Recorded because each was checked rather than inherited:

| assumption | what the code says |
|---|---|
| "The repo has no competitor baselines" | `run_a100_paper_experiments.py` already had faithful SnapKV (real attention observation + GQA head pooling + max-pool clustering), StreamingLLM, KIVI and INT8-KV. Extracted to `benchmarks/kv_baselines.py`. |
| "NF4 is the CUDA default" | Auto-enabled for preset **`low` only** (`hf_dkv_wrapper.py:711`, `cli.py:852`). At mid/high an unset value loads fp16. Every arm must pass it explicitly. |
| "LongBench is a stub" | `ACTIVE_RUNTIME/run_longbench.py` uses the real dataset, but is MLX-oriented **and** broken on `datasets` 4.x, which refuses script-based datasets. |
| "MSVC must be installed" | Build Tools 2022 was already installed; only the environment was missing. |
| "DKV engages at 8k" | Just above 4,096 (§4.4). |
