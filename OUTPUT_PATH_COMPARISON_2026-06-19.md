# diffkv_native vs ACTIVE_RUNTIME — Output-Path Comparison (long-context gibberish)

**Date:** 2026-06-19
**Author:** read-only audit (no code changed in either tree)
**Scope:** the systems that actually produce decode-time output, traced end-to-end on both
sides, with emphasis on **long context**. This is a *findings* document — nothing here is
fixed. Issues unrelated to output that I noticed in the files I read are listed in §6.

> ## ⚠️ CORRECTION (appended 2026-06-19, after first draft)
>
> **§0 below was WRONG and is retained only for honesty. The memory was right: on this Mac
> the live ACTIVE_RUNTIME backend IS the MLX wrapper, not the HF/PyTorch one.**
>
> `serving/cli.py` and the gateway import `DiffKVHFWrapper`, but that name is a **conditional
> alias** (`hf_diffkv_wrapper.py:1212-1214`): on `darwin` + `mlx` installed + `DIFFKV_FORCE_PYTORCH`
> unset, `DiffKVHFWrapper = MLXDiffKVWrapper`. MLX **is** installed in `diffkv_venv`, so the
> live class is `MLXDiffKVWrapper`. The `PyTorchDiffKVHFWrapper.generate()` I quoted all over §3
> is the **fallback that does not run here**. My grep for `MLXDiffKVWrapper` instantiation missed
> the alias.
>
> **What this changes:**
> - **§3.1 (routing coverage) survives and is the headline — and is even bigger than written.**
>   The MLX reference `execute_decode_attention` attends **ALL compressed blocks** (`comp_*` up to
>   `max_blocks`) + the dense window, with **no `srl_k` routing cap at decode**
>   (`mlx_diffkv_wrapper.py:737-763`, `compute_decode_attention_static`). C++ routes down to a
>   **fixed 64 blocks**. So it's "64 vs ALL", not "64 vs ~200". This is the dominant long-context
>   cause.
> - **§3.4, §3.6, §3.7 are NOT divergences.** Against MLX, the C++ is faithful: MLX's factual query
>   uses KEY-proxy / `threshold=0.3` / `active_slots=None` / one-step lag (`mlx:868-876`) — exactly
>   what C++ does; neighbor thresholds 0.35/0.50 match (`mlx:889/895`); rep penalty over all tokens
>   matches (`mlx:1244`). The C++ comments citing MLX were correct.
> - **§3.2 (factual store dormant in turn 1) is NOT a divergence either.** MLX's `finalize_srl_index`
>   is a no-op (`mlx:595-596`) and nothing in the MLX wrapper ever calls `factual_store.build()`, so
>   MLX's factual store is also inactive in single-turn decode. MLX stays coherent anyway *because it
>   attends all blocks* (§3.1) — it doesn't rely on the factual stack for general coherence.
> - **§3.3, §3.5, §3.8** — re-evaluate against MLX; mostly faithful or moot (factual-gated).
>
> **Corrected bottom line:** the long-context gibberish is dominated by **§3.1** — C++'s SRL decode
> routing caps attention at 64 blocks while the MLX reference attends every compressed block. The
> §6 "other issues" still stand. The rest of §3 (originally framed as HF mismatches) is largely the
> C++ *correctly* matching MLX.
>
> ---
>
> *(Original first-draft note, now superseded:)* This is a fresh comparison against current code.

---

## 0. The single most important correction first: which ACTIVE_RUNTIME is the reference?

The prior audits and the saved memory repeatedly assert that *"on Mac, ACTIVE_RUNTIME runs
the MLX backend (`mlx_diffkv_wrapper.py`) — that is the true behavioral reference."*

**That is not what the runnable code does.** In the current tree:

- `ACTIVE_RUNTIME/serving/cli.py` → `run_direct_mode()` imports **`hf_diffkv_wrapper.DiffKVHFWrapper`**
  + `batch_engine.ContinuousBatchEngine` (cli.py:406-407, 510).
- `ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py` also instantiates
  **`DiffKVHFWrapper`** (gateway:854, 986).
- `MLXDiffKVWrapper` is **not instantiated anywhere** outside `ACTIVE_RUNTIME/scratch/*`.
  (`grep -rn MLXDiffKVWrapper ACTIVE_RUNTIME` → only the class def + scratch tests.)

So the ACTIVE_RUNTIME that the user runs and calls *"works fine"* is the **HF wrapper on
MPS**, whose decode attention is `runtime/diffkv_attention.py` and whose generation/biasing
loop is `hf_diffkv_wrapper.generate()` (+ `batch_engine.py` for the batched gateway path).

This matters enormously, because **`diffkv_native` was tuned to match the MLX constants, not
the HF ones.** Several C++ comments literally say *"to match the MLX reference
(mlx_diffkv_wrapper.py:…)"* while the file that's actually the reconstruction target is the
1:1 same-named `runtime/diffkv_attention.py`. The HF reference uses **different numbers**
(thresholds, active-slot filtering, query vector, build timing). The C++ inherited MLX's
looser/different constants, and on C++'s lossier sparse base that drift compounds in long
context. Details in §3.

The clean 1:1 reconstruction mapping (verified by file structure):

| diffkv_native (C++) | ACTIVE_RUNTIME live reference (Python/HF) |
|---|---|
| `src/main.cpp` decode loop | `serving/hf_diffkv_wrapper.py::generate()` (+ `serving/batch_engine.py` decode branch) |
| `runtime/diffkv_attention.cpp` / `.mm` | `runtime/diffkv_attention.py` |
| `native_core/kv_runtime_manager.cpp` | `native_core/kv_runtime_manager.py` |
| `native_core/srl/query_router.cpp/.hpp` | `native_core/srl/query_router.py` |
| `native_core/srl/factual_store.cpp` | `native_core/srl/factual_store.py` |
| `native_core/srl/factual_alignment.hpp` | `native_core/srl/factual_alignment.py` |
| `native_core/streaming_sparse_ingest.cpp` | `native_core/streaming_sparse_ingest.py` |

---

## 1. TL;DR — what most likely makes long-context output gibberish

The user's scenario (paste a long document, ask one question) is a **single-turn** request.
Filtering all the divergences below by *"is this even active in turn 1?"* leaves a short list.
The two dominant, *active-in-turn-1* divergences are:

1. **Routing coverage is fixed in C++, adaptive in Python (§3.1).**
   C++ attends a *constant* `srl_k_keep` blocks (= **64** after the N4.2 floor, ≈1024 tokens)
   no matter how long the context is. The HF reference attends `adaptive_k` =
   `max(20, 0.15·N_total)` up to **200** blocks — i.e. it *scales with context*. On a
   ~24k-token prompt (~1500 blocks) Python attends ~200 blocks (~3200 tokens) while C++
   attends 64 blocks (~1024 tokens). The model literally sees ~⅓ the context, and the
   fraction it sees *shrinks* as the document grows. This is exactly a "fine when short,
   gibberish when long" signature.

2. **The factual / grounding stack is dormant in C++ turn 1 (§3.2).**
   Python builds the `FactualExactStore` **before** decode (`finalize_srl_index`, called at
   `hf_diffkv_wrapper.py:837` *prior* to the generate loop). C++ builds it **after** the
   decode loop finishes (`main.cpp:4179-4204`, post-`commit_turn`). So in a single-turn
   request, **every factual guardrail in C++ is a no-op**: factual +7 bias, +10 transition
   bias, −3.5 anti-hallucination penalty, and the VSL logit mask all gate on
   `factual_store.entries` / `current_step_factual_*`, which are empty. Python runs all of
   them from the first generated token. So Python's decode is *grounded to the source* and
   *penalizes off-source tokens*; C++'s decode free-runs over a lossy 64-block attention
   base with **no grounding signal at all**.

Everything else in §3 (query thresholds, neighbor weights, VSL exemption, factual K/V
injection) is real divergence but is **only active in turn 2+** because it also depends on a
populated factual store. They are worth fixing for multi-turn correctness but are *not* the
turn-1 long-context cause.

Supporting/secondary, active in turn 1: the sparse-decode engage threshold is **2048** in C++
vs **4096** in Python (§3.3) — C++ goes lossy-sparse twice as early; and the C++ attention
base is inherently lossy sparse vs Python's larger-coverage sparse (the "architectural
ceiling").

---

## 2. How I traced it

- Confirmed both `cli.py run_direct_mode` entry points: native spawns the C++ binary
  `build/diffkv_native` via subprocess (`diffkv_native/serving/cli.py:307-309`); ACTIVE_RUNTIME
  imports the HF wrapper in-process (`ACTIVE_RUNTIME/serving/cli.py:406-407`).
- Read the C++ decode loop `main.cpp` ≈ lines 2580–4210, the C++ decode attention
  `runtime/diffkv_attention.cpp`, the C++ router `kv_runtime_manager.cpp::route_decode_slots`
  + `query_router.*`.
- Read the HF reference `hf_diffkv_wrapper.py::generate()` (≈820–1162), `batch_engine.py`
  (`_step`, `_sample`, decode branch), `runtime/diffkv_attention.py` (decode + factual
  query, ≈462–860), `query_router.py` (`route_query_fixed_k`, `adaptive_k`, `two_level_gate`),
  `factual_store.py::query/build`, `factual_alignment.py::get_allowed_tokens_vsl`,
  `kv_runtime_manager.py::finalize_srl_index`.
- The uncommitted `git diff` was reviewed (it is mostly perf: batched anchor upload, cached
  pre-rotation, and a pool auto-expand fix; see §5).

---

## 3. Output-path divergences (ranked by long-context impact)

Legend: **[T1]** = active in single-turn long context (the user's case). **[T2+]** = only
bites once a factual store exists (multi-turn, or if turn-1 build is moved earlier).

### 3.1 [T1] Sparse routing coverage: fixed `srl_k_keep` vs adaptive `k` — **primary**

- **C++:** decode attention attends `selected_slots = anchor_screen(candidate_slots, srl_k_keep)`
  (`main.cpp:730`), and `current_k = min(srl_k_keep, active_slot)` (`main.cpp:3077`).
  `srl_k_keep` defaults to 16 and is raised by the N4.2 floor to
  `max(16, 1024/micro_block_size)` = **64** (`main.cpp:1351`). It is then only clamped down by
  `min(srl_k_keep, n_slots)` (`main.cpp:1479`). **It never scales up with context length.**
  The `route_decode_slots` N4.3 channel scaling (`kv_runtime_manager.cpp:308-311`) only widens
  the *candidate* pool fed into `anchor_screen`; the final attended count is still capped at
  `srl_k_keep`.
- **Python (HF/MPS):** decode routes via `route_query_fixed_k` (`diffkv_attention.py:537`),
  which calls `route_query` → `adaptive_k`. On MPS `route_query_fixed_k` returns the *full*
  adaptive result without truncation (`query_router.py:801-803`: `if Q.device.type == "mps":
  return selected`). `adaptive_k` (`query_router.py:150-196`) sets
  `k_max = min(200, N_total)` and `k_min = min(max(20, int(0.15·N_total)), k_max)`, then scales
  by entropy and a `C_active` cluster boost. **Net: attended blocks grow to ≥15% of all
  blocks, up to 200.**
- **Effect:** at ~1500 blocks, Python ≈ 200 blocks vs C++ = 64 blocks attended; and as the
  document grows the C++ fraction keeps shrinking while Python's floor tracks 15%. This is the
  clearest "degrades specifically in long context" mechanism.
- Also note `route_decode_slots` (`kv_runtime_manager.cpp:283`) uses fixed per-channel budgets
  (recency/lexical/graph/semantic), whereas Python distributes `K` across channels by
  fraction (`_LEX_FRAC/_SEM_FRAC/_GRAPH_FRAC`, `query_router.py:234/294/516`). Different
  candidate composition, but secondary to the final-count cap above.

### 3.2 [T1] Factual store built post-decode in C++, pre-decode in Python — **primary**

- **Python:** `hf_diffkv_wrapper.generate()` calls `self.manager.finalize_srl_index(...)`
  at line **837**, *before* the `for _ in range(max_new_tokens)` loop. `finalize_srl_index`
  builds the `FactualExactStore` (`kv_runtime_manager.py` finalize body, ≈171-191:
  `factual_store.build(...)`, `self._factual_stores[session_id] = factual_store`).
  ⇒ turn-1 decode has the full factual stack live.
- **C++:** the background SRL thread explicitly **skips** the factual build
  (`main.cpp:2656`: *"NOTE: factual_store.build() is intentionally NOT called here"*, with a
  RAM-pressure justification at 2658-2660). The only `factual_store.build()` on the live path
  is **after** the decode loop and `commit_turn` (`main.cpp:4179-4204`, comment claims parity
  with `kv_runtime_manager.py:955`).
- **Consequence — all of these become no-ops in C++ turn 1** (each guards on an empty store):
  - factual +7 bias `main.cpp:3421` (guard `current_step_factual_tokens` empty)
  - VSL +7 active-candidate boost `main.cpp:3463`
  - −3.5 anti-hallucination `main.cpp:3472` (guard `current_step_factual_tokens` + sim≥0.4)
  - +10 transition bias `main.cpp:3487`
  - `sfa_active` / VSL logit mask `main.cpp:3594-3624`
  - end-of-step factual query `main.cpp:3731` (guard `factual_store.entries` empty)
- The C++ comment justifying "post-decode build" as ACTIVE_RUNTIME parity is **a misreading**:
  Python builds the store *both* in `finalize_srl_index` (pre-decode) *and* later. The
  pre-decode build is the one that governs turn-1 generation. So C++ loses the grounding that
  keeps Python coherent over long, low-signal text.

### 3.3 [T1] Sparse-decode engage threshold 2048 (C++) vs 4096 (Python)

- **Python:** `_get_engage_threshold()` = **4096** (`diffkv_attention.py:75`); below it, decode
  stays pure-dense/exact.
- **C++:** `engage_threshold` defaults to **2048** (`main.cpp:644`, `main.cpp:2348`);
  `decode_use_sparse = (L >= 2048)` (`main.cpp:2352`). The ingest side matches at 2048
  deliberately (`streaming_sparse_ingest.cpp:475-486`, F13 note).
- **Effect:** for prompts in **[2048, 4096)**, C++ already runs lossy sparse decode while
  Python is still exact-dense. Widens the band where C++ is degraded. (Memory records this as a
  deliberate RAM-positive choice, so flagging, not asserting a bug.)

### 3.4 [T2+] Factual-store `query()` call-site parameters diverge (C++ follows MLX, not HF)

Same `query()` semantics on both sides, but the **caller** passes different arguments:

| Parameter | HF reference `diffkv_attention.py:771-777` | C++ `main.cpp:3808-3817` |
|---|---|---|
| Query vector `Q` | real **`raw_q`** (unrotated query states, layer 0) | **`decode_k[0]`** — the *KEY* of the just-decoded token, used as a proxy |
| Timing | **inside** the forward pass → same-step | **end of step** → applied next step (explicit *"one-step lag"*, `main.cpp:3729`) |
| `threshold` | **0.50** | **0.30** |
| `active_slots` | `set(block_indices)` (routed slots only) | **`nullptr`** (search all entries) |
| 1-hop / 2-hop inject | **0.45 / 0.65** (`diffkv_attention.py:795/807`) | **0.35 / 0.50** (`main.cpp:3865/3875`, commented *"per MLX"*) |

C++ is uniformly **more permissive** (lower threshold, no slot filter, lower neighbor bars) and
uses a *different vector* (key-proxy, lagged) than the live HF reference's real-query/same-step.
On a lossy base this over-injects spurious "factual" tokens and over-fires the bias/mask stack.
Comments at `main.cpp:3804-3807` confirm these were set to match `mlx_diffkv_wrapper.py:874-875`,
not the HF path.

### 3.5 [T2+] VSL logit-mask exempts factual tokens in C++; HF does not

- **C++:** when `sfa_active`, the mask loop additionally skips factual tokens —
  `if (allowed.count(i) == 0 && fact_toks.count(i) == 0)` (`main.cpp:3616`, the restored "F25"
  exemption).
- **HF:** masks everything not in `allowed_ids` — `mask[list(allowed_ids)] = False; logits[0,
  mask] = -65000` (`hf_diffkv_wrapper.py:1020-1025`). `get_allowed_tokens_vsl` under an active
  lock returns only `helpers ∪ {suffix[0]}` (`factual_alignment.py:207+`); mid-sequence factual
  content is **not** exempt.
- So C++ is materially less strict than HF under a hard lock. Helped NIAH historically, but it
  is a divergence from the live reference. (Moot in turn 1 — `sfa_active` requires a populated
  store.)

### 3.6 [T2+] Anti-hallucination & early-stop thresholds; hard-mask magnitude

- Hard VSL mask value: C++ **−1e10** (`main.cpp:3618`) vs HF **−65000.0** (`hf:1025`). Both
  saturate softmax, cosmetic.
- Factual early-stop gate: C++ `max_sim ≥ 0.4` (`main.cpp:3683`) vs HF `≥ 0.5` (`hf:1063`).
- Anti-hallucination threshold matches (both 0.4); magnitudes match (−3.5); factual/transition
  biases match (+7 / +10). The biasing *constants* are otherwise faithful to HF — the problem is
  the *gating data* (§3.2/§3.4), not the constants.

### 3.7 [T1] Repetition penalty: C++ penalizes all tokens, HF skips non-alphanumeric

- **C++:** penalizes every repeated token in the window incl. punctuation/whitespace
  (`main.cpp:3546-3566`, comment claims MLX parity).
- **HF:** explicitly **skips** non-alphanumeric tokens (`hf_diffkv_wrapper.py:904-912`) to avoid
  suppressing list/format punctuation.
- Direction note: this C++ choice *suppresses* ". . ." spam, so it is a **mitigation**, not a
  cause. Its existence is a tell: the HF reference stays coherent *without* punishing
  punctuation, i.e. HF's coherence comes from coverage + grounding (§3.1/§3.2), not from the rep
  penalty. Worth keeping in mind so the real cause isn't masked by the band-aid.

### 3.8 [T2+] C++ injects factual K/V into attention (3-way LSE combine); HF biases logits only

- C++ `execute_*` does a **3-way** combine: sparse ⊕ **factual K/V** ⊕ dense
  (`diffkv_attention.cpp:680-797`), with MLX-style heuristics `fact_scale = scale/0.12`
  (`:740`) and an `lse_facts += 8·(max_sim−0.4)/0.6` boost (`:760-761`).
- HF's decode attention combines sparse ⊕ dense only; the factual store drives **logit** bias,
  not attention K/V. (This is the MLX-vs-HF architectural difference the memory describes.)
- Only fires when `step_cached_entries` is non-empty ⇒ turn 2+. In turn 1 the C++ combine is
  effectively sparse ⊕ dense, same shape as HF — but over far fewer sparse blocks (§3.1).

---

## 4. What appears faithful / not the cause (verified, so you don't re-chase it)

- **3-way LSE combine math** (`diffkv_attention.cpp:780-797`) is structurally correct
  (max-shifted weights, NaN/-inf guards), consistent with Python's `_combine_outputs`.
- **micro_block_size = 16** on both sides (`main.cpp:1305` ↔ `streaming_sparse_ingest.py:131`).
- **rank = 16** on both (`main.cpp:1481` ↔ serving default).
- **recency window = 512** on both (`main.cpp:2784` ↔ `kv_runtime_manager.py:635`).
- **n-gram loop detection** is identical: every 10 new tokens, 5-gram over last 80, ratio ≥0.35,
  force-stop 40 tokens after detection (`main.cpp:3511-3544` ↔ `hf:871-896`).
- **Sampling** (temperature → top-p nucleus → multinomial) and the dynamic-temperature schedule
  `temp·(1 − max_sim·0.95)` at sim≥0.55 match (`main.cpp:3626-3633`/`sample_logits` ↔
  `hf:994-998`).
- **EOS** = EOG/eos token (`main.cpp:3652` ↔ `hf:1073`).
- **`DIFFKV_NATIVE_ATTN` is OFF by default** (gated, `main.cpp:35/751`), so the known
  native-attn "echo/inflation" bug is not on the live path — it's the CPU/Metal custom op that
  runs. Not the cause.

---

## 5. Notes on the uncommitted working-tree diff (output-relevant)

The current `git diff` is mostly **performance**, plus one behavior change. None of it
introduces an obvious new gibberish source, but for completeness:

- `streaming_sparse_ingest.cpp`: anchor K/V/positions are now **batch-uploaded** to GPU once per
  chunk (min/max slot span) instead of per-slot. Pure perf; host mirrors still written.
- `batch_engine.cpp` + `main.cpp`: prefix-block RoPE rotation is **pre-computed once** into
  `k_rotated_activations` and the current chunk rotated incrementally. Perf/refactor.
- `config.hpp` + `cli.py`: `low` preset `prefill_chunk_size` 256 → **512**. Perf.
- `main.cpp:1263-1267`: removed `std::ios::sync_with_stdio(false)` to fix sentinel/token
  ordering on stdout. Output-plumbing correctness, good.
- `main.cpp:1725-1760` (**behavior**): pool **auto-expand to fit the prompt** + reworked decode
  headroom (cap 512 tokens / 25% of `n_slots`). This is the fix for the previously-reported
  "decode stops after ~4 tokens" slot-exhaustion truncation. It changes *how much* prompt is
  kept, not *quality* — but see §6.4 for an interaction worth checking.

---

## 6. Other issues noticed in the files I read (not the output-quality bug)

These are things I saw while reading the output path. Listed per the request to note every
issue in files I touched, even if unrelated to the gibberish.

1. **Reference attribution is wrong in many C++ comments.** Numerous `main.cpp` /
   `diffkv_attention.cpp` comments cite `mlx_diffkv_wrapper.py:NNN` as "the reference" for
   constants (e.g. `main.cpp:3470,3485,3548,3804-3807,3860`). The live reference is the HF
   wrapper with different values (§3.4). The comments are actively misleading future edits.

2. **`query_router.hpp` routing functions look like dead code on the C++ side.** Memory (N4.1)
   reported `route_query`/`adaptive_k`/`route_query_fixed_k` in `query_router.hpp` have zero
   callers — the live router is `kv_runtime_manager.cpp::route_decode_slots`. I did not re-grep
   every caller this pass, but the live decode path I traced uses `route_decode_slots` +
   in-graph `anchor_screen`, not the `query_router.hpp` adaptive_k. If still uncalled, the C++
   never got an adaptive-k equivalent at all (reinforces §3.1).

3. **Hardcoded absolute paths** in `diffkv_native/serving/cli.py:634-640,855` (model + binary
   paths under `/Users/omchimurkar1/...`). Known tech debt; will break on any other machine/user.

4. **Recency-supplement per-layer position inconsistency (potential, GPU path).**
   In `main.cpp:2788-2811` the dense window is overwritten with the last 512 prompt tokens when
   all blocks compressed, but `dense_start_positions[l] = recency_start` is set **only for
   `l==0`** (`main.cpp:2807-2809`). For layers 1+, `dense_start_positions[l]` keeps its
   block-scan value while `active_k_dense[l]` now holds recency tokens. That field feeds
   `userdata[l].active_slot` (`main.cpp:3092`), used as the **Metal dense-window RoPE base
   position**. The CPU custom op rotates via the shared `active_positions_dense` (correct), so
   this is harmless on CPU but looks wrong for the Metal/`--gpu` path at long context. Worth a
   targeted check; I did not confirm a wrong-output repro.

5. **Two different factual queries historically (N3.1) — verify only one remains.** The decode
   loop's end-of-step query (`main.cpp:3808`) is meant to be the *sole* query, with the
   attention callback only reading `step_cached_entries` (`diffkv_attention.cpp:681-688`). That
   looks consistent now, but the proxy-K + one-step-lag design (§3.4) is structurally weaker
   than HF's in-forward real-Q query regardless.

6. **`commit_turn` prunes blocks by salience after each turn** (`main.cpp:4167`). Not a turn-1
   issue, but in multi-turn long sessions this changes which blocks survive into later turns and
   could interact with the already-narrow C++ coverage (§3.1). Flagging for multi-turn testing.

7. **`max_generate` hard cap = 2048** (`main.cpp:2584`) and headroom cap 512 tokens
   (`main.cpp` uncommitted §5). If a user expects a >512-token answer to a long prompt, the new
   headroom cap (25% of `n_slots`, 512-token ceiling) could clip generation in a way that
   depends on `n_slots`; verify against the auto-expand interaction.

---

## 7. Suggested next reads if you decide to act later (not done here)

- Make C++ routing scale with context: wire an `adaptive_k`-style floor (≥15% of active blocks,
  cap ~200) into the attended-block count, not just the candidate pool (§3.1).
- Decide the factual-store build timing for turn 1: Python builds pre-decode; C++ defers for
  RAM. Either build it pre-decode (accept the RAM spike) or accept that turn-1 has no grounding
  and stop tuning the (dormant) bias stack against it (§3.2).
- Re-tune the factual query call site to the **HF** constants (0.50 / routed `active_slots` /
  real-Q if reachable) rather than MLX, since HF is what "works fine" (§3.4).

---

---

## 8. Fix options for the routing-coverage gap (§3.1) — analysis only, nothing applied

**The gap restated:** MLX decode attends **all** compressed blocks + the dense recency window
with no decode-time routing (`mlx_diffkv_wrapper.py:737-763`). C++ caps decode attention at a
fixed `srl_k_keep` (=64) via `anchor_screen` (`main.cpp:730`, `current_k` `:3077`).

**Why the cap exists (important — it is not a random mistake):** C++ sparse decode attention is
a per-layer CPU custom op (`ggml_map_custom3`, `main.cpp:781` region; `execute_cpu_attention` in
`diffkv_attention.cpp`). Its cost grows with the number of attended blocks × layers × tokens, so
routing down to 64 blocks is what keeps decode TPS usable. MLX can attend *everything* cheaply
because the whole decode is one fused Metal graph. So the 64-block cap is a **direct consequence
of the C++ attention architecture**, and closing the coverage gap trades against decode speed
unless the attention path itself changes. This is the "architectural ceiling" tension, now
precisely located.

### Option A — Zero-code experiment first (do this before any code change)
`srl_k_keep` is env-overridable: `DIFFKV_SRL_K_KEEP=<n>` (`main.cpp:1322`). Re-run the failing
long prompt at, say, 128 / 256 / 512 and watch **both** output quality and TPS.
- If quality recovers as `k` rises → confirms §3.1 is the cause, and quantifies the quality↔speed
  curve for free.
- If it doesn't → the cause is elsewhere (e.g. the lossy compression base or the dense-window
  assembly, §6.4) and no routing change would have helped.
- Cost: none, reversible, no rebuild. **This is the cheapest way to validate the whole report.**

### Option B — Raise the fixed cap
Bump the `srl_k_keep` default / N4.2 floor to a larger constant (128–256).
- **Pro:** trivial, one constant; likely a large quality win for moderate contexts.
- **Con:** still *fixed* — coverage fraction keeps shrinking as the document grows, so very long
  contexts still degrade, just later. Linear TPS cost.

### Option C — Make `srl_k_keep` scale with context (adaptive-k style)
Set attended blocks = `max(64, ceil(0.15 · N_active_blocks))` capped at some ceiling, mirroring
the *shape* of Python's `adaptive_k` (note: that path is the PyTorch fallback's, not MLX's — but
it's the right idea).
- **Pro:** coverage tracks context length, so quality is stable across sizes — addresses the
  "shrinks as it grows" signature directly.
- **Con:** more code; TPS now degrades on long contexts (more blocks = slower per token); still
  below MLX (which is 100%).

### Option D — Match MLX: attend all compressed blocks (remove the decode cap)
Numerically faithful to the reference.
- **Pro:** closes the gap entirely; removes a whole class of long-context quality bugs.
- **Con:** with the **current** per-layer CPU custom-op, decode TPS would crater on long context
  (the very reason routing was added). Only viable if paired with the **native fused attention
  subgraph** (`DIFFKV_NATIVE_ATTN`) — which per the reconstruction log is gated OFF due to the
  unresolved repetitive-input inflation bug. So Option D realistically depends on finishing that
  subgraph first. High effort, highest payoff.

### Option E — Cheap partial mitigation: widen the dense recency window
`DIFFKV_RECENCY_WINDOW` / `recency_window=512` (`main.cpp:2784`). Larger window = more *recent*
context attended exactly, at linear cost.
- **Pro:** trivial, helps prompts where the answer is near the end.
- **Con:** does nothing for retrieval from the *middle* of a long document — orthogonal to §3.1,
  not a real fix.

### Recommendation (analysis, not an instruction)
Run **Option A** first to confirm and quantify. If confirmed, **Option C** is the best
quality/perf balance for a routed architecture; **Option D** is the "true parity with MLX" answer
but is blocked on the native-attention subgraph work. Keep `micro_block_size`/`rank`/recency as-is
(they're faithful, §4).

---

*End of report. No source files in either tree were modified. §0 retained but superseded by the
CORRECTION banner at the top; the operative conclusion is §3.1 + §8.*
