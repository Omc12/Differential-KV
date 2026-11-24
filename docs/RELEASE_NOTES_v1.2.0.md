# 1.2.0 — The Correctness Release

**349 commits · 140 files · +45,750 / −2,869 since v1.1.0**

**Test suite at this tag: 405 passed, 2 skipped, 1 failed** — see Known Limitations.

v1.0.0 and v1.1.0 shipped a runtime whose CUDA path was substantially **inert**. The production Triton decode kernel never launched. The residual router raised on every partial-rotary model. The content-aware boost had never fired. Thirteen per-token gates were dead on every hybrid model. `DKV_COMPRESSED_DECODE` had no CUDA reader at all.

This release makes those paths run, aligns the CUDA and MLX engines, and re-measures everything that was reported on top of them — including retracting the results that did not survive.

> **Headline:** natural-text needle recall goes **2/12 → 12/12 at 8k** and **3/12 → 12/12 at 32k**, equal to the dense control at both context lengths, with **no budget raised**.

---

## 🛠️ What's Changed

### 1. Code That Never Ran

| Defect | Consequence |
|---|---|
| Triton production decode kernel never launched (autotune broke it; `try/except` fell back silently) | Every "fused decode" measurement was the unfused path |
| Residual router **raised** on every partial-rotary model | Routing had never run on Qwen3.5 or any partial-RoPE architecture |
| `DKV_COMPRESSED_DECODE` / `_MIN_CTX` had **no CUDA reader** | 32× divergence between documented and actual behaviour |
| Content-aware residual boost had **never fired** | Off-by-one in its gate |
| 13 once-per-token gates dead on **every** hybrid model | Hybrid models ran an unintended configuration |
| `DKV_REMAT_CACHE` unreachable on the production path | Re-materialisation cache had a **0%** hit rate |
| Routing ran on layer 0 only; a 40-block gate with no MLX counterpart | Router off entirely at 2k and 8k |
| `DKV_SPARSE_BIAS='auto'` never adaptive on CUDA | A constant `+2.0` |
| Edge propagation never ran in the production router | — |

### 2. Reconstruction & RoPE Frames

* **Compressed keys were never rotated** — prefill stored pre-RoPE keys while the decoder was told the pool held post-RoPE ones. Four further prefill capture sites stored `K` in the wrong frame.
* **Both CUDA read-time residual rotations used the wrong position.**
* **RoPE tables reshaped by `head_dim` instead of `rotary_dim`**; fp16 RoPE tables read as `float32` caused **nondeterministic decode**; partial RoPE fixed on the CUDA dense-window path (**Qwen3.5-2B could not decode at all** before this).
* **All three residual readers now SUBSTITUTE rather than ADD**, matching MLX — adding onto the lossy twin double-counted them.
* **Re-materialised blocks were missing their anchor row** (one real token dropped per block), and the remat cache reconstructed blocks **without residuals entirely**.
* Residuals were computed against the **fp16** reconstruction while decode reads the **int8** one.
* A skip block was only exact for its **first `max_residual` tokens**.
* The dense-window workspace was sized with the wrong block size, **trimming live context**.

### 3. Pool, Paging & Session State

* A freed pool slot stayed **"occupied" forever** and kept its tiering state, so eviction **corrupted live blocks**.
* **Eviction could zero a block the same decode step was about to read.**
* The batched pool writer did not clear a recycled slot's stratified group.
* **`clear_session` never cleared the decode block cache** — decode read the *previous generation's* pool slots.
* `_MUTATION_OUT_ACTIVE` leaked across sessions as a module global.
* Block metadata was allocated as an **inference tensor**, unwritable after prefill.
* `seq_lens` now describes what was **stored**, not what was offered; the pool is sized by **attended** layers.

### 4. Serving & Sampling

* **The batch engine owns its KV cache — word salad becomes real text.**
* **CUDA stream races** in both the batch-engine decode and prefill streams: the result was read before it finished.
* **Greedy sampling had no NaN guard** — and the engine's logits do contain NaN.
* The dense-only decode path returned **EMPTY attention** on a zero-block step.
* `DKV_DETERMINISTIC` now defaults **ON**: greedy decode at long context was *not* reproducible without it, and the cause is the decode attention's reduction, **not** compression.

### 5. 🎯 Recall on Natural Text

The tiled-haystack NIAH suites in this repo **inflate recall** — the filler is one sentence repeated, so a random code is a colossal outlier the residual budget is all but guaranteed to keep. Refilled from real papers, recall was **2/12**.

The needle was found and **corrupted**, not missed — `Falcon-9427-618`**`5`** for `...618`**`3`**. Qwen splits that code into eleven tokens; residual selection ranked them one at a time and lost the tail, so the model reproduced the captured prefix and invented the rest.

Three changes fix it, **none of which raises a budget**:

1. Residual selection takes a token **RUN** whole or not at all.
2. The router scores the **exact key**, instead of summing two different rotational frames.
3. Whole runs are ordered by whether the **QUERY** asks for them before by how badly they reconstruct — falling back to SnapKV's observation window (last 64 prompt tokens) when no question span can be pinned.

| filler | ctx | dense | DKV before | DKV now |
|---|---|---|---|---|
| natural text | 8k | 12/12 | 2/12 | **12/12** |
| natural text | 32k | 12/12 | 3/12 | **12/12** |

`multifact_eval_cuda` passes **9/9 for the first time** (relational 4/4, multi-needle 3/3); synthesis **13.3 → 30.0**, then **→ 46.7** with rarity-aware capture. Cost: **~4% of prefill**, decode and memory unchanged.

Reproduce with `colab/needle_depth_sweep.py --filler natural`.

---

## ✨ New Features

### Physical 4-Bit Quantized Residuals (default)
True asymmetric group quantization (`group_size=64, bits=4`) packed into `uint32`/`int32` on **both** MLX and CUDA/Triton. `DKV_RESIDUAL_QUANT=int4` is now the default (**3.56×** compression). Persistent fp16 residual buffers are `None` in INT4 mode; only routed top-K blocks are dequantized into transient scratchpads.

> **R=256 INT4 residuals (112.9 MB @16k) cost 44% less than the old R=128 FP16 residuals (200.7 MB) — twice the exact tokens, in less memory.**

### Content-Aware / Rarity-Aware Residual Selection
Vanilla DKV selects residuals by pure $L_2$ error, so on JSON, code and logs the high-norm delimiters (`{`, `}`, `:`, `\n`) eat every slot and starve entity names and passkeys. Now IDF-weighted, with per-class boosts — **digits 20.0×, entity names 14.6×, rare terms 7.3×** — and a punctuation-exclusion guard. Shipped on CUDA, MLX and the C++ core.

### Unrotated Key Storage (`mid` / `high` / `ultra`)
This is what buys dense parity. Qwen3.5-2B @32k:

| benchmark | dense | `mid` (unrotated) | rotated |
|---|---|---|---|
| digit-table, 24 seeds | 24/24 | **24/24** | 14/24 |
| linkbench `direct`, 48 seeds | 47/48 | **47/48** | 40/48 |

Storing keys rotated costs **42% of exact digit recall** — invoices, logs, IDs, any table. Not free: read-time rotation costs ~11% of decode on a hybrid model, ~27% on a dense-attention one. `low` and `DKV_ROTATED_POOL=1` take the speed instead.

### Routed CUDA-Graph Decode (`DKV_FAST_DECODE=1`, default)
Gated per session to whether a graph will actually be captured. **1.41–1.48×** on replay where routing is non-selective; 25% at 4k and 9% at 8k end-to-end. Not bit-identical to eager (one ULP at step 1, greedy argmax flips ~step 32) — accuracy unaffected (digit-table 24/24, linkbench 23/48, equal to eager *and* dense). Use `DKV_FAST_DECODE=0 DKV_DETERMINISTIC=1` for token-for-token comparisons.

### Also new
* **`ultra` preset** — matches the dense control on synthesis.
* **Re-materialisation cache ported to CUDA** from MLX, on by default.
* **Shared low-rank bases** (`DKV_SHARED_BASIS=1`, opt-in) — pool **91.4 → 69.8 MB (−23.6%)**, 2.58× sharing, zero forced joins. **A capacity gain, not a memory saving**: peak moves only 0.6–1.8%, because weights dominate. The pool now **refuses** the rotated-pool and q4_0 combinations at construction, since both fail invisibly with pool MB unchanged.
* **Gemma 4 E2B hybrid architecture support** (selective global-only DKV patching).
* **Port to `transformers` 5.14.1 `AttentionInterface` registry.**
* **`DKV_POOL_ATTENDED_ONLY=1`** — saves 406.9 MB of a 542.5 MB pool at 11.4k on Qwen3.5 (6 of 24 layers attended).
* **~30 new test modules** and an extensive `colab/` diagnostic suite (MLX↔CUDA parity, logit fidelity, needle depth sweeps, linkbench, tablebench, batch-engine corruption reproducers).

---

## 🔧 Changed Defaults

| Setting | Was | Now | Why |
|---|:---:|:---:|---|
| MLX `block_size` | 256 | **1024** | linkbench **9/24 → 24/24 (= dense)**, needles 6/6 either way, session pool **135.6 → 60.0 MB**, pool vs dense KV **0.95× → 0.28× (3.61× smaller)**. At 256 the fixed 128-token residual budget stored *half of every block verbatim*. |
| `DKV_RESIDUAL_QUANT` | `none` | **`int4`** | 3.56× compression |
| Residual ladder `R` | flat | **64 / 128 / 256** | `low` / `mid` / `high` |
| `DKV_DECODE_CACHE_INTERVAL` | 16 | **4** | The routed set is FROZEN for the interval, so a late-routed needle stays invisible that long. The old default's claimed 15% speed win **does not reproduce**: paired and A/A-calibrated, +0.095 ms/token, 95% CI [−0.486, +0.677]. |
| `DKV_DETERMINISTIC` | `0` | **`1`** | Greedy decode was not reproducible without it |
| `DKV_FAST_DECODE` | — | **`1`** | Routed CUDA-graph decode |

---

## ⚠️ Retracted Results

Withdrawn rather than quietly dropped:

* **The rank sweep** — randomised-SVD noise. `svd_energy` is the dial, and which of `rank` and `svd_energy` binds depends on the **input**, not the config.
* **"Algorithmic, not dispatch"** — MLX's re-materialisation cache had simply never been ported.
* **32k decode results** — that path is nondeterministic without `DKV_DETERMINISTIC`; the numbers were not comparable.
* **The `ultra` unpatch control** — invalid; the component it claimed to establish is **not** established.
* **Two speed flags that shipped without a speed measurement**, and a decode regression that was a delta between two unmeasurable numbers.

---

## 🚧 Known Limitations

* `dkv_native/` (standalone C++ engine) remains **experimental / WIP**.
* `DKV_SHARED_BASIS` is opt-in and, on the models measured, has **no operating point where it currently pays** — MLX decode is 0.5–0.65× in paired rounds.
* `DKV_ROTATED_POOL=0` is measured **inert on MLX** (same score *and the same predicted answer* on all 24 linkbench seeds) while costing ~39% of decode. Off by default there.
* **NIAH numbers on a tiled haystack are inflated.** Judge recall with `colab/needle_depth_sweep.py --filler natural`.
* **`DKV_RESIDUAL_RUN_RESERVE=query` (strict) does not honour its own contract when no query is pinned.** In `_select_residual_rows` (`ACTIVE_RUNTIME/native_core/compression/lowrank.py:405`) the `elif not _has_priority: pass` branch shadows the `elif _mode == "query": return [], set()` branch below it, so with no priority information strict `query` mode reserves **every** run — silently behaving as `all`, the opposite of the documented "reserve nothing". **The shipped `query_first` default is not affected** (it cannot reach that branch), so no default behaviour is wrong — but any A/B run with `DKV_RESIDUAL_RUN_RESERVE=query` against unprioritised runs was comparing `all` to `all`. `test_unasked_runs_do_not_claim_slots` is left **failing deliberately** rather than skipped.

---

## 📦 Installation

```bash
pip install git+https://github.com/Omc12/Differential-KV.git@v1.2.0
```

Or from source, with submodules:

```bash
git clone --recurse-submodules --branch v1.2.0 https://github.com/Omc12/Differential-KV.git
cd Differential-KV && make setup && make chat
```

**Full changelog:** [`CHANGELOG.md`](CHANGELOG.md) · **Compare:** https://github.com/Omc12/Differential-KV/compare/v1.1.0...v1.2.0
