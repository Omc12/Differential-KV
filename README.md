# Differential-KV (DKV): Routed, Compressed KV Cache for Long-Context Inference

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21539110.svg)](https://doi.org/10.5281/zenodo.21539110)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20WSL2-lightgrey.svg)](BUILD.md)
[![Backend](https://img.shields.io/badge/Backend-PyTorch%20%2B%20Triton%20(CUDA)%20%7C%20vLLM-blueviolet.svg)](ACTIVE_RUNTIME/README.md)

**Differential-KV (DKV)** keeps a long prompt's KV cache in compressed blocks and,
at each decode step, reads only the blocks the query routes to plus an exact
recent window. Each block keeps one exact anchor token; how the other rows are
stored is a pluggable **codec**: by default KIVI's 4-bit layout, optionally a
joint low-rank factor with exact residual rows (about a third of the bytes, at
lower quality), or any codec you write in two functions. It runs on Hugging Face
`transformers` on CUDA, and as a [vLLM attention backend](integrations/vllm/).

> **Status (2026-11).** The CUDA runtime (`ACTIVE_RUNTIME/`, PyTorch + Triton)
> is the maintained path and the source of every current number below. The
> Apple Silicon (MLX) runtime and the C++ engine (`dkv_native/`) are earlier
> work, not kept current; sections measured on them are labelled.

ResearchGate: https://www.researchgate.net/publication/410864213

Author: **Om Chimurkar** (Newton School of Technology, Rishihood University)  
Paper: https://doi.org/10.5281/zenodo.21539110

***Try a demo here***: https://huggingface.co/spaces/Om1232/Differential-KV

---

## 📌 Core Architecture & Paper Overview

The Key-Value (KV) cache is the primary memory bottleneck in long-context LLM inference: its memory footprint scales linearly $O(L)$ with sequence length $L$, causing commodity hardware to run out of memory (OOM) during prefill or generation long before model weights exhaust VRAM.

Differential-KV partitions the KV cache into fixed blocks (1,024 tokens by default on CUDA) and keeps, per block:

1. **Anchor Token ($a_k, a_v$):** The first token in the block, preserved in exact precision.
2. **The block's other rows, in a codec** (`DKV_STORE`, see [Pluggable KV codecs](#-pluggable-kv-codecs)):
   * **`codes` (default):** KIVI's layout, 4-bit. Keys are quantized per channel over 32-token groups, as deltas from the anchor (a per-channel range does not change under that shift). Values are quantized per token over 32-channel groups, as they are.
   * **`lowrank` (opt-in, `DKV_LOWRANK=1`):** a joint $K \mid V$ truncated SVD of the anchor-relative deltas plus **exact residual rows** for the tokens the factor reconstructs worst (whole digit/name runs kept together). About a third of the codes store's bytes, at clearly lower quality (table below).
3. **Bounded Dense Recency Window:** the most recent tokens, uncompressed.

### Decode & Prefill
* **Routed decode:** each step scores the blocks with the current query (each query head votes by its own softmax over the blocks) and attends the top-$K$ (`DKV_TOPK_BLOCKS`, 16), plus more blocks while the estimated attention mass they hold is under 0.8, plus the window; `DKV_TOPK_BLOCKS=0` reads every block, the quality ceiling. Partial attentions merge by fp32 log-sum-exp.
* **Exact prefill (default):** the prompt is attended exactly and compressed afterwards, so the first token matches dense. `DKV_STREAMING_COMPRESS=1` compresses during prefill instead, which fits longer prompts on a small card at some quality cost.
* **Codec-agnostic machinery:** routing, the paged pool, tiered CPU offload, prefill tiling and the decode kernels never look inside a codec.

---

## 🚀 Quickstart

### macOS / Apple Silicon (MLX)
```bash
# 1. Clone repository with submodules
git clone --recurse-submodules https://github.com/Omc12/Differential-KV.git
cd Differential-KV

# 2. Setup virtual environment and dependencies
make setup

# 3. Launch interactive DKV Chat CLI (downloads default model Qwen/Qwen2.5-1.5B-Instruct on first run)
make chat

# Or specify a custom model:
make chat MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

### Linux / CUDA
```bash
# 1. Setup python virtual environment
make setup

# 2. Build CUDA native extensions or Triton kernel paths
# See BUILD.md for detailed instructions on cuSOLVER/cuBLAS & Triton requirements
```

### Useful Commands (`Makefile`)
| Command | Description |
|---|---|
| `make setup` | Creates `dkv_venv` Python virtualenv & installs required packages |
| `make chat` | Starts interactive terminal CLI in Direct Mode |
| `make serve` | Launches OpenAI-compatible REST API gateway on `http://localhost:8000` |
| `make test` | Runs needle-in-a-haystack (NIAH) recall guardrail tests at 8k & 16k context |

> **NIAH numbers on a TILED haystack are inflated, and this repo's suites all
> use one.** `niah_recall.FILLER` is a single 291-character sentence (38 unique
> words) repeated to length; `validate_cuda_dkv.py` cycles eight. A random code
> in that text is a colossal outlier, so the residual budget — which spends its
> slots on each block's worst-reconstructed tokens — is all but guaranteed to
> keep it. Refill the same prompts from real papers in this repo and hold
> everything else fixed (Qwen2.5-1.5B, `mid`, `block_size` 256, `max_residual`
> 40, 12 depths, dense control at every point):
>
> | filler | ctx | dense | DKV before | DKV now |
> |---|---|---|---|---|
> | natural text | 8k | 12/12 | **2/12** | **12/12** |
> | natural text | 32k | 12/12 | **3/12** | **12/12** |
>
> Tiled filler is unaffected — the same sweep reads 12/12 at both 8k and 32k
> after the change, and `validate_cuda_dkv.py --long` (Qwen3.5-2B, 2k/8k/32k x 3
> depths) stays ALL CHECKS PASSED: 9/9 recall, 9/9 determinism, no fallback.
> Cost: ~4% of prefill, decode and memory unchanged (no budget is raised — the
> same 40 residual slots, spent differently).
>
> The needle was found and CORRUPTED, not missed — `Falcon-9427-618`**`5`** for
> `...618`**`3`**. Qwen splits that code into eleven tokens, residual selection
> ranked tokens one at a time, and the tail of the run lost; the model then
> reproduced exactly the captured prefix and invented the rest. Three things fix
> it, none of which raises a budget: selection takes a token RUN whole or not at
> all; the residual router scores the EXACT key instead of summing two different
> rotational frames; and whole runs are ordered by whether the QUERY asks for
> them before by how badly they reconstruct, and the reservation is scoped to
> the runs the query asks for — falling back to the last 64 prompt tokens as the
> query when no question span can be pinned. Recall is now AT
> DENSE on natural text at both contexts, and `multifact_eval_cuda` passes 9/9
> for the first time: relational 4/4, multi-needle 3/3, synthesis 30.0 against a
> pre-existing 13.3. Reproduce with
> `colab/needle_depth_sweep.py --filler natural`; the full account is
> `ACTIVE_RUNTIME/docs/cuda_work_record.md` §4d (the defect) and §4f (the fix).
| `make native` | Compiles high-performance C++ engine (`dkv_native`) with Metal/CUDA support |

---

## 🖥️ CLI Commands & Operating Modes

The Differential-KV terminal interface (`ACTIVE_RUNTIME/serving/cli.py`) supports two primary operational modes: **Direct Execution Mode** and **Client-Server Mode**.

### 1. Operational Modes

#### Direct Mode (Local Inference)
Runs model weights directly in-process via MLX or PyTorch:
```bash
python ACTIVE_RUNTIME/serving/cli.py --model Qwen/Qwen2.5-1.5B-Instruct --preset mid --serving-mode balanced
```

#### Client-Server Mode (Remote / API Gateway)
Connects the CLI UI to a running `make serve` API server instance:
```bash
# Terminal 1: Start API server
make serve MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Terminal 2: Connect CLI client
python ACTIVE_RUNTIME/serving/cli.py --api-url http://localhost:8000/v1
```

### 2. CLI Options & Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID or local directory path |
| `--api-url` | `str` | `None` | API Gateway base URL. When provided, runs CLI in Client Mode |
| `--serving-mode` | `choice` | `balanced` | KV Cache strategy: `lightweight`, `balanced`, `performance`, `long-context`, `fused-sparse` |
| `--preset` | `choice` | `mid` | Fidelity preset: `low`, `mid`, `high`, `ultra` — see the ladder below |
| `--fastdc` | `flag` | **on** | CUDA only. Routed CUDA-graph decode, **now the default**: 25% faster at 4k, 9% at 8k, 5% slower at 16k, neutral at 32k+. Output is a 1-ULP *distribution* difference from eager, not a quality one. `DKV_FAST_DECODE=0` disables. |
| `--rank` | `int` | `32` | SVD rank for KV compression (capped at $d_{\text{head}}$) |
| `--micro-block-size` | `int` | `256` | Number of tokens per compressed KV micro-block ($B_s = 256$) |
| `--batch-size` | `int` | `4` | Maximum continuous batching size for engine |
| `--load-in-4bit` | `flag` | `False` | Load model weights using 4-bit NF4 quantization |
| `--load-in-8bit` | `flag` | `False` | Load model weights using 8-bit quantization |
| `--max-tokens` | `int` | `16384` | Maximum tokens to generate per response |
| `--temperature` | `float` | `0.7` | Sampling temperature |
| `--top-p` | `float` | `0.9` | Top-p nucleus sampling probability |
| `--repetition-penalty` | `float` | `1.15` | Repetition penalty factor |
| `--draft-model` | `str` | `None` | Optional draft model ID for speculative decoding |
| `--max-resident-sessions` | `int` | `4` | Maximum active resident chat sessions held in VRAM |

### 3. The Preset Ladder (Standardized with INT4 Residuals)

| preset | keys stored | energy / rank | residual $R$ (INT4) | memory / block (28L) | pick it when |
|---|---|---|:---:|:---:|---|
| `low` | rotated | 0.999 / 32 | **64 tokens** | 2.02 MB | memory is the binding constraint |
| **`mid`** (default) | **unrotated** | 0.9999 / 64 | **128 tokens** | 4.03 MB | **general use — dense-parity accuracy** |
| `high` | **unrotated** | 0.99999 / 128 | **256 tokens** | 8.06 MB | largest fidelity budget & CAD decode |
| `ultra` | unrotated | 0.9999 / 64 | **256 tokens** | 8.06 MB | dense parity at ultra-long context |

**`mid` stores keys un-rotated, and that is what buys dense parity.** On the two
metrics with power to resolve anything, `mid` now equals dense exactly:

| benchmark (Qwen3.5-2B, 32k) | dense | `mid` | rotated |
|---|---|---|---|
| digit-table, 24 seeds | 24/24 | **24/24** | 14/24 |
| linkbench `chain`, 48 seeds | 23/48 | **23/48** | 21/48 |
| linkbench `direct`, 48 seeds | 47/48 | **47/48** | 40/48 |

Storing keys rotated costs **42% of exact digit recall** — invoices, logs, IDs,
any table — and nothing else recovers it (residual budget, attend-every-block and
residual tiering are all measured inert).

Two things worth knowing before choosing:

* **It is not free, and the cost depends on your model.** Rotating at read is paid
  on every *attended* layer: a hybrid model (Qwen3.5-2B, 6 of 24 attended) pays
  ~11% of decode, a dense-attention one (Qwen2.5-1.5B, 28 of 28) ~27%. Check your
  model's attended-layer count before budgeting.
* **If you need that speed back, `low` still stores rotated keys**, or set
  `DKV_ROTATED_POOL=1` on any preset. You lose the digit accuracy above.
  Rotated is now a *speed* choice rather than a fidelity one, which is why it
  sits on the memory rung and the env flag instead of being scattered up the
  ladder — `high` is the max-fidelity preset and should not be the one losing
  42% of digit recall.
  `DKV_ROTATED_POOL=0` takes the same trade on any preset.

---

---

## 🔬 Core Innovations: Content-Aware Selection & 4-Bit Quantized Residuals

### 1. Content-Aware / Rarity-Aware Residual Selection
> **Architecture Note**: Content-Aware Residual Selection operates on **prompt ingest during block compression**, completely independent from runtime Context-Aware Decoding (CAD) contrastive logits.

Vanilla DKV selects residual tokens purely by mathematical $L_2$ reconstruction error ($\arg\max \|\mathbf{k} - \hat{\mathbf{k}}\|_2$). On structured text, JSON, code, and logs, high-norm delimiters (`{`, `}`, `[`, `]`, `:`, `\n`) dominate $L_2$ error and consume all residual slots, starving rare keywords, entity names, and numeric passkeys.

Differential-KV introduces a **multi-tiered semantic selection engine**:
1. **Token Boost Multipliers**:
   * Numeric Digits / Passkeys: **20.0x multiplier** (`DKV_BOOST_DIGITS`)
   * TitleCase / Entity Names: **14.6x multiplier** (`DKV_BOOST_OWNER`)
   * Rare Technical / Domain Terms: **7.3x multiplier** (`DKV_BOOST_RARE`)
2. **IDF-Weighted Rarity Layer**: Computes inverse document frequency `IDF(t) = ln(max(N, 2) / (count(t) + 0.1))`. Rare tokens above the threshold receive a multiplicative boost to ensure document facts are preserved verbatim. The `+0.1` smoothing keeps a hapax finite, and both engines use this exact form — MLX and the CUDA/C++ path must not diverge here, or they select different residual rows for the same block.
3. **Punctuation Exclusion Guard**: Delimiters and whitespace are strictly barred from receiving rarity boosts.

#### Tunable Selection Dials
| Environment Variable | Default | Description |
|---|:---:|---|
| `DKV_RARITY_CAPTURE` | `1` | `1` enables rarity/content-aware residual selection; `0` reverts to vanilla $L_2$. |
| `DKV_RARITY_WEIGHT` | `1.5` | Multiplicative weight applied to token IDF score. |
| `DKV_RARITY_MIN_IDF` | `2.0` | Minimum token IDF threshold required to receive rarity boosting. |
| `DKV_BOOST_DIGITS` | `20.0` | Error boost multiplier for numeric digit sequences. |
| `DKV_BOOST_OWNER` | `14.6` | Error boost multiplier for entity owner names and title-cased terms. |
| `DKV_BOOST_RARE` | `7.3` | Error boost multiplier for rare prose entities. |

---

### 2. Physical 4-Bit Group-Quantized Residual Buffers (INT4)

To eliminate the memory cost of exact residual storage, Differential-KV implements **true physical asymmetric group quantization** (`group_size=64, bits=4`) across Apple Silicon (MLX) and NVIDIA (CUDA/Triton):

* **Packed Physical Storage**: Residual keys and values are packed into `uint32` / `int32` tensors (`head_dim * bits // 32 = 16` words for $D=128$) alongside half-precision scale and bias vectors (8 bytes).
* **Zero Persistent FP16 Buffers**: In INT4 mode, persistent full-precision residual cache buffers are set to `None`, freeing $3.56x$ physical VRAM.
* **Transient On-the-Fly Dequantization**: Only the gathered top-$K$ routed blocks are dequantized into transient scratchpads during decode attention.
* **$2x$ Residual Capacity in $44\%$ Less Memory**: Storing **$R=256$ INT4 residuals (112.9 MB at 16k)** consumes almost half the memory of the old **$R=128$ FP16 residuals (200.7 MB at 16k)** while keeping twice as many exact tokens.

#### Tunable Quantization Dials
| Environment Variable | Default | Description |
|---|:---:|---|
| `DKV_RESIDUAL_QUANT` | `int8` | Quantization format: `int8` (default, $1.9x$ compression), `int4` ($3.56x$), or `none` (FP16 fallback). |
| `DKV_RESIDUAL_QUANT_GROUP_SIZE` | `64` | Sub-vector quantization group size. |
| `DKV_RESIDUAL_QUANT_BITS` | follows the format | 4 for `int4`, 8 for `int8`. Set it only to vary width without renaming the format. |

**The default was `int4` until 2026-08-31 and the change is measured, not stylistic.**
Residuals are the *exact-copy* tokens — the design drops a residual token's lossy
SVD twin precisely because the residual is meant to be faithful — so quantizing
them coarsely can leave those tokens with no accurate representation anywhere in
the store. On MLX that was catastrophic (4-needle verbatim NIAH at 20k, 48
needles/arm: int4 **0/48**, int8 41/48, fp16 42/48). On CUDA
(`colab/residual_format_niah_cuda.py`, 12 trials x 16k/32k, paired prompts) int4
is only mildly worse — 6 lost needles out of 96 with none gained, sign test
p=0.031 — while **int8 returned the identical per-needle result to fp16 on all 96
paired needles** at half fp16's residual bytes. `DKV_RESIDUAL_QUANT=int4` remains
available and is the right dial to reach for first if VRAM regresses at 32k+.

> Do **not** re-validate a change to this dial with a single-needle sweep. That
> metric is saturated and scored int4 8/9 against fp16 7/9 — it ranks the worse
> format above the better one. Use multi-needle verbatim codes.

---

## 🔌 Pluggable KV codecs

DKV's runtime (routing, the paged pool, tiered offload, prefill tiling, decode)
is independent of how a compressed block's K and V are encoded. That encoding
is a **codec**, and you can swap it with one setting:

```bash
DKV_STORE=codes      # default: KIVI-4 layout (4-bit keys per-channel, values per-token)
DKV_STORE=lowrank    # joint low-rank factor + exact residual rows: ~3x fewer bytes, lower quality
DKV_STORE=hybrid     # 4-bit keys, low-rank values
DKV_STORE=mypackage.mycodec:MY_CODEC    # your own, imported on demand
```

or from Python: `DKVHFWrapper(model_id, config={"codec": MY_CODEC, ...})`.

Writing one takes two functions. `x` is `[blocks, tokens, kv_heads, head_dim]`
float32: a block's keys as deltas from its anchor, or its values as they are
(subtracting an outlier anchor would widen every value row's range):

```python
import torch
from runtime.kv_codecs import register_codec

def encode(x):
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    return {"q": (x / s).round().to(torch.int8), "s": s.half()}

def decode(state):
    return state["q"].float() * state["s"].float()

register_codec("int8", encode=encode, decode=decode)     # DKV_STORE=int8
```

DKV infers the storage layout from one `encode` call and handles allocation,
growth, offload, gathering, tiling and routing. The pair codes both keys and
values; `values=False` codes keys only (values stay in the low-rank factor),
and `encode_values=` / `decode_values=` give values their own pair.
Anything with `encode` and `decode` attributes works the same way: a module, a
class, a namespace. [`ACTIVE_RUNTIME/examples/int8_codec.py`](ACTIVE_RUNTIME/examples/int8_codec.py)
is this codec as a module (`DKV_STORE=examples.int8_codec:INT8`). For more
control, such as a cheap single-row key decode for the router, subclass
`DeltaCodec` in [`runtime/kv_codecs.py`](ACTIVE_RUNTIME/runtime/kv_codecs.py).
`tests/test_kv_codecs.py` checks the contract.
[`examples/lossless_codec.py`](ACTIVE_RUNTIME/examples/lossless_codec.py) stores
everything exactly: run it as an arm to tell whether a gap to dense comes from
the store or from the rest of the pipeline.

## ⚙️ Key Environment Knobs

Differential-KV runtime behaviors can be fine-tuned using environment variables:

| Variable | Default | Scope | Description |
|---|---|---|---|
| `DKV_STORE` | `codes` (since 2026-11-26) | CUDA | Which store compressed history lives in. `codes`: KIVI-4's layout (keys per-channel, values per-token, 4-bit; `DKV_QUANT_BITS=2` for 2-bit) read through DKV's routed, tiled decode. `lowrank`: the joint low-rank K/V factor + int8 residual rows, about a third of the bytes, so more context on a small card, at lower quality. `hybrid`: 4-bit keys, low-rank values. Any other value is a registered or importable codec (see Pluggable KV codecs above). A head_dim that is not a multiple of 32 falls back to `lowrank` unless `codes` was asked for explicitly. |
| `DKV_LOWRANK` | `0` | CUDA | `1` = the one-switch opt-in to the low-rank store (same as `DKV_STORE=lowrank`; an explicit `DKV_STORE` wins). |
| `DKV_RESIDUAL_RUN_SHARE` | `0.5` | CUDA | With no question pinned, the best token run per block is kept whole and further runs may take at most this share of the residual budget; the rest goes to the worst-reconstructed rows. `1` = the old runs-first fill (V15, `docs/IMPROVEMENTS_LOG.md`). |
| `DKV_QUANT_BITS` | `4` | CUDA | Bit width of the `codes` store (`2` for 2-bit). |
| `DKV_STREAMING_COMPRESS` | `0` | CUDA | `1` compresses history during prefill instead of after it, so the prompt never sits in 16 bits whole: longer prompts fit a small card (the two 8B models at 32k on 12.9 GB), at some quality cost on some models. The one prefill switch; exact prefill is the default. |
| `DKV_COMPRESSED_DECODE` | `auto` | MLX | Controls sparse decode (`on`, `off`, `auto`). Auto engages sparse decode at sequence lengths $\ge 8\text{k}$. |
| `DKV_ENGAGE_THRESHOLD` | Budget-gated | Native / MLX | Context length at which sparse decode engages. Keeps dense decode for short contexts where dense fits. |
| `DKV_HIGH_QUALITY_ROUTING` | `0` | Cross-runtime | `0` = Fast bounded-K pruning (attends top-$K$ blocks, context-independent speed). `1` = High-Quality routing (dynamic candidate routing). |
| `DKV_CACHE_LIMIT_GB` | `1` | MLX | Buffer-cache allocation cap in GB (halves peak prefill RAM). |
| `DKV_TOPK_BLOCKS` | `16` | Both engines | Number of compressed micro-blocks routed per decode step. |
| `DKV_MAX_RESIDUAL` | `128` MLX / `40` CUDA | Both engines | Exact residual token rows per block. CUDA `mid`/`ultra`/`low` use 40, `high` keeps 128; MLX is flat 128. The budget measured **inert** on four benchmarks including a digit-table one, and the pool allocates these slots on *every* block, so 40 is a straight saving. |
| `DKV_ROTATED_POOL` | CUDA: `0` on `mid`/`high`/`ultra`, `1` on `low`; MLX: `1` | CUDA / MLX | `1` stores POST-RoPE keys, baking in the position a block held at **compression** time. `0` stores PRE-RoPE keys and rotates them to their absolute positions at read. **Worth it on CUDA** (40/48 → 47/48 distractor retrieval over 48 seeds, matching dense, for 18–24% of decode). **NOT worth it on MLX, measured:** 9/24 either way at 16k over 24 seeds — the same score *and the same predicted answer on all 24 seeds* — while costing ~39% of decode and a second dense-window buffer per attended layer. The knob is implemented and correct (the pool, the SVD basis and the decode logits all demonstrably change), it simply buys nothing here, so it stays **off by default on MLX and is not part of MLX's `ultra`**. On MLX it also requires `DKV_DECODE_CACHE=1` — only that path materialises the keys, so the low-rank scorer **refuses** rather than scoring unrotated keys against a rotated query. Judge it with `colab/linkbench_mlx.py`, never the needle sweep, which has no confusable distractors and cannot see it. |
| `DKV_SHARED_BASIS` | `0` (opt-in) | CUDA; MLX on branch `mlx-shared-basis` | Blocks whose delta subspaces agree read **one** basis row instead of each storing its own $V$. $V$ is 39% of a pool slot and the item adjacent blocks of a document most nearly agree on. Measured on Qwen2.5-1.5B @8k, `mid`, `frac=0.50`: pool **91.4 → 69.8 MB (−23.6%)** with 2.58× sharing, 463 voluntary joins, **zero forced**, retained delta energy **0.969**. `_bytes_per_block` amortises $V$ by the same fraction, so the budget holds proportionally **more blocks** rather than the same context in less memory. **That capacity framing is the only one the measurement supports.** Pool bytes and peak measured in the SAME process on CUDA (`colab/bench_shared_basis_peak.py`, pool summed from its real tensors, peak from `torch.cuda.max_memory_allocated`): at 8k pool 95.9 → 73.2 MB (−23.7%) for a peak of 3619 → 3598 MB (**−0.6%**), and at 32k pool 366.1 → 279.4 MB (−23.7%) for a peak of 4614 → 4533 MB (**−1.8%**). Weights alone are 3087 MB of every one of those peaks, which is why the pool is not where the memory is. MLX independently measured 1.1% and 3.4%; do not quote the −23.6% as a memory saving. **Prefill cost is not resolvable**: paired, interleaved, A/A-calibrated (`colab/bench_prefill_paired.py`, 8 rounds @8k) puts it at +36.0 ms on a ~2.7 s prefill, 95% CI [−13.9, +85.9] — a point estimate of +1.3% with an upper bound of +3.2%, against a harness resolution of ±1.9%. **Requires fp16 KV — see `DKV_SHARED_BASIS_FRAC` for the q4_0 trap.** Not a preset default, but it is no longer UNVALIDATED: it has now been run through all three accuracy harnesses at `frac=0.50` and none of them can resolve a regression — needle suite 4/4 (unchanged), multifact multi-needle 3/3 and **relational 4/4** (the binding test, unchanged), first-token KL 0.00024 (identical) and decode-step KL 0.00125 → 0.00146. Synthesis moves 13.3 → 10.0, inside the ±15-point RSVD-seed band this metric carries and far below its ≥30 bar, which this model does not clear with the feature off either. The accuracy instrument (`colab/logit_fidelity.py`) shows no measurable harm. It was previously reported as unable to resolve a small change "because the DKV baseline already sits far from dense (KL 10.58, dense's top-1 at rank 1255)" — **that number was an instrument defect, not a property of DKV**. The harness captured the LAST `lm_head` call, which at `max_new_tokens=1` is a decode forward, and scored it against a control captured one token earlier; it was comparing token *N+2* against token *N+1*. Re-measured against a true dense control (plain `transformers`, DKV never loaded) at the same position, the DKV baseline reads **KL 2.4e-4 with 5/5 top-1 agreement and 896 compressed blocks engaged**. The harness now carries a `dense_true/dec` self-check row that contains no compression at all and must read 0. Also requires exact-form residuals (the default); under `DKV_RESIDUAL_EXACT_KEYS=0` the pool **refuses** and says so, since correction-form residuals are defined against a block's own reconstruction. **Requires an UNROTATED pool, and this is the sharpest trap in the feature.** Sharing compares *subspaces*, and RoPE rotates every key by its absolute position, so blocks holding the same text at different offsets have subspaces rotated apart. Measured on MLX, same document and block size, only `DKV_ROTATED_POOL` differing: best-partner retained energy **0.486 → 0.972**, voluntary joins **10 → 520**, forced **186 → 0**. CUDA is exposed only on `low`, its one rotated preset — every preset it enables sharing on (`mid`/`high`/`ultra`) is already unrotated, which is why CUDA never saw this. **The pool now REFUSES the rotated combination** at construction and names the setting to change; `DKV_SHARED_BASIS_ALLOW_ROTATED=1` keeps the bad configuration measurable. Refusing rather than warning is deliberate: the failure produces no error and no shape change. As with the q4_0 trap, **pool MB is identical either way**, so a rotated run reports the full memory win with fidelity quietly bought by forced joins. **On MLX the V store halves exactly as designed, but the saving is 1-3% of PEAK** — measured in one process: pool 65.2 → 58.3 MB at 8k and 221.7 → 198.3 MB at 32k, against a peak of 5.7 / 6.6 GB dominated by weights and prefill activations. Peak is unchanged (7.12 GB across six runs). Recall is clean — `mlx_needle_parity.py --long` is **9/9 with sharing on, including `32k@0.9`** — but decode is slower in 3/3 paired rounds at a **0.5-0.65x** ratio (absolute tok/s here cannot be resolved below ~20%, so only the paired ratio is quoted). **Opt-in, and on this model there is no operating point where it currently pays.** |
| `DKV_SHARED_BASIS_FRAC` | `0.50` | CUDA; MLX on branch | Basis rows allocated, as a fraction of block slots — a **capacity contract**, not a prediction. When the store fills, blocks force-join their nearest group, so a document that shares worse than assumed degrades in *fidelity*, never in correctness. Deeper settings are a different trade and should be argued for as one: retained energy `0.50 → 0.969`, `0.25 → 0.905`, `0.125 → 0.759`. **Do not pair with 4-bit KV.** On `low` (`kv_quant=q4_0`) quantisation noise destroys the subspace agreement entirely — **zero** voluntary joins, 294 forced, retained energy **0.685** — while pool MB is *identical*, because the saving comes from allocating fewer basis rows and not from grouping succeeding. That makes the failure invisible in memory numbers; the pool warns at construction, and `pool.basis_stats()['joined'] == 0` is the signature. |
| `DKV_POOL_ATTENDED_ONLY` | `1` | MLX | Size each session's pool by the layers that actually own a KV cache. On hybrid models (Qwen3.5: **6 of 24** layers) the rest can never hold a block, and allocating for them cost 406.9 MB of a 542.5 MB pool at 11.4k. Set `0` to restore the old full-width allocation. No-op on dense-attention models. |
| `--micro-block-size` / `block_size` | **`1024`** (MLX) | MLX | Tokens per compressed block. Raised from 256 after measuring all four metrics at once on Qwen3.5-2B-4bit: linkbench 9/24 → **24/24 (= dense)**, needles 6/6 either way, session pool 135.6 → **60.0 MB**, and the pool against the dense KV it replaces 0.95× → **0.28× (3.61× smaller)**. Retrieval tracks the **number of blocks** the context is split into, not fidelity or routing — at 16k, 256 gives ~58 blocks and 1024 gives ~15. At 256 the fixed 128-token residual budget stored *half of every block verbatim*, which is why the old default barely compressed. Use `512` for synthesis-shaped work. |
| `DKV_DETERMINISTIC` | `0` | CUDA | `1` forces a deterministic SDPA backend. Greedy decode is **not** reproducible at long context without it — the cause is the decode attention's reduction, not compression. **Turn this on for any run you intend to compare.** |
| `DKV_FAST_DECODE` | **`1`** | CUDA | Routed CUDA-graph decode, gated per session to whether a graph will actually be captured. Set `0` to disable. Its output is not bit-identical to eager — the first difference is **one ULP at step 1** and the greedy argmax flips around step 32 — because a CUDA graph makes no bit-identity promise. Accuracy is unaffected (digit-table 24/24, linkbench 23/48, both equal to eager and to dense). Use `DKV_FAST_DECODE=0` with `DKV_DETERMINISTIC=1` for runs you intend to compare token-for-token. |
| `DKV_DECODE_CACHE_INTERVAL` | **`4`** (MLX) | MLX | Tokens between re-routing and re-materialising the decode cache. Lowered from 16: the routed block set is FROZEN for the interval, so a needle whose block is routed late stays invisible that long. The old default's stated 15% speed justification does **not** reproduce — a paired, A/A-calibrated measurement puts 16 vs 4 at +0.095 ms/token, 95% CI [-0.486, +0.677], i.e. not resolvable. 4 buys a 4× shorter staleness window for no measurable cost. |
| `DKV_SVD_SEED` | `1234` | MLX | SVD random state seed. **Vary it for any accuracy A/B** — temperature-0 replication is deterministic and proves nothing, and at a fixed config this seed alone spans ~30 synthesis points. |
| `DKV_EARLY_LAYER_RANK_BOOST` | `0` | MLX | Set `1` to boost SVD rank ($2\times$) in early layers ($\le 15\%$ network depth) for syntactic protection. |
| `DKV_MAX_RANK_EARLY` | `0` | MLX | Cap for early layer boosted rank ($0$ = auto-selects $2\times$ base rank). |
| `DKV_PROFILE_CB` | `0` | Both engines | Enable layer-wise routing, GPU kernel, and readback latency profiling logs. |
| `DKV_CB_GQA_ROUTE` | `on` | Native | Grouped-Query Attention (GQA) head-averaging in native routing loop ($8\times$ latency reduction). |
| `DKV_CB_ROUTE_ALL` | `on` | Native | Forces routing across all resident blocks to prevent candidate screening drops. |
| `DKV_FUSED_DECODE` | `0` | MLX | Experimental Metal decode kernel ($0$ = disabled; keep disabled for benchmark accuracy). |

---

## 🏗️ Dual-Engine Architecture

| Feature | `ACTIVE_RUNTIME/` (Python Overlay) | `dkv_native/` (C++ Engine) |
|---|---|---|
| **Language** | Python 3.10+ (with optional C++ `dkv_core` extension) | Pure C++17 |
| **Backends** | MLX (Apple Silicon) / PyTorch + Triton (CUDA) | forked `llama.cpp` / `ggml` (Metal & CUDA) |
| **Model Format** | HuggingFace Transformers / `mlx-community` | GGUF |
| **Primary Target** | Research, rapid iteration, serving gateway | Production edge deployment, minimal host overhead |
| **Status** | Reference accuracy (4k–64k NIAH 100% exact recall — **on TILED filler only; see below**) | Experimental / Work-In-Progress |

---

## 📊 Current Results (CUDA, 2026-11)

One RTX 4070 SUPER (12.9 GB), Windows 11, `transformers` 5.14.1, `torch` 2.11,
nf4 weights, 32k-token RULER prompts. Every row is paired against dense and
tiled KIVI-4 run in the same harness. Full tables, logs and the commands that
produce them: [`docs/IMPROVEMENTS_LOG.md`](docs/IMPROVEMENTS_LOG.md).

**Decode fidelity** (`benchmarks/decode_fidelity.py`): dense's own answer is
forced through each arm and the per-step KL to dense is averaged (lower is
better). 4 prompts, 32 decode tokens each.

| model | KIVI-4 (tiled) | DKV codes (default, routed) | DKV codes, every block read | DKV low rank | decode tok/s, KIVI-4 / DKV |
|---|---|---|---|---|---|
| SmolLM3-3B | 0.0077 | **0.0055** | 0.0041 | 0.128 | 3.6 / **8.0** |
| Phi-4-mini | 0.0223 | **0.0077** | 0.0052 | 0.211 | 3.8 / **9.8** |
| Qwen3.5-2B | 0.0071 | **0.0016** | 0.0022 | 0.0042 | 11.9 / **16.3** |
| Ministral-3-8B | **0.0013** | 0.0020 | 0.0020 | 0.324 | 3.7 / 1.9 (spills) |
| Qwen2.5-7B | **0.0155** | 0.060 | 0.047 | 0.789 | 4.8 / **10.2** |
| granite-4.2-8b | 0.0066 | 0.075 | **0.0013** | 0.372 | 3.1 / 1.9 (spills) |

Qwen2.5-7B and granite were rerun after the 2026-11-26 value-store fix and the
2026-11-27 router changes (heads vote; adaptive growth at any block count;
before them 0.101 and 0.111); the other rows predate both. Reading it:

* With every block read, the default store is below KIVI-4 on 4 of these 6
  models and tied on Ministral (0.0020 vs 0.0013; the routed arm's paired 95%
  CI spans zero). Where DKV loses, routing is most of the loss
  (granite: 0.075 routed, 0.0013 reading everything).
* Qwen2.5-7B's remaining gap is the model's numerics, not the store: a lossless
  codec through the same pipeline scores 0.042, and the model's own dense eager
  and dense SDPA prefills differ more than DKV and dense do.
* DKV decodes 1.4-2.7x faster than tiled KIVI-4 in this harness (both without
  CUDA graphs) wherever it fits the card. Its
  cost is memory: exact prefill holds the prompt in 16 bits, so the two 8B models
  spill past 12.9 GB at 32k (`DKV_STREAMING_COMPRESS=1` fits them).
* The gemma-4 models are left out: their DKV arms are not yet comparable.

**RULER 32k** (`benchmarks/run_ruler_cuda.py`, official generators; one item
per task, so one item moves a score by 6-25 points):

| model | dense | KIVI-4 (tiled) | DKV codes (default) | DKV low rank |
|---|---|---|---|---|
| SmolLM3-3B (13 tasks) | 84.6 | 86.9 | 78.5 | 64.7 |
| Qwen2.5-7B | 95 | 100 | 82.5 (63.8 before the 2026-11-27 router changes; 88.8 reading every block) | 63.8 |
| Phi-4-mini | 70 | 70 | 70 | 68.8 |
| Qwen3.5-2B | 100 | 100 | 100 | 100 |
| gemma-4-e2b | 88.8 | 88.8 | 88.8 | 88.8 |
| gemma-4-e4b | 100 | 100 | 100 | 100 |
| gemma-4-12B | 100 | 93.8 | 93.8 | 75 |
| Ministral-3-8B | 100 | 100 | 100 | 75 |

**vLLM** ([`integrations/vllm`](integrations/vllm/), vLLM 0.34.0, Qwen2.5-1.5B,
CUDA graphs on every arm): DKV's `k8v4` codec pages hold **2.01x** stock's KV
tokens at 32k and decode at 107.8 tok/s (stock 102.4), needle 6/6, and track
stock's greedy output more closely than vLLM's own TurboQuant k8v4. Routed decode
over pages did not pay at 16k-32k there and stays opt-in. One model, one GPU.

**What DKV does not do yet:** routing costs quality on some models (above); the
low-rank store is well below KIVI-4 on quality; MLX and the C++ engine trail the
CUDA runtime.

## 📊 Earlier Measurements (MLX and the low-rank store)

Results in this section predate the codes default and, unless labelled CUDA,
were measured on the MLX runtime: a single **Apple M3 with 8.6 GB unified memory**, evaluating **Qwen2.5-1.5B-Instruct (int4)** using rank $r=32$, residual budget $R=128$, micro-block size $B_s=256$, top-$K=16$, and residual-key router (raw logs preserved in `benchmarks/results/results_latest.json`). They are kept as a record, not as current claims.

### 0. CUDA Runtime — Preset Accuracy and the Cost of `ultra`

Measured on the low-rank store, when `mid` still stored rotated keys (it no
longer does). Separate host and runtime from the MLX numbers below: **RTX-class CUDA GPU,
Qwen3.5-2B, 32k context**, `transformers` 5.14.1 / `torch` 2.11.0+cu130. **A dense
control was run in the same process generation as every DKV number** — absolute
scores here are not comparable to scores taken in another environment, and this
project has been burned by exactly that (see the note under the accuracy table).

**Accuracy — `ultra` reaches dense exactly on both metrics that can resolve anything**

| benchmark | dense | `ultra` | `mid` |
|---|---|---|---|
| linkbench (multi-hop retrieval), 48 seeds | 23/48 | **23/48** | 21/48 |
| tablebench (exact 4-digit recall), 24 seeds | 24/24 | **24/24** | 14/24 |
| needle sweep, 9 depths × 2 models | — | 9/9 | 9/9 |

> **Do not compare these absolutes against older numbers in this repo.** An
> environment change (most likely a `transformers`/`torch` update) roughly halved
> *every* linkbench arm — **including dense, which shares no DKV code**. The
> *relationships* are what hold, and they reproduce: `ultra == dense`, `mid` below
> both. Always run a dense arm alongside, in the same environment.

**What closes the gap — and what does not**

The whole `mid`→`ultra` difference is the un-rotated pool. Everything else was
measured and is inert on `tablebench` (all 14/24, unchanged from `mid`):

| lever tried | result |
|---|---|
| residual budget $R$ = 40 vs 128 | no change (also no change on 3 prose benchmarks) |
| attend **every** block (`DKV_TOPK_BLOCKS=0`) | no change → not a routing problem |
| tier residuals on error tail, not median | no change |
| **un-rotated pool (`ultra`)** | **14/24 → 24/24, i.e. dense parity** |

**Speed — what the un-rotated pool costs**, paired and in-process, 8 rounds, 32k:

| model | attended layers | `mid` | `ultra` | cost |
|---|---|---|---|---|
| Qwen3.5-2B | 6 of 24 | 38.63 ms/tok | 43.02 ms/tok | **+11.5%** |
| Qwen2.5-1.5B | 28 of 28 | 51.28 ms/tok | 65.10 ms/tok | **+26.5%** |

**`--fastdc` is now the default.** Paired, in-process, Qwen2.5-1.5B, 6 rounds:

| context | effect | 95% CI |
|---|---|---|
| 4k | **25.4% faster** | [−11.89, −10.55] ms |
| 8k | **9.3% faster** | [−4.46, −3.87] ms |
| 16k | 5.2% slower | [+2.51, +3.09] ms |
| 32k | no effect | [−4.03, +4.29] ms |

It was opt-in for a long time on two objections, both now answered. It *was* 2×
slower at 4k — that was three chained bugs making it fall back to a slower kernel
on every step, since fixed. And it does not reproduce eager token-for-token: the
first difference is **one ULP at step 1**, and the greedy argmax flips around step
32. A CUDA graph makes no bit-identity promise, so that is a *distribution*
difference of the same kind `DKV_DETERMINISTIC` already exists to manage — not a
quality one. Accuracy is unchanged: digit-table 24/24 and linkbench 23/48 with it
on, both identical to off. (An earlier sweep that found it 2x slower at 4k
measured the three fallback bugs above, since fixed.)

The un-rotated pool's cost is the rotation applied at read on every attended layer, so it scales
with attended-layer count — **a figure for one model tells you nothing about
another.** Since these measurements `mid`, `high` and `ultra` all store keys
un-rotated (the preset ladder above); `low` keeps the rotated pool for the speed.

### 1. Context Length Sweep (DKV vs. Dense Baselines)

| Context Length | Runtime / Engine | Prefill Time (s) | Decode Speed (tok/s) | Peak Allocator Memory (GB) | Needle Recalled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4k** | **DKV Active (Compressed)** | 6.16s | 33.3 | 1.55 GB | **Yes** |
| | MLX Dense Baseline | 5.00s | 68.9 | 1.68 GB | **Yes** |
| | PyTorch Dense Baseline | 0.83s | 3.6 | 8.05 GB | **Yes** |
| **8k** | **DKV Active (Compressed)** | 13.37s | 31.4 | 1.67 GB | **Yes** |
| | MLX Dense Baseline | 11.08s | 54.5 | 1.79 GB | **Yes** |
| | PyTorch Dense Baseline | 0.48s | 3.5 | 8.05 GB | **Yes** |
| **16k** | **DKV Active (Compressed)** | 29.77s | 29.7 | 2.20 GB | **Yes** |
| | MLX Dense Baseline | 27.04s | 48.2 | 2.03 GB | **Yes** |
| | PyTorch Dense Baseline | *OOM* | *OOM* | *OOM* (8.58 GB) | *OOM* |
| **32k** | **DKV Active (Compressed)** | 72.33s | 27.2 | 3.12 GB | **Yes** |
| | MLX Dense Baseline | 75.00s | 37.5 | 2.45 GB | **Yes** |
| | PyTorch Dense Baseline | *OOM* | *OOM* | *OOM* (8.58 GB) | *OOM* |
| **64k** | **DKV Active (Compressed)** | **477.12s** | **21.4** | 5.04 GB | **Yes** |
| | MLX Dense Baseline | **820.98s** | **20.2** | 3.23 GB | **Yes** |
| | PyTorch Dense Baseline | *OOM* | *OOM* | *OOM* (8.58 GB) | *OOM* |

> 💡 **Key Benchmark Takeaways:**
> - **Prefill Crossover at 64k:** At extreme context ($64\text{k}$), DKV prefilling beats MLX Dense prefilling by **$1.72\times$** (477.1s vs 821.0s), demonstrating sub-quadratic scaling ($O(L \cdot K)$).
> - **Baseline Differentiation:** Standard PyTorch Dense (`AutoModelForCausalLM`) suffers an Out-Of-Memory (OOM) failure at $\ge 16\text{k}$ context on an 8.6 GB memory host, whereas MLX unified memory optimization enables MLX Dense to complete 64k.
> - **Needle Recall & Heuristic-Protection Transparency:**
>   - **Pattern-Protected NIAH Needle Recall (Digits/Codes):** 100% exact recall (protected by `streaming_sparse_ingest.py` content-aware regex safeguards).
>   - **Unassisted Prose Entity Recall (`benchmarks/prose_fact_recall.py`):** ~60% recall under default unassisted SVD compression. Toggle regex rules via `DKV_DISABLE_REGEX_HEURISTICS=1`.

### 2. Residual Budget ($R$) Trade-Off Sweep (16k Context)

The residual budget $R$ acts as an explicit memory-speed-accuracy dial:

| Residuals ($R$) | Passcode Recall | Decode Speed (tok/s) | KV Cache Store Size (GB) | Compression Ratio vs. Dense |
| :---: | :---: | :---: | :---: | :---: |
| **8** | **Yes** | 21.4 | 0.124 GB | $3.80\times$ |
| **16** | **Yes** | 21.1 | 0.139 GB | $3.41\times$ |
| **32** | **Yes** | 20.5 | 0.167 GB | $2.83\times$ |
| **64** | **Yes** | 21.5 | 0.224 GB | $2.11\times$ |
| **128 (Default)** | **Yes** | 19.6 | 0.338 GB | $1.40\times$ |

### 3. Per-Block Memory Breakdown ($B_s = 256$ Tokens)

| Component | Dimensions / Shape | Bytes | Note |
| :--- | :--- | :--- | :--- |
| **$U$ coefficients** | $[255, 16]$ | 16,320 B | Low-rank core |
| **$V_K, V_V$ bases** | $[2, 16, 128]$ | 32,768 B | Low-rank core |
| **Anchors $a_k, a_v$** | $[2, 128]$ | 1,024 B | Exact block reference |
| **Key Min/Max** | $[2, 128]$ | 1,024 B | Decode router bounds |
| **Scalars / Metadata** | scale, seq_len | 8 B | Per-block control |
| **Low-Rank Core Total** | | **51,144 B** | **49.9 KiB (Fixed)** |
| Exact residuals ($R=128$) | $[128, 2, 128]$ | 131,072 B | 128.0 KiB |
| Exact residuals ($R=64$) | $[64, 2, 128]$ | 65,536 B | 64.0 KiB |
| **DKV Block ($R=128$)** | | **182,216 B** | **177.9 KiB ($1.44\times$ compression)** |
| **DKV Block ($R=64$)** | | **116,680 B** | **113.9 KiB ($2.25\times$ compression)** |
| **Dense Block** | $[256, 2, 128] \times 2$ | **262,144 B** | **256.0 KiB ($1.00\times$)** |

**$V_K, V_V$ is the largest item after the residual store**, and the one blocks
most nearly agree on — which is what `DKV_SHARED_BASIS` (opt-in) exploits.
At `frac=0.50` the 32,768 B row is amortised across two slots, taking the
low-rank core from 51,144 B to **34,760 B (−32%)** and the whole $R=128$ block
to 165,832 B. Measured end to end on Qwen2.5-1.5B @8k the pool goes
**91.4 → 69.8 MB (−23.6%)** at retained delta energy 0.969 with zero forced
joins. The saving is spent on holding *more blocks* in the same budget, not on
using less memory for the same context.

### 4. THUDM LongBench & NVIDIA RULER Benchmark Suites (32k Context)

Evaluated on **Qwen2.5-1.5B-Instruct (int4)** under up to $32,768$ context length using DKV active runtime ($r=32$, $R=128$, $B_s=256$). These predate the current RULER protocol (official generators, chat template, answer prefix, dense and KIVI-4 controls; 2026-10-20 onward) and have no dense control, so they are not comparable with the current RULER table above:

#### NVIDIA RULER Benchmark Suite
| Task Category | Task | Evaluated Metric | DKV Score |
| :--- | :--- | :--- | :---: |
| **Multi-Document QA** | `qa_2` | Exact Match / Substring | **100.0%** |
| **Single-Needle Recall** | `niah_single` | Needle Retrieval | **100.0%** |
| **Multi-Key Recall** | `niah_multikey` | Key-Value Retrieval | **100.0%** |
| **Multi-Value Recall** | `niah_multivalue` | Value Association | **100.0%** |
| **Variable Tracking** | `variable_tracking` | Chain-of-Thought Variable | **100.0%** |
| **Common Words Extraction** | `cwe` | Frequency Extraction | **100.0%** |

#### THUDM LongBench Comprehensive Suite
| Category | Task | Metric | DKV Score |
| :--- | :--- | :--- | :---: |
| **Single-Doc QA** | `qasper` | F1 Score | **87.90%** |
| | `multifieldqa-en` | F1 Score | **62.61%** |
| | `narrativeqa` | F1 Score | **24.56%** |
| **Multi-Doc QA** | `2wikimultihopqa` | F1 Score | **9.68%** |
| | `hotpotqa` | F1 Score | **5.11%** |
| | `musique` | F1 Score | **4.97%** |
| **Summarization** | `multi_news` | ROUGE-L | **26.48** |
| | `gov_report` | ROUGE-L | **20.38** |
| | `qmsum` | ROUGE-L | **7.79** |
| **Code Completion** | `repobench-p` | Code Sim | **30.00%** |
| | `lcc` | Code Sim | **24.33%** |
| **Synthetic & Few-Shot** | `passage_count` | Accuracy | **65.05%** |
| | `samsum` | ROUGE-L | **58.99** |
| | `passage_retrieval-en` | Rank Accuracy | **1.58%** |
| | `trec` | Accuracy | **2.56%** |
| | `triviaqa` | F1 Score | **0.13%** |

> 💡 **Routing & Pruning Insight (`DKV_TOPK_BLOCKS=16`):**
> - **High-Efficiency Pruning Trade-off:** The default benchmark setting (`DKV_TOPK_BLOCKS=16`) routes only 16 micro-blocks ($16 \times 256 = 4,096$ tokens) per decode step out of the full 32,768 ($32\text{k}$) context window, pruning **87.5%** of sequence blocks to achieve $>30$ tok/s decode speed on unified memory.
> - **Multi-Hop Retrieval Scaling:** For complex multi-document reasoning tasks (such as `2wikimultihopqa`, `hotpotqa`, and `musique`) where evidence is scattered across multiple distant passages, increasing the candidate budget to `DKV_TOPK_BLOCKS=32` or `64` ($8\text{k}$–$16\text{k}$ attended tokens) expands evidence gathering and further boosts multi-hop recall.

---

## ⚡ Advanced Systems Features & Safeguards

### 1. Algorithmic Rank-Boosting
- **Early-Layer Boosting:** Boosts SVD rank by up to $2\times$ in the first 15% of network layers to safeguard syntactic representations (`DKV_EARLY_LAYER_RANK_BOOST=1`).
- **Content-Aware Dynamic Boosting:** Automatically detects micro-blocks containing numerical data, mathematical formulas (e.g., LaTeX `$$`, `\sum`), or formal definition patterns, boosting block SVD rank by $1.5\times$ on-the-fly.

### 2. Multi-Signal Residual Selection
To ensure critical factual information is retained, residuals are selected using three combined IDF-weighted priority signals:
- **Owner-Capture:** Entity names accompanying high reconstruction error tokens.
- **Edge-Capture:** Relational connectives with potential low-rank key collision.
- **Coverage Bonus:** Enforces uniform spread across block token positions to prevent localized error clustering.

### 3. Shared Low-Rank Bases (opt-in — `DKV_SHARED_BASIS=1`)

*CUDA on `main`; MLX on branch `mlx-shared-basis`. Both require an
UNROTATED pool — see the knob table.*
Blocks whose delta subspaces agree read **one** basis row instead of each storing
its own $V$. For a group basis $V_g$ with orthonormal rows, the best
approximation of $UV$ inside $\mathrm{span}(V_g)$ is $U' = U(V V_g^\top)$ — a
single matmul, no re-decomposition and no touching the original K/V. Blocks join
on **retained energy**, $\mathrm{tr}(GCC^\top)/\mathrm{tr}(GVV^\top)$ with
$G=U^\top U$, $C=VV_g^\top$, rather than on principal angles, because angles
weight every basis direction equally while a block's energy sits in the first
few. Both traces are $[k,k]$, so scoring never touches the token dimension.
Measured: **−23.6% pool** at retained energy 0.969 with zero forced joins.
**fp16 KV only** — see the `DKV_SHARED_BASIS_FRAC` knob for why 4-bit KV breaks
it silently.

### 4. Tiered Offloading & Asynchronous Prefetch (kTransformers-Inspired)
- **Tiered CPU-GPU KV Offloading:** Maintains a heat score for each micro-block, evicting cold blocks to pinned host RAM when GPU pool utilization exceeds 80%.
- **Step-Ahead Async Prefetch:** Background prefetching retrieves cold blocks into GPU memory before the subsequent decode step touch point, hiding PCIe transfer latency.

---

## 🧪 Testing & Verification

Execute test suites from repository root using `dkv_venv`:

```bash
# 1. Kernel Parity Oracle Test (MLX)
pytest ACTIVE_RUNTIME/tests/test_dkv_kernel_parity.py -q

# 2. Needle-in-a-Haystack Recall Benchmark (Canonical)
cd benchmarks && python niah_recall.py --bench --ctx 4096 8192 16384 32768 \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit

# 3. Multi-Entity Relational Binding Test
cd benchmarks && python relational_ab.py --mode sparse --natural --spread

# 4. Native C++ Engine Honest Sweep
cd dkv_native/tests && ./test_niah_native.sh

# 5. Unrotated-pool round-trip (DKV_ROTATED_POOL=0)
#    Checks the invariant, not the benchmark: a residual key read back out of the
#    pool and rotated to its absolute position must reproduce the rotated key that
#    lived there. RoPE is PARTIAL on Qwen3.5 (64 of 256 dims), so a wrong position
#    perturbs only ~25% of each key -- it degrades retrieval while the needle sweep
#    still passes. Do not judge a position fix on recall alone.
pytest ACTIVE_RUNTIME/tests/test_unrotated_pool.py -q

# 6. Greedy-sampler NaN guard
pytest ACTIVE_RUNTIME/tests/test_sampler_nan_guard.py -q

# 7. Distractor retrieval (what DKV_ROTATED_POOL is judged on; the needle sweep
#    has no confusable distractors and cannot see it). Always run a dense control
#    in the SAME configuration, and record the question mode next to the score.
SEEDS=$(python3 -c "print(','.join(str(i) for i in range(1,25)))")
CTX=16000 QMODE=direct SEEDS=$SEEDS python colab/linkbench_mlx.py
CTX=16000 QMODE=direct SEEDS=$SEEDS ENGINE=dense python colab/linkbench_mlx.py

# 8. Synthesis with a confidence interval (paired, replicated, MLX arms)
python colab/synthesis_power.py --arm dkv   --reps 6 --out dkv.json
python colab/synthesis_power.py --arm dense --reps 6 --out dense.json
python colab/synthesis_power.py --compare dkv.json dense.json
```

> **Before running any A/B on either runtime:** temperature-0 replication is
> deterministic and therefore proves nothing — a number "reproduced twice" is one
> sample, not two. The randomised SVD's seed (`DKV_SVD_SEED` on MLX) is the axis
> that has to move; at a fixed config it alone spans ~30 synthesis points. Treat
> any synthesis difference under ~15 points as no difference. See
> `ACTIVE_RUNTIME/docs/cuda_port_record.md` for the rest of the method rules.

---

## 📖 Citation & References

If you use Differential-KV in your research or project, please cite:

```bibtex
@misc{chimurkar_2026_21539110,
  author       = {Chimurkar, Om},
  title        = {DKV: Anchor + Low-Rank Differential KV-Cache
                   Compression for Scalable Long-Context Inference},
  month        = jul,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.21539110},
  url          = {https://doi.org/10.5281/zenodo.21539110},
}
```

Persistent DOI: https://doi.org/10.5281/zenodo.21539110

For build instructions and native compilation details, see **[BUILD.md](BUILD.md)**.

---

## ⚖️ License

Differential-KV is released under the **MIT License** — see [LICENSE](LICENSE).

Vendored third-party code keeps its own license and is **not** covered by the root LICENSE:

| Component | Path | License |
|---|---|---|
| `llama.cpp` / `ggml` | `dkv_native/third_party/llama.cpp/` | MIT — [LICENSE](dkv_native/third_party/llama.cpp/LICENSE) (© 2023–2026 The ggml authors) |
| `nlohmann/json` | `dkv_native/third_party/llama.cpp/licenses/` | MIT — [LICENSE-jsonhpp](dkv_native/third_party/llama.cpp/licenses/LICENSE-jsonhpp) |

Both are MIT and therefore compatible with this project's license; their copyright
notices are retained in place, as MIT requires.
