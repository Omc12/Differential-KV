# DKV in vLLM: integration design

Status, 2026-11-26: **phases 1-3 implemented** in `integrations/vllm/`
(usage and measurements: `integrations/vllm/README.md`). Phase 1 gate passed
(20/20 token-identical); phase 2 gate passed (2.01x KV capacity at 32k, needle
6/6, decode 107.8 vs stock 102.4 tok/s); phase 3 (routed decode) works but did
not pay at 16k-32k, so it is opt-in. Phase 4 (serving under load) is not done.

What the build changed from the design below: vLLM writes the cache one token
at a time, so the vLLM codec is PER TOKEN (8-bit keys + KIVI's 4-bit values,
`k8v4`; KIVI's per-channel 4-bit keys need 32-token groups); pages hold no
anchor and there is no 16-bit window yet; speed came from fused Triton store
/ decode kernels and CUDA-graph capture, not from routing.

## What carries over, and what does not

DKV today is a Hugging Face runtime: it patches each attention module and owns
the KV cache (`ACTIVE_RUNTIME/runtime/native_block_pool.py`). vLLM owns the KV
cache itself: a paged pool sized from a per-layer KV-cache spec, a block table
per sequence, continuous batching and CUDA graphs. So the integration is not a
port of the HF wrapper. It is a vLLM **attention backend** that keeps vLLM's
scheduler and pages, and brings three DKV pieces into it:

| DKV piece | Where it lives in vLLM |
|---|---|
| KV codec (`runtime/kv_codecs.py`) | encodes a page once it leaves the recent window; the page then occupies the codec's bytes, not 16-bit bytes |
| exact recent window | the newest pages of each sequence stay 16-bit, attended exactly |
| router + routed attend (`native_core/srl`, `remat_cache.attend_fast`) | at decode, picks the top pages per sequence and attends only those plus the window, merging by log-sum-exp |

The codec interface was built so that this mapping needs nothing new:
`encode`/`decode` are per block, block-first, with no runtime assumptions.

## Extension points (verify against the pinned version)

vLLM's own docs call these interfaces unversioned and changing (RFC
vllm-project/vllm#57566, still open). The plugin therefore pins one exact vLLM
release and its CI checks these names on upgrade.

- **Registration:** a `vllm.general_plugins` entry point whose function calls
  `vllm.v1.attention.backends.registry.register_backend` to add the backend as
  `AttentionBackendEnum.CUSTOM` (the `_Backend` route was removed in v0.13.0).
  Selected with the backend flag, or by a `vllm.platform_plugins` platform
  whose `get_attn_backend_cls` returns its qualified name.
- **Backend:** `AttentionBackend` (`supported_kv_cache_layouts`,
  `get_supported_kernel_block_sizes_for_layers`, `supports_head_size_v`,
  `customize_spec`), an `AttentionImpl` (forward, KV write), and an
  `AttentionMetadataBuilder` (`build`, `build_for_cudagraph_capture`,
  `update_block_table`).
- **Page size:** `customize_spec(spec)` adjusts the layer's KV-cache spec after
  the Attention layer builds it; this is how a page is given the codec's byte
  size. Registration of custom spec managers moved from `spec_manager_map`
  (v0.20) to `KVCacheSpecRegistry` (v0.28) per the RFC; unverified, so check
  against source.

Precedent: a vLLM 0.26 plugin quantizes pages with a Triton KIVI codec once
they leave the attention window and frees the 16-bit page, with the engine
unmodified (LoopKV, arXiv 2611.04137). That is phase 2 below.

## Design

**Two page kinds per attention layer.** A 16-bit page pool for each sequence's
window (the newest `W` tokens, W from DKV's existing formula), and a codec page
pool for everything older. A page moves from the first to the second when it
leaves the window: encoded once (`codec.encode`), the 16-bit page returned to
vLLM's free list. vLLM sizes both pools from the spec; the codec pool's page
bytes are `codec.slot_bytes()` scaled to the page length, plus one exact anchor
row per DKV block.

**Block = group of pages.** DKV's unit is a block of `B` tokens with one
anchor; vLLM's page is small (16 tokens by default). Pages are grouped into
blocks of B tokens (B a multiple of the page size); the codec and the router
work per block, vLLM's allocator per page. A partial block stays in the window.

**Prefill:** unchanged attention over the prompt (vLLM's own kernel). Pages
that leave the window are encoded at the end of each prefill chunk, so peak
memory follows the window, not the prompt.

**Decode:** for each sequence, score its codec blocks with DKV's router
(per-block key summaries kept next to the codes), take the top K, decode those
blocks, attend them plus the 16-bit window, merge by log-sum-exp. K grows with
the same mass estimate DKV uses today.

**Batching and graphs:** the routed set differs per sequence, so decode
gathers per-sequence block lists into one padded batch (the metadata builder
owns this). CUDA graphs need a fixed K per capture size; capture with K_max and
mask unused slots.

## Phases, each with its gate

| phase | delivers | gate (all on a small model, then one 8B) |
|---|---|---|
| 1. plumbing | `dkv_vllm` package; entry point; a CUSTOM backend that delegates to vLLM's own backend | token-identical to stock vLLM on 20 prompts |
| 2. codec pages | window + codec pages, encode on leaving the window, attend ALL codec blocks | KL vs stock within KIVI-4's at equal bytes; KV capacity (tokens per GB) about 3x stock |
| 3. routed decode | router + top-K + LSE merge | same quality as phase 2 (paired); decode tok/s above phase 2 and stock at 32k and 128k |
| 4. serving | continuous batching under load, CUDA graphs, prefix caching off then on | throughput vs stock at equal memory; no regression on RULER at 32k |

Phase 2 alone already helps users (about 3x more context per GPU at KIVI-level
quality); phase 3 is DKV's speed claim inside a real server.

## Requirements

- Linux or WSL2: vLLM does not run natively on Windows. This machine has WSL2
  with Ubuntu 24.04.
- An install of one pinned vLLM release in a fresh virtual environment inside
  WSL (several GB with its CUDA wheels): needs the owner's approval.
- The GPU free of other runs for phases 2-4 measurements.

## Out of scope

Prefill sparsity inside vLLM, disaggregated prefill/decode, and the low-rank
codec in vLLM (its fused kernels are HF-runtime code; the KIVI-layout codec and
any function codec carry over as is).
