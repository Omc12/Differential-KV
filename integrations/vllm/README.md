# dkv-vllm: Differential-KV as a vLLM attention backend

Pinned to **vLLM 0.34.0** (vLLM's backend interfaces are unversioned:
vllm-project/vllm#57566). Linux or WSL2.

```bash
pip install vllm==0.34.0
pip install -e integrations/vllm            # registers the backend via vllm.general_plugins

# DKV codec pages: ~2x the KV cache per GPU
DKV_VLLM_CODEC=k8v4 vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend CUSTOM
```

or from Python: `LLM(model, attention_backend="CUSTOM")` with `DKV_VLLM_CODEC`
set in the environment.

| setting | effect |
|---|---|
| `DKV_VLLM_CODEC` unset | the backend delegates to vLLM's Triton backend (token-identical to stock) |
| `DKV_VLLM_CODEC=k8v4` | pages hold 8-bit keys + 4-bit values per token (224 B per head-token at head_dim 128 vs 512 at 16 bits); fused Triton store and decode kernels; CUDA-graph capturable decode |
| `DKV_VLLM_CODEC=<DKV codec>` | any per-token codec from `runtime/kv_codecs.py` (e.g. `examples.int8_codec:INT8`; needs `ACTIVE_RUNTIME` on `PYTHONPATH`), through the reference path |
| `DKV_VLLM_ROUTE_TOKENS=N` | routed decode: read sink + recent window + the N tokens' worth of pages with the highest key bound (default 0 = read every page) |
| `DKV_VLLM_WINDOW` | recent tokens always read when routing (1024) |

Not yet supported with codec pages: sliding-window, softcap, attention-sink
and ALiBi layers (refused at load, with the reason); prefill and mixed batches
run a reference PyTorch path (memory-bounded, not fused).

## Measured (RTX 4070 SUPER 12 GB, WSL2, Qwen2.5-1.5B-Instruct, CUDA graphs on every arm)

`tests/phase23_gate.py`: 20 short prompts (greedy agreement with stock), 6
passkey depths, KV capacity from vLLM's own startup line, decode-only tok/s
(128-token minus 1-token run).

| context | arm | greedy agreement (matching prefix / 48) | needle | KV tokens | decode tok/s |
|---|---|---|---|---|---|
| 16k | stock | - | 6/6 | 207k (1.00x) | 118.5 |
| 16k | vLLM TurboQuant k8v4 | 23.9 | 6/6 | 439k (2.12x) | 129.3 |
| 16k | **DKV k8v4** | **29.4** | 6/6 | 398k (1.92x) | 111.0 |
| 32k | stock | - | 6/6 | 207k (1.00x) | 102.4 |
| 32k | **DKV k8v4** | 29.4 | 6/6 | **417k (2.01x)** | **107.8** |
| 32k | DKV k8v4, routed 2048 | 29.4 | 5/6 | 417k (2.01x) | 99.9 (reads 11% of pages) |

Reading: codec pages give ~2x the KV per GPU at stock speed or better, and
track stock's greedy output more closely than vLLM's own TurboQuant at about
the same bytes. Routing does not pay at these lengths on this model (scoring
costs more than the pages it skips) and lost one needle, so it is opt-in.
One model, one GPU: a starting point, not a claim across models.

Tests: `tests/test_slot_codecs.py` (CPU), `tests/test_triton_decode.py`
(store kernel vs the reference encoder, decode and score kernels vs plain
references), `tests/phase1_identical.py` (delegating backend token-identical
to stock: 20/20), `tests/phase23_gate.py`, `tests/bench_decode_step.py`.
