# Known Limitations

## Current Hard Limits

- **Single GPU only** — distributed is not implemented.
- **Batch size 1 decode** — Triton kernel asserts `bsz==1` for decode.
- **Qwen2 architecture** — attention patch imports `Qwen2Attention` directly.
- **Fixed rank** — NativeBlockPool allocated with fixed rank=32; rank must match at runtime.
- **micro_block_size=16** — SVD overhead dominates if set below ~8.
- **CUDA graph disabled by default** — `StaticSparseDecodeGraph` exists but is not
  wired into the live path until benchmarks prove replay is stable.

## Known Issues

- `RESEARCH_PROTOTYPES/compression/adaptive.py` import in kv_runtime_manager.py
  will silently fall back to fixed rank=8 if the path is missing (safe, not critical).
- The prefill path still reconstructs dense KV via `get_kv()` for the attention
  compute over new tokens — O(N) reconstruction is unavoidable for correctness.
