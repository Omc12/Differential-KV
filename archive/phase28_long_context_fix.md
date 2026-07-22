# Phase 28 — Long Context Fix

## Objective
Fix the unguarded import path for `RetrievalAwareSparsePrefill` which could crash any prompt with `q_len > 1024`.

## Action Taken
- Modified `ACTIVE_RUNTIME/runtime/dkv_attention.py`.
- Wrapped the import of `research.sparse_prefill_anchors.RetrievalAwareSparsePrefill` in a `try/except` block.
- If the import fails (which it often would, as the module might be missing or experimental), it catches the exception and cleanly falls back to `F.scaled_dot_product_attention`, passing `is_causal=(q_len > 1)`.

## Validation
- Fixed a buggy import path in `test_long_context.py` where `runtime.kv_runtime_manager` was changed to `native_core.kv_runtime_manager`.
- Ran `test_long_context.py` which passes a ~25K token prompt to the model.
- Verified that the fallback is triggered instead of a fatal `ModuleNotFoundError`, ensuring stable long-context serving.

**Status:** SUCCESS
