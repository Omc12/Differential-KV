import os
import pytest
import inspect

# ── Good DKV defaults for the test suite ──────────────────────────────────
# Mirror the production config the CLI/serving gateway apply (serving/decode_config
# .py BEST_DECODE_DEFAULTS + cli.py CUDA↔MLX parity), so tests exercise the same
# fast+accurate path users get — with ONE deliberate change: COMPRESSED_DECODE is
# pinned to "1" (always-sparse) instead of the serving "auto" (which runs the fast
# DENSE path below 8k). DKV tests must measure DKV/sparse numbers, never a
# dense fallback. Applied via setdefault so any test that explicitly sets one of
# these still wins; a per-test fixture restores the environment afterward so there
# is no cross-test leak. Source of truth: serving/decode_config.py (kept in sync).
_GOOD_DKV_TEST_DEFAULTS = {
    "DKV_COMPRESSED_DECODE": "1",     # force sparse — never the dense fallback for DKV runs
    "DKV_COMPRESSED_MIN_CTX": "8192",  # (moot while COMPRESSED_DECODE=1; kept for parity)
    "DKV_DECODE_CACHE": "1",          # decompress-and-cache fast decode (~2x tps; bit-exact)
    "DKV_SPARSE_BIAS": "auto",        # adaptive merge bias (synthesis- AND NIAH-safe)
    "DKV_V_SCALE": "1",               # V rebalanced before joint SVD (CUDA↔MLX parity)
}


def pytest_collection_modifyitems(items):
    for item in items:
        if inspect.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.anyio)


@pytest.fixture(autouse=True)
def good_dkv_defaults():
    """Run every test with the production DKV levers on (sparse path forced),
    then restore the prior environment so tests that set their own values (e.g.
    test_vscale_parity toggling DKV_V_SCALE) are unaffected and nothing leaks."""
    saved = {k: os.environ.get(k) for k in _GOOD_DKV_TEST_DEFAULTS}
    for k, v in _GOOD_DKV_TEST_DEFAULTS.items():
        os.environ.setdefault(k, v)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture(autouse=True)
def cleanup_memory():
    yield
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()

