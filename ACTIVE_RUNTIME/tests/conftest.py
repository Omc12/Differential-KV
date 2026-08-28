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
    # NOTE: this is a NO-OP ON CUDA. The only reader of DKV_DECODE_CACHE is
    # mlx_dkv_wrapper.py:1772; the CUDA path gates the same feature on
    # DKV_DECODE_CACHE_CUDA (dkv_attention.py:145), which defaults to "0".
    # So the "~2x tps decode cache" that this line and
    # serving/decode_config.py both believe is enabled is actually OFF on CUDA,
    # in tests AND in production. Left set for MLX parity and to keep the two
    # default sets identical; flipping the CUDA flag on is a perf change that
    # needs measuring, not a rename.
    "DKV_DECODE_CACHE": "1",          # decompress-and-cache fast decode (MLX only)
    "DKV_SPARSE_BIAS": "auto",        # adaptive merge bias (synthesis- AND NIAH-safe)
    "DKV_V_SCALE": "1",               # V rebalanced before joint SVD (CUDA↔MLX parity)
    # PIN THE POOL BUDGET — this is an ISOLATION fix, not a tuning knob.
    # Without it the budget is derived from FREE VRAM at manager init
    # ("ceiling: 50% of N GB free VRAM"), so max_blocks depends on how much
    # memory the PREVIOUS test happened to leave allocated. Block sizing feeds
    # routing and eviction, which makes numeric results order-dependent: a test
    # passes alone and fails in the suite, or flips between suite runs, with no
    # code change anywhere. test_formatting_rendering was intermittent for
    # exactly this reason.
    # 2.0 GB comfortably covers the 0.5B and 1.5B models the suite uses (they
    # derive 1.0 and 1.9 GB respectively when VRAM is free) so pinning it
    # reproduces the uncontended sizing every time.
    "DKV_POOL_BUDGET_GB": "2.0",
}


# ── Keys we do NOT set, but MUST restore ─────────────────────────────────────
# native_core/config.py:814,819 EXPORT preset-derived values into the process
# environment with setdefault(), because pool_stores_rotated_k() and
# _svd_energy_target() read them at call time. setdefault means the FIRST
# DKVConfig built in the process wins for the whole process -- and
# MLXKVBlockManager.__init__ resolves `self.rotated_pool` from that same
# variable (mlx_dkv_wrapper.py:1783, default "1").
#
# So one test building a config on a preset whose rotated_pool is False (`high`,
# `mid` and `ultra` all are) silently flips the pool frame for every MLX manager
# constructed later in the same process. Measured: at test_residual_budget_clamp
# the environment holds DKV_ROTATED_POOL='0' and DKV_SVD_ENERGY='0.9999', a fresh
# manager reports rotated_pool=False, and its five tests fail on
# ingest_streaming's unrotated-key guard -- while all five PASS when the file is
# run alone. Same class as the DKV_POOL_BUDGET_GB pin above, and same reasoning:
# an ISOLATION fix, not a tuning knob.
_ISOLATED_ENV_KEYS = ("DKV_ROTATED_POOL", "DKV_SVD_ENERGY")


def pytest_collection_modifyitems(items):
    for item in items:
        if inspect.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.anyio)


@pytest.fixture(autouse=True)
def good_dkv_defaults():
    """Run every test with the production DKV levers on (sparse path forced),
    then restore the prior environment so tests that set their own values (e.g.
    test_vscale_parity toggling DKV_V_SCALE) are unaffected and nothing leaks."""
    saved = {k: os.environ.get(k)
             for k in (*_GOOD_DKV_TEST_DEFAULTS, *_ISOLATED_ENV_KEYS)}
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

