"""Single source of truth for the DiffKV serving 'best decode config'.

These DIFFKV_* env defaults make DiffKV decode fast AND accurate. They are applied
via ``setdefault`` (an explicit env always wins) and MUST be set BEFORE the
wrapper/manager is constructed, because the manager reads them at init time.

Why this module exists
----------------------
The *wrapper-level* defaults in ``mlx_diffkv_wrapper.py`` differ on purpose:
``DIFFKV_COMPRESSED_DECODE`` defaults to ``"1"`` there so the test suite always
drives the sparse path. The *serving* entrypoints instead want the user-optimal
policy below (fast dense for short prompts, DiffKV sparse for long). This config
used to be copy-pasted into both ``cli.py`` and the OpenAI gateway, which (a) let
them drift and (b) made external audits misread the flags as "disabled by default"
because they only looked at the wrapper default. Keeping it here fixes both.

See ``PARKED_SYSTEMS.md`` §0 for the full context.
"""
import os

# Verified 2026-07-06 (Opus 4.8): NIAH 2/2 @8k+16k (~21 tps @16k); relational A/B 5/5.
BEST_DECODE_DEFAULTS = {
    "DIFFKV_COMPRESSED_DECODE": "auto",   # fast exact-dense for short prompts, DiffKV sparse when long
    "DIFFKV_COMPRESSED_MIN_CTX": "8192",  # engage DiffKV sparse at 8k+ tokens
    "DIFFKV_DECODE_CACHE": "1",           # decompress-and-cache fast decode (~2x tps when sparse; bit-exact)
    "DIFFKV_SPARSE_BIAS": "auto",         # adaptive merge bias (multi-fact synthesis-safe AND NIAH-safe)
}


def apply_best_decode_defaults(log=None):
    """setdefault the best serving decode config (an explicit env still wins).

    Call BEFORE constructing the DiffKV wrapper/manager. ``log`` is an optional
    callable (e.g. ``print``) used to echo the resolved config. Returns the
    resolved {name: value} dict.
    """
    for k, v in BEST_DECODE_DEFAULTS.items():
        if os.environ.get(k) is None:
            os.environ[k] = v
    resolved = {k: os.environ.get(k) for k in BEST_DECODE_DEFAULTS}
    if log is not None:
        log("DiffKV decode config: " + ", ".join(f"{k}={v}" for k, v in resolved.items()))
    return resolved
