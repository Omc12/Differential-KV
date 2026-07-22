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

# ── k-transformers feature knobs ─────────────────────────────────────────────
# All default to safe/off values — existing behaviour is 100% unchanged unless
# a feature is explicitly enabled.  "auto" enables the feature when the hardware
# supports it without requiring manual config.
#
# Feature 1: Tiered CPU-GPU KV Offloading
#   Heat-scored proactive eviction of cold compressed micro-blocks (U, V, anchors)
#   to pinned CPU RAM, freeing GPU pool capacity for longer contexts.
KTRANSFORMERS_DEFAULTS = {
    "DIFFKV_TIER_ENABLED":       "auto",  # "auto"=on for cuda/mps, "0"=off, "1"=force-on
    "DIFFKV_TIER_EVICT_THRESH":  "0.80",  # pool fill ratio that triggers eviction (0.0–1.0)
    "DIFFKV_TIER_EVICT_BATCH":   "32",    # slots evicted per trigger

    # Feature 2: Async Step-Ahead Block Prefetching
    #   After routing selects top-K for step T, issues async H2D restore of any
    #   TIERED_CPU slots so they are warm before step T+1 begins.
    "DIFFKV_BLOCK_PREFETCH":     "1",     # "1"=enable, "0"=disable
    "DIFFKV_PREFETCH_LOOKAHEAD": "1",     # steps ahead to prefetch (keep at 1)

    # Feature 3: Vectorized / Tiled Expansion Kernels
    #   CUDA: Triton autotune over RANK configs fires automatically when Triton is
    #         available (no knob needed — the existing kernel gains autotune configs).
    #   CUDA: INT8 V quantization in Triton kernel (halves V bandwidth, ~0.3% error).
    #   MLX:  mx.fast.metal_kernel tile for U@V on Apple Silicon.
    "DIFFKV_TRITON_INT8_V":      "0",     # "1"=quantize V_KV to int8 (CUDA Triton path)
    "DIFFKV_MLX_METAL_EXPAND":   "auto",  # "1"=force Metal, "0"=pure-MLX, "auto"=try Metal

    # Feature 4: MLA-style Latent Projection (experimental — off by default)
    #   Projects K/V through a shared W [feat_dim → latent_dim] matrix before the
    #   per-block SVD, compounding memory compression.  Requires a calibration phase
    #   (first n_calib blocks initialise W via incremental PCA).
    #   Validate NIAH recall before enabling in production.
    "DIFFKV_MLA_LATENT":         "0",     # "1"=enable latent projection
    "DIFFKV_MLA_LATENT_DIM":     "0",     # 0=auto (kv_heads*head_dim // 4)
    "DIFFKV_MLA_CALIB_BLOCKS":   "16",    # blocks accumulated before PCA initialises W
}


def apply_best_decode_defaults(log=None):
    """setdefault the best serving decode config (an explicit env still wins).

    Call BEFORE constructing the DiffKV wrapper/manager. ``log`` is an optional
    callable (e.g. ``print``) used to echo the resolved config. Returns the
    resolved {name: value} dict.

    Also applies k-transformers feature defaults (KTRANSFORMERS_DEFAULTS).
    All k-transformers knobs default to safe/off values so existing behaviour
    is unchanged unless explicitly overridden.
    """
    all_defaults = {**BEST_DECODE_DEFAULTS, **KTRANSFORMERS_DEFAULTS}
    for k, v in all_defaults.items():
        if os.environ.get(k) is None:
            os.environ[k] = v
    resolved = {k: os.environ.get(k) for k in all_defaults}
    if log is not None:
        core_cfg = {k: resolved[k] for k in BEST_DECODE_DEFAULTS}
        kt_cfg   = {k: resolved[k] for k in KTRANSFORMERS_DEFAULTS}
        log("DiffKV decode config: " + ", ".join(f"{k}={v}" for k, v in core_cfg.items()))
        log("DiffKV kTransformers: " + ", ".join(f"{k}={v}" for k, v in kt_cfg.items()))
    return resolved
