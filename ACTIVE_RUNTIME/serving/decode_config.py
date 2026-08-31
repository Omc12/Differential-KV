"""Single source of truth for the DKV serving 'best decode config'.

These DKV_* env defaults make DKV decode fast AND accurate. They are applied
via ``setdefault`` (an explicit env always wins) and MUST be set BEFORE the
wrapper/manager is constructed, because the manager reads them at init time.

Why this module exists
----------------------
The *wrapper-level* defaults in ``mlx_dkv_wrapper.py`` differ on purpose:
``DKV_COMPRESSED_DECODE`` defaults to ``"1"`` there so the test suite always
drives the sparse path. The *serving* entrypoints instead want the user-optimal
policy below (fast dense for short prompts, DKV sparse for long). This config
used to be copy-pasted into both ``cli.py`` and the OpenAI gateway, which (a) let
them drift and (b) made external audits misread the flags as "disabled by default"
because they only looked at the wrapper default. Keeping it here fixes both.

See ``PARKED_SYSTEMS.md`` §0 for the full context.
"""
import os

# Verified 2026-07-06 (Opus 4.8): NIAH 2/2 @8k+16k (~21 tps @16k); relational A/B 5/5.
BEST_DECODE_DEFAULTS = {
    "DKV_COMPRESSED_DECODE": "auto",   # fast exact-dense for short prompts, DKV sparse when long
    "DKV_COMPRESSED_MIN_CTX": "8192",  # engage DKV sparse at 8k+ tokens
    "DKV_DECODE_CACHE": "1",           # decompress-and-cache fast decode (~2x tps when sparse; bit-exact)
    "DKV_SPARSE_BIAS": "auto",         # adaptive merge bias (multi-fact synthesis-safe AND NIAH-safe)

    # ── CUDA graph decode ────────────────────────────────────────────────────
    # Capture the decode forward as a CUDA graph and replay it, which removes the
    # per-step kernel-launch and Python dispatch cost. Decode is ~39% GPU-idle
    # without it (27.19 ms/token of kernels against 44.7 ms/token of wall), so
    # this is launch overhead, not compute.
    #
    # Enabled for the BYPASS path only -- the capture site additionally requires
    # a dense StaticCache to exist, which is true exactly while DKV has not
    # engaged. Verified BIT-EXACT there: same prompt, graphs on vs off, identical
    # md5 of the generated text (0fec68e15bdab8c6), wall 3.7 -> 1.7 s.
    #
    # The ROUTED path is deliberately NOT enabled. Capture succeeds and replay
    # measures 1.41x (16.59 -> 23.38 tok/s wall at 16k), but the dense window of
    # recently generated tokens is assembled with a HOST write index, so replay
    # freezes it and the text diverges. See the note in hf_dkv_wrapper.
    #
    # DKV_GRAPH_SAFE_DECODE also removes two device->host syncs from the ROUTED
    # step (a torch.equal and an SRL key-trail .cpu()), which changes routed
    # behaviour slightly even with graphs off: the routed-set comparison becomes
    # conservative (assume changed) and the SRL recent-key trail is not
    # collected. Gated on the accuracy suite rather than assumed safe -- needle
    # sweep 9/9 with 9/9 determinism on Qwen3.5-2B and linkbench 20/24, both
    # unchanged.
    "DKV_DISABLE_CUDA_GRAPH": "0",     # 1 disables capture entirely
    "DKV_GRAPH_SAFE_DECODE": "1",      # sync-free decode step (bypass capture)
    # ── RESIDUAL FORMAT: int8. Measured, not inherited. ───────────────────────
    # THIS LINE, not the wrapper's constructor default, is what reaches serving:
    # apply_best_decode_defaults() writes it into os.environ (:172) and both the
    # CLI (cli.py:849, NOT device-gated) and the gateway (:938) call it. Changing
    # only the constructor default ships inert.
    #
    # ON CUDA it arrives one hop further along than it used to: since e38f3cd1 the
    # pool no longer reads the environment itself (that read had a "none" default
    # and ignored the config object). DKVConfig resolves DKV_RESIDUAL_QUANT and
    # KVRuntimeManager forwards it into NativeBlockPool.__init__. So this line
    # still reaches CUDA, but only for callers that run apply_best_decode_defaults;
    # everything else gets native_core/config.py's default, which is kept equal to
    # this one on purpose. Change the two together.
    #
    # WHY int8 AND NOT int4 — AND THE TWO BACKENDS DISAGREE ABOUT WHY.
    #
    # MLX, 4-needle verbatim NIAH, ctx=20000, 12 randomised trials (48 needles/arm),
    # Qwen2.5-1.5B-4bit, block 1024, greedy, DKV_DECODE_CACHE=1:
    #     int4   0/48 =  0.0%  95% CI [ 0.0,  7.4]   <- the old default
    #     int8  41/48 = 85.4%  95% CI [72.8, 92.8]   <- THIS
    #     fp16  42/48 = 87.5%  95% CI [75.3, 94.1]
    # int8 is statistically indistinguishable from fp16 and halves the residual
    # buffers (63.0 -> 33.5 MB at 16k; residuals are the LARGEST session category
    # there at 37.6%, vs dense window 33.4% and comp_U 14.1%).
    #
    # CUDA, measured independently on its own harness (12 trials x 16k/32k, 96
    # paired needles, Qwen2.5-1.5B-Instruct fp16 weights): int8 returns the
    # IDENTICAL per-needle result to fp16 on all 96, and int4 loses only 6/96
    # (sign test p=0.031) while keeping the correct entity name in 72-77 of 96
    # and never producing degenerate output. So int8 is right on both backends,
    # but for different reasons: on CUDA int4 is a fidelity trade with a small
    # measurable price; ON MLX int4 IS A BUG.
    #
    # THE MLX FAILURE IS CORRUPTION, NOT LOST RECALL. Its int4 output collapses
    # into ':\n\n:\n\n:' and token soup with zero entity names in 8/8 trials,
    # where CUDA's int4 returns near misses (Cobra-4367-5976 -> Cobra-4363-5976).
    # CUDA's residuals-OFF control settles the interpretation: 2/96 exact, name
    # gone 93/96, and still FLUENT — losing 100% of the residual signal does not
    # perturb generation, so a merely degraded signal cannot be what stops MLX
    # emitting language. Localisation (all measured, whole-word needles):
    #   * trigger is the PACKED WIDTH 4, not the int4 branch — format name "int8"
    #     with DKV_RESIDUAL_QUANT_BITS=4 reproduces the soup BYTE-IDENTICALLY.
    #     The same crossing on CUDA is also byte-identical to its int4 and stays
    #     fluent, so behaviour follows the width on BOTH backends; only MLX's
    #     width-4 corrupts.
    #   * onset > ~8192 compressed tokens PER LAYER, independent of block_size,
    #     block count and K (block 512 nb=15 ok/19 fail; block 1024 nb=7 ok/8
    #     fail; block 2048 nb=3 ok/5 degraded; DKV_TOPK_BLOCKS 8->16 does not
    #     move it). CUDA is clean at 14,336 and 30,720 compressed tokens PER
    #     LAYER (1.75x and 3.75x this onset, both read off a live pool), with a
    #     deficit FLAT in context, so this is not a shared design consequence.
    #   * NOT residual volume (res=8 corrupts too), NOT the compile boundary
    #     (forcing mx.eval on fetched residuals changes nothing), NOT weight
    #     precision (CUDA NF4 does not collapse int4), NOT needle construction.
    #   * values are provably fine in isolation: live _store_residuals ->
    #     _fetch_res_k matches synthetic mx.quantize at 1.00x.
    #   Width-keyed, values correct, sharp onset at a power-of-two token count =
    #   ADDRESSING BUG. Look for an offset/extent computed with the 8-bit stride.
    #
    # DEAD HYPOTHESES, killed by measurement — do not re-derive: residual
    # dilution via the RoPE frame split (CUDA remat confirmed ACTIVE), low-rank
    # base quality (CUDA bases floor at 2/96 with residuals off), rotation order
    # (real but small: +7/+8 needles to fp16/int8, +1 to int4), weight precision,
    # needle construction, lazy-dequantise at the compile boundary.
    #
    # DO NOT re-validate any of this with a SINGLE-NEEDLE sweep: that metric is
    # saturated and scored int4 8/9 against fp16 7/9 here, ranking the broken
    # format ABOVE the good one. And do not use partial-word needle names —
    # OMEGA splits ' O','ME','GA' and makes recall a coin flip (see
    # colab/multifact_eval_cuda.py:19-25). Use whole-word names.
    "DKV_RESIDUAL_QUANT": "int8",
}


# ── CONSTRUCTOR defaults (not environment variables) ─────────────────────────
# These reach the runtime through the wrapper's `config` dict rather than the
# environment, and they were the ones that actually drifted: `cli.py` and the
# OpenAI gateway each carried `--micro-block-size default=256` and passed it
# EXPLICITLY, so the runtime's measured default could never take effect through
# either of them. A benchmark that constructs MLXDKVWrapper directly sees the
# right value; the CLI and the server did not. That is invisible unless you
# diff four files, which is exactly what this module exists to prevent.
#
# Rule: entry points MUST NOT carry their own default for anything listed here.
# Argparse defaults to None and the value is forwarded only when the user
# actually passed one.
#
# The justification for each value lives next to its use in
# `mlx_dkv_wrapper.MLXDKVWrapper.__init__`, with the measurements attached; this
# dict is the single place that OWNS the number.
MLX_CONSTRUCTOR_DEFAULTS = {
    "block_size": 1024,
    "max_residual": 128,
    # WAS "int4" on the linkbench result below. That result is NOT retracted, but it
    # does not survive as a default: 4-needle NIAH at ctx=20000 (12 randomised trials,
    # 48 needles/arm, Qwen2.5-1.5B-4bit, block 1024, greedy) measures
    #     int4   0/48 =  0.0%  95% CI [ 0.0,  7.4]
    #     int8  41/48 = 85.4%  95% CI [72.8, 92.8]
    #     fp16  42/48 = 87.5%  95% CI [75.3, 94.1]
    # i.e. int4 residuals cost ALL long-context verbatim recall. The two findings are
    # compatible: linkbench is multi-hop RELATIONAL on Qwen3.5-2B, NIAH is verbatim
    # code recall, and verbatim is far more precision-sensitive. Residuals are the
    # exact-copy tokens (see the note at mlx_dkv_wrapper.MLXKVBlockManager.__init__).
    # CUDA runs int4 residuals too -- that is NOT why it scores well, and e38f3cd1
    # has since made its int8 real and propagated the config that selects it. If
    # int4 is wanted back, re-run linkbench AND a MULTI-needle NIAH under both.
    "residual_quant": "int8",   # measured equal to fp16 at 1.88x smaller; see
                                # DKV_RESIDUAL_QUANT above for the full ladder.
    "rank": 48,           # a CEILING, not a target; the rank sweep that suggested
                          # otherwise was randomised-SVD projection noise.
                          # 32 -> 48 on 2026-08-31: it was NOT actually a ceiling
                          # before (the per-layer middle band returned 1.5x, so
                          # 32 delivered 48). The schedule multipliers were
                          # divided by 1.5 and this scaled up by 1.5, so the
                          # delivered ranks are unchanged -- 24/48/16 either way
                          # -- and the word "CEILING" above is now true.
}


def resolved_runtime_config(wrapper=None) -> dict:
    """Everything that decides runtime behaviour, in one printable dict.

    Point of this: after a default changes, `assert resolved_runtime_config()[k]`
    is a one-line check that it actually reaches the process — rather than
    reading the wrapper, the CLI, the gateway and native's env plumbing and
    hoping they agree. Pass a constructed wrapper to include what it resolved to.
    """
    out = {f"env:{k}": os.environ.get(k) for k in
           list(BEST_DECODE_DEFAULTS) + list(KTRANSFORMERS_DEFAULTS)}
    out.update({f"default:{k}": v for k, v in MLX_CONSTRUCTOR_DEFAULTS.items()})
    for k in ("DKV_ROTATED_POOL", "DKV_STREAMING_COMPRESS", "DKV_POOL_ATTENDED_ONLY",
              "DKV_DECODE_CACHE_INTERVAL", "DKV_MAX_RESIDUAL", "DKV_TOPK_BLOCKS"):
        out[f"env:{k}"] = os.environ.get(k)
    if wrapper is not None:
        out["effective:block_size"] = getattr(wrapper, "block_size", None)
        out["effective:rank"] = getattr(wrapper, "base_rank", None)
        mgr = getattr(wrapper, "manager", None)
        if mgr is not None:
            out["effective:recency_window"] = mgr.recency_window
            out["effective:rotated_pool"] = mgr.rotated_pool
            out["effective:attended_layers"] = (
                len(mgr._attended_layers) if mgr._attended_layers is not None else "all")
            out["effective:decode_cache_interval"] = mgr._decode_cache_interval
    return out


# ── k-transformers feature knobs ─────────────────────────────────────────────
# All default to safe/off values — existing behaviour is 100% unchanged unless
# a feature is explicitly enabled.  "auto" enables the feature when the hardware
# supports it without requiring manual config.
#
# Feature 1: Tiered CPU-GPU KV Offloading
#   Heat-scored proactive eviction of cold compressed micro-blocks (U, V, anchors)
#   to pinned CPU RAM, freeing GPU pool capacity for longer contexts.
KTRANSFORMERS_DEFAULTS = {
    "DKV_TIER_ENABLED":       "auto",  # "auto"=on for cuda/mps, "0"=off, "1"=force-on
    "DKV_TIER_EVICT_THRESH":  "0.80",  # pool fill ratio that triggers eviction (0.0–1.0)
    "DKV_TIER_EVICT_BATCH":   "32",    # slots evicted per trigger

    # Feature 2: Async Step-Ahead Block Prefetching
    #   After routing selects top-K for step T, issues async H2D restore of any
    #   TIERED_CPU slots so they are warm before step T+1 begins.
    "DKV_BLOCK_PREFETCH":     "1",     # "1"=enable, "0"=disable
    "DKV_PREFETCH_LOOKAHEAD": "1",     # steps ahead to prefetch (keep at 1)

    # Feature 3: Vectorized / Tiled Expansion Kernels
    #   CUDA: Triton autotune over RANK configs fires automatically when Triton is
    #         available (no knob needed — the existing kernel gains autotune configs).
    #   CUDA: INT8 V quantization in Triton kernel (halves V bandwidth, ~0.3% error).
    #   MLX:  mx.fast.metal_kernel tile for U@V on Apple Silicon.
    "DKV_TRITON_INT8_V":      "0",     # "1"=quantize V_KV to int8 (CUDA Triton path)
    "DKV_MLX_METAL_EXPAND":   "auto",  # "1"=force Metal, "0"=pure-MLX, "auto"=try Metal

    # Feature 4: MLA-style Latent Projection (experimental — off by default)
    #   Projects K/V through a shared W [feat_dim → latent_dim] matrix before the
    #   per-block SVD, compounding memory compression.  Requires a calibration phase
    #   (first n_calib blocks initialise W via incremental PCA).
    #   Validate NIAH recall before enabling in production.
    "DKV_MLA_LATENT":         "0",     # "1"=enable latent projection
    "DKV_MLA_LATENT_DIM":     "0",     # 0=auto (kv_heads*head_dim // 4)
    "DKV_MLA_CALIB_BLOCKS":   "16",    # blocks accumulated before PCA initialises W
}


def apply_best_decode_defaults(log=None):
    """setdefault the best serving decode config (an explicit env still wins).

    Call BEFORE constructing the DKV wrapper/manager. ``log`` is an optional
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
        log("DKV decode config: " + ", ".join(f"{k}={v}" for k, v in core_cfg.items()))
        log("DKV kTransformers: " + ", ".join(f"{k}={v}" for k, v in kt_cfg.items()))
    return resolved
