import os
from typing import Dict, Any

class DKVConfig:
    def __init__(self, config_dict: Dict[str, Any] = None):
        config_dict = config_dict or {}

        # 1. Detect preset
        # Preset can be specified in config_dict['preset'] or environment variable DKV_PRESET.
        # Default is "mid".
        preset = config_dict.get("preset", os.environ.get("DKV_PRESET", "mid")).lower()
        if preset not in ("low", "mid", "high", "ultra"):
            preset = "mid"
        self.preset = preset

        import sys
        is_macos = (sys.platform == "darwin")

        # Apply preset defaults
        # NOTE on CUDA prefill_chunk_size: ingest_chunk creates full blocks of exactly
        # (1 + micro_block_size) tokens — 1 anchor + micro_block_size active keys.
        # micro_block_size defaults to 256, so block_capacity = 257.
        # prefill_chunk_size MUST be >= 2 * block_capacity (= 514) so that at least one
        # full block is produced per inner chunk.  On macOS/MPS the chunk size is already
        # small (256) because MLX handles compression differently; on CUDA we need larger
        # chunks or every chunk produces only a partial block and nothing is ever compressed.
        if self.preset == "low":
            self.decode_cache_enabled = False
            self.decode_cache_max_tokens = 0
            # CUDA: 1024 ensures ≥3 full blocks (3×257=771 < 1024) per inner chunk.
            # macOS/MLX keeps 256 (handled post-forward by compress_deferred_prefill_blocks).
            self.prefill_chunk_size = 256 if is_macos else 1024
            self.srl_threshold = 30
            self.async_svd = False if is_macos else True
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.0  # MLX parity: pure relevance, no recency bias (see override note)
            self.kv_quant = "q4_0"
            self.max_active_dense_tokens = 1024
            # Residual budget per block — see the "mid" branch for the full
            # rationale.  `low` is memory-priority.
            #
            # CAUTION: the old justification here ("40 already covers the
            # adaptive prose cap int(0.15*256)=38, so it loses essentially
            # nothing"), and the 13.4K A/B that "confirmed res40 matched res128
            # quality", are both INVALID.  They were measured while the budget
            # was bugged to start from int(0.15*n)=38 (handoff §10j), which
            # capped 40 and 128 to the same 38 exact tokens — of course they
            # matched.  With that cap removed this ladder is real for the first
            # time, and 40 now genuinely stores ~3x fewer exact tokens per block
            # than `high`.  Re-measure before treating `low` as quality-neutral.
            self.max_residual_tokens = 40
            # Spectral energy a block's low-rank form must retain, and the rank
            # ceiling that serves it. This -- not `rank` -- is what actually sets
            # a block's rank: the compressor keeps the smallest k carrying this
            # fraction of the energy, so raising the ceiling alone does nothing
            # (asking for rank 32 vs 128 moved the real median rank only 24->34).
            #
            # Measured on Qwen3.5-2B at 16k, multifact synthesis, block 1024:
            #   0.999   / rank 32   -> 30.0   peak_alloc 5.07 GB
            #   0.9999  / rank 64   -> 43.3   peak_alloc 5.16 GB
            #   0.99999 / rank 128  -> 46.7   peak_alloc 5.41 GB
            # TTFT is flat across all three (10.45-10.77 s, inside noise), so the
            # cost is VRAM, not latency. Distractor retrieval stays 24/24 at
            # every setting.
            self.svd_energy = 0.999
            self.rank = 32
        elif self.preset == "high":
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 16384
            self.prefill_chunk_size = 2048
            self.srl_threshold = 100
            self.async_svd = False if is_macos else True  # Disable background async SVD on macOS for MPS stability
            self.mps_watermark = 0.0
            self.torch_compile = False if is_macos else True
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.0  # MLX parity: pure relevance, no recency bias (see override note)
            self.kv_quant = "f16"
            self.max_active_dense_tokens = 4096
            # `high` = max fidelity: full 128-residual ceiling (paper config of
            # record, MLX default) for table/factual-dense docs, accepting the
            # larger pool.  See the "mid" branch for the ladder rationale.
            self.max_residual_tokens = 128
            # Quality end: keep enough of the spectrum that synthesis recovers.
            # Nothing cheaper reproduces this -- selective per-block rank boost
            # (DKV_RANK_BOOST=auto) and routing every token instead of every 4
            # (DKV_REMAT_CACHE=0) both leave synthesis at 30.0, because at this
            # block size K already routes every block and no routing change can
            # add information the model is not already being shown.
            #
            # `high` is the top of the ladder for cost-sensitive quality work.
            # `ultra` goes further (rank 192) and is the only setting that
            # matches dense on synthesis -- see its branch for the sweep and for
            # what it costs in TTFT and VRAM.
            self.svd_energy = 0.99999
            self.rank = 128
        elif self.preset == "ultra":
            # `ultra` is MID with rank 192 -- not `high` with rank 192.
            #
            # That distinction is the whole preset. The rank sweep below was run
            # on top of mid's other settings (RANK= override, default preset), and
            # rank 192 there scores 60.0. Building this branch on `high`'s
            # settings instead scored 50.0 with the SAME rank, so the result
            # needs mid's configuration around it and is not a property of rank
            # alone. Copy mid, change rank, change nothing else.
            #
            # WHICH mid setting, bisected one at a time against high's value:
            # prefill_chunk_size, and ONLY that. srl_threshold 100, kv_quant f16,
            # max_active_dense_tokens 4096 and decode_cache_max_tokens 16384 each
            # left the score untouched; prefill_chunk_size 2048 dropped it from
            # 60.0 to 33.3 (facts 7/15, links 1/5).
            #
            # The mechanism is block formation, not chunking as such. The wrapper
            # rounds the chunk UP to a multiple of block capacity, so with
            # micro_block_size 1024 (capacity 1025) a 1024 chunk becomes 1025 --
            # exactly ONE block per chunk -- and 2048 becomes 2050, two. Forming
            # two blocks per chunk is what costs the synthesis.
            #
            # It is not a rule that smaller is better, and it interacts with rank:
            # at rank 128, chunk 2048 scores 50.0 and chunk 1024 scores 46.7 --
            # the opposite direction. Do not carry "chunk 1024 is better" over to
            # another rank without re-measuring.
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 4096
            self.prefill_chunk_size = 512 if is_macos else 1024
            self.srl_threshold = 50
            self.async_svd = False if is_macos else True
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.0
            self.kv_quant = "q8_0"
            self.max_active_dense_tokens = 2048
            self.max_residual_tokens = 128
            # RANK IS THE DRIVER, NOT ENERGY -- measured by separating them, and
            # it corrects how this ladder was first described. Synthesis at 16k
            # on Qwen3.5-2B, mid's settings, `--tests synthesis` held constant:
            #
            #   energy 0.9999  rank 64  -> 50.0 (facts 6/15, links 3/5)
            #   energy 0.99999 rank 64  -> 50.0 (facts 6/15, links 3/5)
            #   energy 0.9999  rank 128 -> 50.0 (facts 9/15, links 2/5)
            #   energy 0.99999 rank 128 -> 50.0 (facts 9/15, links 2/5)
            #
            # Energy changes nothing at either rank, so the knob is rank alone.
            #
            # The rank landscape is JAGGED, not a quality dial. Do not interpolate:
            #
            #   64 -> 50.0 (6/3)    160 -> 50.0 (9/2)    224 -> 63.3 (10/3)
            #   80 -> 53.3 (7/3)    192 -> 60.0 (9/3)    240 -> 60.0 (9/3)
            #   96 -> 43.3 (7/2)    208 -> 46.7 (8/2)    256 -> 50.0 (9/2)
            #  128 -> 50.0 (9/2)
            #
            # 208 sits between two of the best points and is one of the worst, so
            # neighbouring ranks say nothing about each other.
            #
            # 224 is the setting, at facts 10/15 and links 3/5. THE DENSE CONTROL
            # IS 60.0 (9/15, 3/5), so this is one fact AHEAD of dense -- the first
            # configuration in this project to beat dense rather than match it.
            #
            # Replication at temperature 0 is deterministic and therefore proves
            # nothing, so 224 was checked on conditions it was NOT tuned on:
            #
            #   --tests synthesis @16k (fresh session)  63.3   dense 60.0
            #   full run @16k (warm session)            63.3   dense 60.0
            #   --tests synthesis @8k                   63.3   dense 60.0
            #
            # Holding at 63.3 in the FULL run matters most: rank 192 scores 60.0
            # fresh but collapses to 50.0 once earlier tests have shared the
            # session, and 224 does not. That session-history sensitivity is why
            # 192 is not the choice despite also beating `high`.
            #
            # Cost on Qwen3.5-2B at 32k against mid: TTFT 8.86 -> 10.33 s, device
            # VRAM 5.23 -> 5.93 GB, decode roughly flat. Needle sweep 9/9 with 9/9
            # determinism, linkbench unchanged at 20/24. So the cost is latency and
            # memory only, which is what a preset at this end of the ladder is for.
            self.svd_energy = 0.99999
            self.rank = 224
        else:  # "mid" (Default)
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 4096
            # CUDA: 1024 ensures ≥3 full blocks per inner chunk.
            self.prefill_chunk_size = 512 if is_macos else 1024
            self.srl_threshold = 50
            self.async_svd = False if is_macos else True  # Disable background async SVD on macOS for MPS stability
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.0  # MLX parity: pure relevance, no recency bias (see override note)
            self.kv_quant = "q8_0"
            # Middle of the fidelity ladder -- see the `low` branch for the
            # measurements. 0.9999/64 recovers most of the synthesis that 0.999
            # gives up (30.0 -> 43.3 of the 46.7 that `high` reaches) for a third
            # of its VRAM cost (+0.09 GB against +0.34).
            self.svd_energy = 0.9999
            self.rank = 64
            self.max_active_dense_tokens = 2048
            # Residual budget per block: how many exact (uncompressed) tokens
            # correct the lossy SVD.  This is the main quality dial.
            #
            # It used to be described here as NOT a flat knob, on the grounds
            # that "the compressor caps actual usage at int(0.15*T_active) (=38
            # for T=256) ... so prose blocks never use more than ~38 regardless
            # of this value".  That cap was a BUG, not a design (handoff §10j):
            # the budget started from int(0.15*n) instead of max_residual, so a
            # block could never exceed 38 exact tokens and raising this setting
            # to 128 did literally nothing.  Both compress paths now start from
            # the pool value, and the separate 0.08 error floor that was
            # discarding the budget entirely on ordinary prose is gone (§10k,
            # MLX picks residuals by pure top-k).  The value below is now the
            # real per-block ceiling.
            #
            # The pool allocates this many slots UNIFORMLY per block, so the
            # physical VRAM cost is paid on every block even though most sit
            # mostly empty.  A/B at 13.4K: res128 pool = 2.8 GB, res40 = 1.5 GB,
            # identical output quality on prose synthesis -- which is why `mid`
            # used to be 64.
            #
            # ladder it: `mid` = 64 (covers the prose cap plus boost headroom),
            # `high` = 128 (full table/factual fidelity, accepts the VRAM), and
            # `low` = 40 (memory-priority).  Override with
            # DKV_MAX_RESIDUAL_TOKENS.
            #
            # RAISED TO 128 (2026-07-28) to match MLX, which uses 128 FLAT at
            # every preset (mlx_dkv_wrapper.py: DKV_MAX_RESIDUAL default "128");
            # the 40/64/128 ladder is a CUDA-only invention.
            #
            # Doing so initially produced GARBAGE output, which turned out to be
            # two CUDA-side limits this value reaches and 64 did not:
            #   * the decode kernel's residual scratch was 64 wide while its READ
            #     loops ran to max_residual -- out-of-bounds reads above 64
            #     (fixed: DKV_MAX_RESIDUAL_SHARED in dkv_decode.metal);
            #   * pool sizing divided the budget by a per-slot cost that EXCLUDED
            #     the residual arrays, so the over-allocation grew from 2.2x to
            #     3.3x (fixed in KVRuntimeManager).
            # Neither was a reason to keep 64 -- both were bugs 64 happened to
            # hide. See handoff §9u.
            self.max_residual_tokens = 128

        # 2. Individual options overrides (dict or env variables)
        self.decode_cache_enabled = self._get_bool(
            "decode_cache_enabled", "DKV_DECODE_CACHE_ENABLED", self.decode_cache_enabled, config_dict
        )
        self.decode_cache_max_tokens = self._get_int(
            "decode_cache_max_tokens", "DKV_DECODE_CACHE_MAX_TOKENS", self.decode_cache_max_tokens, config_dict
        )
        self.prefill_chunk_size = self._get_int(
            "prefill_chunk_size", "DKV_PREFILL_CHUNK_SIZE", self.prefill_chunk_size, config_dict
        )
        # Safety guard: prefill_chunk_size must accommodate at least 2 full streaming
        # blocks (each block = 1 anchor + micro_block_size active tokens = 257 tokens
        # at the default micro_block_size=256).  If the resolved value is too small,
        # ingest_chunk produces zero full blocks → all blocks stay ACCUMULATING →
        # dense window overflows at decode → model collapse.  We clamp upward on
        # non-macOS (CUDA) only; on macOS MLX compression runs post-forward so chunks
        # can be small without this constraint.
        import sys as _sys
        if _sys.platform != "darwin":
            _min_chunk = 2 * 257  # 2 × (1 anchor + 256 active) = 514
            if self.prefill_chunk_size < _min_chunk:
                self.prefill_chunk_size = _min_chunk
        self.srl_threshold = self._get_int(
            "srl_threshold", "DKV_SRL_THRESHOLD", self.srl_threshold, config_dict
        )
        self.async_svd = self._get_bool(
            "async_svd", "DKV_ASYNC_SVD", self.async_svd, config_dict
        )
        self.mps_watermark = self._get_float(
            "mps_watermark", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", self.mps_watermark, config_dict
        )
        self.torch_compile = self._get_bool(
            "torch_compile", "DKV_USE_TORCH_COMPILE", self.torch_compile, config_dict
        )
        self.approximate_attn = self._get_bool(
            "approximate_attn", "DKV_MPS_APPROXIMATE_ATTN", self.approximate_attn, config_dict
        )
        # srl_age_penalty: subtracts age*penalty from each block's relevance in
        # two_level_gate, biasing selection toward RECENT blocks.  Default moved
        # from 0.01 to 0.0 to match the MLX router, which ranks blocks purely by
        # q·k relevance with no recency term — a recency bias actively drops
        # early-document content on whole-document synthesis (the likely cause
        # of the routed-decode degradation observed at 13.4K).  Re-enable with
        # DKV_SRL_AGE_PENALTY>0 for multi-turn chat, where damping stale
        # concepts from earlier turns can help.
        self.srl_age_penalty = self._get_float(
            "srl_age_penalty", "DKV_SRL_AGE_PENALTY", self.srl_age_penalty, config_dict
        )
        self.kv_quant = self._get_str(
            "kv_quant", "DKV_KV_QUANT", self.kv_quant, config_dict
        )
        self.max_active_dense_tokens = self._get_int(
            "max_active_dense_tokens", "DKV_MAX_ACTIVE_DENSE_TOKENS", self.max_active_dense_tokens, config_dict
        )
        # Issue 10: max_residual_tokens — configurable upper bound on correction slots per block.
        # NativeBlockPool reads DKV_MAX_RESIDUAL_TOKENS directly for backward-compat;
        # DKVConfig surfaces it here for callers that pass config objects.
        self.max_residual_tokens = self._get_int(
            "max_residual_tokens", "DKV_MAX_RESIDUAL_TOKENS", self.max_residual_tokens, config_dict,
            alias_env="DKV_MAX_RESIDUAL",   # MLX's name for the same knob
        )

        # ── CUDA-specific performance flags ──────────────────────────────────
        # These have no effect on MPS/CPU; they are documented here so that
        # DKV_TELEMETRY=1 output gives a complete picture of active defaults.

        # factual_store: retain full prefill K/V on CPU and build FactualExactStore.
        # Default OFF to match MLX path and documentation.  Enable with
        # DKV_FACTUAL_STORE=1 when factual-recall accuracy matters more than
        # the additional RAM/D2H cost.
        self.factual_store = self._get_bool(
            "factual_store", "DKV_FACTUAL_STORE", False, config_dict
        )

        # gpu_compress: run randomized SVD on the GPU instead of CPU workers.
        # Default ON for CUDA (GPU-rSVD is ~30× faster than CPU rSVD for typical
        # rank/block sizes).  Force CPU with DKV_GPU_COMPRESS=0.
        _cuda_default_gpu_compress = not is_macos
        self.gpu_compress = self._get_bool(
            "gpu_compress", "DKV_GPU_COMPRESS", _cuda_default_gpu_compress, config_dict
        )

        # cuda_graph: capture a static CUDA decode graph.
        # Default OFF until the graph ABI is redesigned around device-resident
        # routing/session state.  The current implementation captures mutable
        # Python state and produces stale outputs after any routing change.
        # DKV_DISABLE_CUDA_GRAPH=0 is retained as a compatibility request,
        # but the current mutable model does not have the static-state ABI
        # required for a valid full-forward graph.  Keep the effective flag
        # false so config telemetry cannot claim graphs are active merely
        # because an environment variable was set.
        _disable_graph = os.environ.get("DKV_DISABLE_CUDA_GRAPH", "1")
        self.cuda_graph_requested = (not is_macos and _disable_graph != "1")
        self.cuda_graph = False

        # gc_interval: decode steps between torch.cuda.empty_cache() calls.
        # 500 on CUDA amortises allocator overhead without large fragmentation.
        # 100 on MPS matches the original value (MPS memory model differs).
        _default_gc = 100 if is_macos else 500
        self.gc_interval = self._get_int(
            "gc_interval", "DKV_GC_INTERVAL", _default_gc, config_dict
        )

        # srl_route_every: run route_query_fixed_k every N decode tokens; reuse
        # cached slots in between.  Reduces D2H traffic from SRL entropy/.item()
        # and centroid/.tolist() calls during long generations.
        # Default 1 = every token (preserves original behaviour).
        # Set to 2-4 on CUDA for 2-4× less D2H during SRL-routed decode.
        self.srl_route_every = self._get_int(
            "srl_route_every", "DKV_SRL_ROUTE_EVERY", 1, config_dict
        )

        # 3. Per-layer rank options
        # early_layer_rank_boost: when True, layers in the first 15% of the network
        # use up to 2× base_rank to improve syntactic representation quality.
        # Default: False for backward compatibility.
        # Enable via: config_dict={'early_layer_rank_boost': True} or DKV_EARLY_LAYER_RANK_BOOST=1
        self.early_layer_rank_boost = self._get_bool(
            "early_layer_rank_boost", "DKV_EARLY_LAYER_RANK_BOOST", False, config_dict
        )
        # max_rank_early: cap for early-layer rank. 0 = auto (2× base_rank).
        # Only used when early_layer_rank_boost=True.
        # Enable via: config_dict={'max_rank_early': 32} or DKV_MAX_RANK_EARLY=32
        self.max_rank_early = self._get_int(
            "max_rank_early", "DKV_MAX_RANK_EARLY", 0, config_dict
        )
        # layer_adaptive_rank: when True, early/late layers use lower ranks (e.g. 8 or 12)
        # and middle layers use higher ranks (e.g. 24), rather than a uniform rank.
        # Default: True. Disable via DKV_LAYER_ADAPTIVE_RANK=0 or config dict.
        # This is a major win for decode throughput (TPS) and VRAM reduction on both CUDA and MLX.
        self.layer_adaptive_rank = self._get_bool(
            "layer_adaptive_rank", "DKV_LAYER_ADAPTIVE_RANK", True, config_dict
        )
        # ── Streaming Compression Default Tradeoffs ───────────────────────────
        # DKV_STREAMING_COMPRESS defaults:
        # - CUDA: OFF (0). SVD compression is a highly parallelizable operation.
        #   Doing it layer-by-layer during the forward pass forces sequential
        #   GPU dispatches (e.g. 624 dispatches for 13k context), incurring massive
        #   launch overhead and serialized latency. Batched deferred SVD at the end
        #   is 20x faster.
        # - MLX: ON (1). macOS unified memory and low launch overhead make streaming
        #   compression critical for bounding peak VRAM without performance penalty.
        # ──────────────────────────────────────────────────────────────────────

        # Publish the fidelity target where the compressor reads it. lowrank's
        # _svd_energy_target() consults DKV_SVD_ENERGY at call time; setdefault so
        # an explicit environment override still wins over the preset.
        os.environ.setdefault("DKV_SVD_ENERGY", str(getattr(self, "svd_energy", 0.999)))

        verbose = os.environ.get("DKV_TELEMETRY", "0") == "1"
        if verbose:
            print(f"[DKV Config] Loaded preset: {self.preset.upper()}")
            print(f"  svd_energy / rank         = {getattr(self, 'svd_energy', None)} / {self.rank}")
            print(f"  decode_cache_enabled      = {self.decode_cache_enabled}")
            print(f"  decode_cache_max_tokens   = {self.decode_cache_max_tokens}")
            print(f"  prefill_chunk_size        = {self.prefill_chunk_size}")
            print(f"  srl_threshold             = {self.srl_threshold}")
            print(f"  async_svd                 = {self.async_svd}")
            print(f"  mps_watermark             = {self.mps_watermark}")
            print(f"  torch_compile             = {self.torch_compile}")
            print(f"  approximate_attn          = {self.approximate_attn}")
            print(f"  srl_age_penalty           = {self.srl_age_penalty}")
            print(f"  early_layer_rank_boost    = {self.early_layer_rank_boost}")
            print(f"  layer_adaptive_rank       = {self.layer_adaptive_rank}")
            print(f"  kv_quant                  = {self.kv_quant}")
            print(f"  max_active_dense_tokens   = {self.max_active_dense_tokens}")
            if self.early_layer_rank_boost:
                print(f"  max_rank_early            = {self.max_rank_early} (0=auto 2×base)")
            if not is_macos:
                print(f"  --- CUDA-specific ---")
                print(f"  factual_store             = {self.factual_store}")
                print(f"  gpu_compress              = {self.gpu_compress}")
                _graph_note = "static ABI unavailable"
                if self.cuda_graph_requested:
                    _graph_note += "; request ignored"
                print(f"  cuda_graph                = {self.cuda_graph} ({_graph_note})")
                print(f"  gc_interval               = {self.gc_interval}")
                print(f"  srl_route_every           = {self.srl_route_every}")

    def _get_bool(self, key: str, env_name: str, default: bool, config_dict: dict) -> bool:
        if key in config_dict:
            val = config_dict[key]
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("true", "1", "yes", "on")
        env_val = os.environ.get(env_name)
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes", "on")
        return default

    def _get_int(self, key: str, env_name: str, default: int, config_dict: dict,
                 alias_env: str = None) -> int:
        """`alias_env` is a second accepted name for the SAME knob.

        The two runtimes grew different names for identical settings (MLX calls
        the residual budget DKV_MAX_RESIDUAL, this side DKV_MAX_RESIDUAL_TOKENS),
        so a config written against MLX silently configured nothing here. The
        primary name still wins when both are set.
        """
        if key in config_dict:
            try:
                return int(config_dict[key])
            except (ValueError, TypeError):
                pass
        for name in (env_name, alias_env):
            if not name:
                continue
            env_val = os.environ.get(name)
            if env_val is not None:
                try:
                    return int(env_val)
                except (ValueError, TypeError):
                    pass
        return default

    def _get_float(self, key: str, env_name: str, default: float, config_dict: dict) -> float:
        if key in config_dict:
            try:
                return float(config_dict[key])
            except (ValueError, TypeError):
                pass
        env_val = os.environ.get(env_name)
        if env_val is not None:
            try:
                return float(env_val)
            except (ValueError, TypeError):
                pass
        return default

    def _get_str(self, key: str, env_name: str, default: str, config_dict: dict) -> str:
        if key in config_dict:
            return str(config_dict[key])
        env_val = os.environ.get(env_name)
        if env_val is not None:
            return env_val
        return default
