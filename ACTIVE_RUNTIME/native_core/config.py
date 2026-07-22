import os
from typing import Dict, Any

class DKVConfig:
    def __init__(self, config_dict: Dict[str, Any] = None):
        config_dict = config_dict or {}

        # 1. Detect preset
        # Preset can be specified in config_dict['preset'] or environment variable DKV_PRESET.
        # Default is "mid".
        preset = config_dict.get("preset", os.environ.get("DKV_PRESET", "mid")).lower()
        if preset not in ("low", "mid", "high"):
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
            # rationale.  The presets ladder it as a memory/quality dial: `low`
            # is memory-priority, and 40 already covers the adaptive prose cap
            # (int(0.15*256)=38), so it loses essentially nothing on prose while
            # roughly halving the pool vs 128.  A/B at 13.4K confirmed res40
            # matched res128 output quality at ~1.5 GB vs ~2.8 GB pool.
            self.max_residual_tokens = 40
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
            self.max_active_dense_tokens = 2048
            # Residual budget per block: how many exact (uncompressed) tokens
            # correct the lossy SVD.  This is the main quality dial, but it is
            # NOT a flat cost=benefit knob.  The compressor caps actual usage at
            # n_max_residual = int(0.15*T_active) (=38 for T=256), clamped down
            # to 8/16 for low-error blocks and raised only for digit/table
            # blocks (up to T_active).  So prose blocks never use more than ~38
            # regardless of this value; only factual/table-dense blocks benefit
            # from a higher ceiling.
            #
            # The pool allocates this many slots UNIFORMLY per block, so the
            # physical VRAM cost is paid on every block even though most sit
            # mostly empty.  A/B at 13.4K: res128 pool = 2.8 GB, res40 = 1.5 GB,
            # identical output quality on prose synthesis.  So the presets
            # ladder it: `mid` = 64 (covers the prose cap plus boost headroom),
            # `high` = 128 (full table/factual fidelity, accepts the VRAM), and
            # `low` = 40 (memory-priority).  Override with
            # DKV_MAX_RESIDUAL_TOKENS; raise toward 128+ for table-heavy or
            # exact-recall (needle) workloads.
            self.max_residual_tokens = 64

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
            "max_residual_tokens", "DKV_MAX_RESIDUAL_TOKENS", self.max_residual_tokens, config_dict
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

        verbose = os.environ.get("DKV_TELEMETRY", "0") == "1"
        if verbose:
            print(f"[DKV Config] Loaded preset: {self.preset.upper()}")
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

    def _get_int(self, key: str, env_name: str, default: int, config_dict: dict) -> int:
        if key in config_dict:
            try:
                return int(config_dict[key])
            except (ValueError, TypeError):
                pass
        env_val = os.environ.get(env_name)
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
