import os
from typing import Dict, Any

class DiffKVConfig:
    def __init__(self, config_dict: Dict[str, Any] = None):
        config_dict = config_dict or {}

        # 1. Detect preset
        # Preset can be specified in config_dict['preset'] or environment variable DIFFKV_PRESET.
        # Default is "mid".
        preset = config_dict.get("preset", os.environ.get("DIFFKV_PRESET", "mid")).lower()
        if preset not in ("low", "mid", "high"):
            preset = "mid"
        self.preset = preset

        import sys
        is_macos = (sys.platform == "darwin")

        # Apply preset defaults
        if self.preset == "low":
            self.decode_cache_enabled = False
            self.decode_cache_max_tokens = 0
            self.prefill_chunk_size = 256
            self.srl_threshold = 30
            self.async_svd = False
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.01
            self.kv_quant = "q4_0"
            self.max_active_dense_tokens = 1024
            self.max_residual_tokens = 8
        elif self.preset == "high":
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 16384
            self.prefill_chunk_size = 2048
            self.srl_threshold = 100
            self.async_svd = False if is_macos else True  # Disable background async SVD on macOS for MPS stability
            self.mps_watermark = 0.0
            self.torch_compile = False if is_macos else True
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.01
            self.kv_quant = "f16"
            self.max_active_dense_tokens = 4096
            # On CUDA, allow more residuals for better correction at high quality
            self.max_residual_tokens = 8 if is_macos else 16
        else:  # "mid" (Default)
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 4096
            self.prefill_chunk_size = 512
            self.srl_threshold = 50
            self.async_svd = False if is_macos else True  # Disable background async SVD on macOS for MPS stability
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.01
            self.kv_quant = "q8_0"
            self.max_active_dense_tokens = 2048
            self.max_residual_tokens = 8

        # 2. Individual options overrides (dict or env variables)
        self.decode_cache_enabled = self._get_bool(
            "decode_cache_enabled", "DIFFKV_DECODE_CACHE_ENABLED", self.decode_cache_enabled, config_dict
        )
        self.decode_cache_max_tokens = self._get_int(
            "decode_cache_max_tokens", "DIFFKV_DECODE_CACHE_MAX_TOKENS", self.decode_cache_max_tokens, config_dict
        )
        self.prefill_chunk_size = self._get_int(
            "prefill_chunk_size", "DIFFKV_PREFILL_CHUNK_SIZE", self.prefill_chunk_size, config_dict
        )
        self.srl_threshold = self._get_int(
            "srl_threshold", "DIFFKV_SRL_THRESHOLD", self.srl_threshold, config_dict
        )
        self.async_svd = self._get_bool(
            "async_svd", "DIFFKV_ASYNC_SVD", self.async_svd, config_dict
        )
        self.mps_watermark = self._get_float(
            "mps_watermark", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", self.mps_watermark, config_dict
        )
        self.torch_compile = self._get_bool(
            "torch_compile", "DIFFKV_USE_TORCH_COMPILE", self.torch_compile, config_dict
        )
        self.approximate_attn = self._get_bool(
            "approximate_attn", "DIFFKV_MPS_APPROXIMATE_ATTN", self.approximate_attn, config_dict
        )
        self.srl_age_penalty = self._get_float(
            "srl_age_penalty", "DIFFKV_SRL_AGE_PENALTY", self.srl_age_penalty, config_dict
        )
        self.kv_quant = self._get_str(
            "kv_quant", "DIFFKV_KV_QUANT", self.kv_quant, config_dict
        )
        self.max_active_dense_tokens = self._get_int(
            "max_active_dense_tokens", "DIFFKV_MAX_ACTIVE_DENSE_TOKENS", self.max_active_dense_tokens, config_dict
        )
        # Issue 10: max_residual_tokens — configurable upper bound on correction slots per block.
        # NativeBlockPool reads DIFFKV_MAX_RESIDUAL_TOKENS directly for backward-compat;
        # DiffKVConfig surfaces it here for callers that pass config objects.
        self.max_residual_tokens = self._get_int(
            "max_residual_tokens", "DIFFKV_MAX_RESIDUAL_TOKENS", self.max_residual_tokens, config_dict
        )

        # ── CUDA-specific performance flags ──────────────────────────────────
        # These have no effect on MPS/CPU; they are documented here so that
        # DIFFKV_TELEMETRY=1 output gives a complete picture of active defaults.

        # factual_store: retain full prefill K/V on CPU and build FactualExactStore.
        # Default OFF to match MLX path and documentation.  Enable with
        # DIFFKV_FACTUAL_STORE=1 when factual-recall accuracy matters more than
        # the additional RAM/D2H cost.
        self.factual_store = self._get_bool(
            "factual_store", "DIFFKV_FACTUAL_STORE", False, config_dict
        )

        # gpu_compress: run randomized SVD on the GPU instead of CPU workers.
        # Default ON for CUDA (GPU-rSVD is ~30× faster than CPU rSVD for typical
        # rank/block sizes).  Force CPU with DIFFKV_GPU_COMPRESS=0.
        _cuda_default_gpu_compress = not is_macos
        self.gpu_compress = self._get_bool(
            "gpu_compress", "DIFFKV_GPU_COMPRESS", _cuda_default_gpu_compress, config_dict
        )

        # cuda_graph: capture a static CUDA decode graph.
        # Default OFF until the graph ABI is redesigned around device-resident
        # routing/session state.  The current implementation captures mutable
        # Python state and produces stale outputs after any routing change.
        # Enable with DIFFKV_DISABLE_CUDA_GRAPH=0.
        self.cuda_graph = self._get_bool(
            "cuda_graph", "__DIFFKV_CUDA_GRAPH_PLACEHOLDER", False, config_dict
        )
        # Read disable flag directly for parity with static_decode_graph.py.
        if not is_macos:
            _disable_graph = os.environ.get("DIFFKV_DISABLE_CUDA_GRAPH", "1")
            self.cuda_graph = (_disable_graph != "1")

        # gc_interval: decode steps between torch.cuda.empty_cache() calls.
        # 500 on CUDA amortises allocator overhead without large fragmentation.
        # 100 on MPS matches the original value (MPS memory model differs).
        _default_gc = 100 if is_macos else 500
        self.gc_interval = self._get_int(
            "gc_interval", "DIFFKV_GC_INTERVAL", _default_gc, config_dict
        )

        # 3. Per-layer rank options
        # early_layer_rank_boost: when True, layers in the first 15% of the network
        # use up to 2× base_rank to improve syntactic representation quality.
        # Default: False for backward compatibility.
        # Enable via: config_dict={'early_layer_rank_boost': True} or DIFFKV_EARLY_LAYER_RANK_BOOST=1
        self.early_layer_rank_boost = self._get_bool(
            "early_layer_rank_boost", "DIFFKV_EARLY_LAYER_RANK_BOOST", False, config_dict
        )
        # max_rank_early: cap for early-layer rank. 0 = auto (2× base_rank).
        # Only used when early_layer_rank_boost=True.
        # Enable via: config_dict={'max_rank_early': 32} or DIFFKV_MAX_RANK_EARLY=32
        self.max_rank_early = self._get_int(
            "max_rank_early", "DIFFKV_MAX_RANK_EARLY", 0, config_dict
        )

        verbose = os.environ.get("DIFFKV_TELEMETRY", "0") == "1"
        if verbose:
            print(f"[DiffKV Config] Loaded preset: {self.preset.upper()}")
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
            print(f"  kv_quant                  = {self.kv_quant}")
            print(f"  max_active_dense_tokens   = {self.max_active_dense_tokens}")
            if self.early_layer_rank_boost:
                print(f"  max_rank_early            = {self.max_rank_early} (0=auto 2×base)")
            if not is_macos:
                print(f"  --- CUDA-specific ---")
                print(f"  factual_store             = {self.factual_store}")
                print(f"  gpu_compress              = {self.gpu_compress}")
                print(f"  cuda_graph                = {self.cuda_graph}")
                print(f"  gc_interval               = {self.gc_interval}")

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
