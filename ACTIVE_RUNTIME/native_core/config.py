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

        # Apply preset defaults
        if self.preset == "low":
            self.decode_cache_enabled = False
            self.decode_cache_max_tokens = 0
            self.prefill_chunk_size = 256
            self.srl_threshold = 30
            self.async_svd = False
            self.mps_watermark = 0.7
            self.torch_compile = False
            self.approximate_attn = False
        elif self.preset == "high":
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 16384
            self.prefill_chunk_size = 2048
            self.srl_threshold = 100
            self.async_svd = True
            self.mps_watermark = 0.0
            self.torch_compile = True
            self.approximate_attn = False
        else:  # "mid" (Default)
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 4096
            self.prefill_chunk_size = 512
            self.srl_threshold = 50
            self.async_svd = True
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = False

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

        # Print telemetry when verbose/telemetry enabled
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
