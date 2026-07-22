import os
import json
import yaml
from typing import Dict, Any, Optional

class EnvironmentConfigurationManager:
    """
    Centralizes runtime configuration, handles environment overrides,
    and manages deployment profiles.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/default_profile.yaml"
        self.config = self._load_initial_config()
        self._apply_env_overrides()

    def _load_initial_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        
        # Default fallback config
        return {
            "model_name": "qwen2.5-7b-sparse",
            "serving": {
                "port": 8000,
                "host": "0.0.0.0",
                "max_sessions": 10,
                "concurrency": 4
            },
            "runtime": {
                "device": "cuda",
                "vram_limit_gb": 12,
                "sparse_budget": 0.15,
                "max_context": 32768
            },
            "observability": {
                "enable_prometheus": True,
                "metrics_port": 9090
            }
        }

    def _apply_env_overrides(self):
        """
        Applies environment variable overrides.
        Example: DKV_RUNTIME__DEVICE=cpu overrides runtime.device
        Nesting is handled via double underscores (__).
        """
        for key, value in os.environ.items():
            if key.startswith("DKV_"):
                parts = key.replace("DKV_", "").lower().split("__")
                self._update_nested_config(self.config, parts, value)

    def _update_nested_config(self, d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        
        # Try to cast value to appropriate type
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        
        d[keys[-1]] = value

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def save_profile(self, profile_name: str):
        os.makedirs("configs", exist_ok=True)
        path = f"configs/{profile_name}.yaml"
        with open(path, 'w') as f:
            yaml.dump(self.config, f)
        print(f"[ECM] Saved configuration profile to {path}")

    def validate_safety(self) -> bool:
        """
        Validates that the configuration is safe for the current environment.
        """
        if self.config["runtime"]["device"] == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            print("[ECM] WARNING: CUDA device requested but CUDA_VISIBLE_DEVICES not set.")
        
        if self.config["runtime"]["vram_limit_gb"] > 24:
            print("[ECM] WARNING: High VRAM limit requested. Ensure hardware compatibility.")
            
        return True
