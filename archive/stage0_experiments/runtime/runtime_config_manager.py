import json
import os
from typing import Dict, Any, Optional

class RuntimeConfigManager:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        self.current_config = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        return {
            "model": "7B-Sparse-Stub",
            "device": "cuda",
            "vram_limit_gb": 8,
            "sparse_budget": 0.1,
            "paging_policy": "lru",
            "max_context": 32768,
            "batch_size": 1
        }

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, f"{profile_name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.current_config.update(json.load(f))
        return self.current_config

    def save_profile(self, profile_name: str, config: Dict[str, Any]):
        path = os.path.join(self.config_dir, f"{profile_name}.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    def get_config(self) -> Dict[str, Any]:
        return self.current_config
