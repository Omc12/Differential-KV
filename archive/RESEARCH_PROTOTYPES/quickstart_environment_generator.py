"""
quickstart_environment_generator.py

Quickstart environment generator for Differential KV.
Creates starter configurations and local development presets.
"""

import os
import json
from typing import Dict, Any

class QuickstartEnvironmentGenerator:
    """
    Sets up a local workspace for new Differential KV developers.
    """
    def __init__(self, workspace_path: str = "workspace"):
        self.workspace_path = workspace_path

    def initialize_workspace(self) -> str:
        """Creates the workspace directory and initial config."""
        os.makedirs(self.workspace_path, exist_ok=True)
        
        config = {
            "model_path": "models/qwen-7b",
            "sparse_mode": "lowrank_sparse",
            "block_size": 64,
            "device": "cuda",
            "server": {
                "host": "127.0.0.1",
                "port": 8000
            }
        }
        
        config_path = os.path.join(self.workspace_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
        return config_path

    def create_env_template(self):
        """Generates a .env template."""
        content = """# Differential KV Environment Variables
DIFFKV_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
DIFFKV_VRAM_LIMIT_GB=12
DIFFKV_LOG_LEVEL=INFO
"""
        env_path = os.path.join(self.workspace_path, ".env.example")
        with open(env_path, "w") as f:
            f.write(content)
            
    def get_presets(self) -> Dict[str, Dict[str, Any]]:
        """Returns hardware presets."""
        return {
            "rtx_4090": {"sparse_mode": "lowrank_sparse", "block_size": 128},
            "a100": {"sparse_mode": "shared_basis", "block_size": 64},
            "cpu_fallback": {"sparse_mode": "int8", "block_size": 32}
        }

if __name__ == "__main__":
    generator = QuickstartEnvironmentGenerator()
    print(f"Workspace initialized: {generator.initialize_workspace()}")
