"""
reproducibility_controller.py

Ensures deterministic benchmark execution and environment capture.
Handles seeds, repeated trials, and hardware/dependency manifests.
"""

import os
import sys
import json
import torch
import numpy as np
import random
import platform
import subprocess
from typing import Dict, Any, List

class ReproducibilityController:
    """
    Enforces deterministic execution and captures the exact environment state.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.trials = 3

    def enforce_determinism(self):
        """Sets all seeds and CUDA flags for deterministic execution."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # Note: These may impact performance but are required for scientific reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"[CBP] Determinism enforced with seed: {self.seed}")

    def capture_hardware_manifest(self) -> Dict[str, Any]:
        """Captures detailed hardware information."""
        manifest = {
            "os": platform.system(),
            "os_release": platform.release(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "gpu": []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                manifest["gpu"].append({
                    "name": prop.name,
                    "total_memory_gb": prop.total_memory / (1024**3),
                    "multi_processor_count": prop.multi_processor_count,
                    "capability": f"{prop.major}.{prop.minor}"
                })
        return manifest

    def capture_environment_manifest(self) -> Dict[str, Any]:
        """Captures dependencies and environment state."""
        try:
            pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
            dependencies = pip_freeze.splitlines()
        except:
            dependencies = ["Could not capture pip freeze"]
            
        return {
            "dependencies": dependencies,
            "env_vars": {k: v for k, v in os.environ.items() if "CUDA" in k or "PYTHON" in k or "KV" in k}
        }

    def export_reproducibility_package(self, path: str = "benchmark_reproducibility.json"):
        """Exports the full reproducibility context."""
        package = {
            "seed": self.seed,
            "trials": self.trials,
            "hardware": self.capture_hardware_manifest(),
            "environment": self.capture_environment_manifest()
        }
        with open(path, "w") as f:
            json.dump(package, f, indent=4)
        print(f"[CBP] Reproducibility manifest saved to {path}")
        return package

# Global instance
repro_controller = ReproducibilityController()
