import json
import os
import platform
import torch
import time

class RuntimeManifestExporter:
    """
    Exports exact runtime configuration for external reproducibility.
    MANDATORY for Phase 18 reporting.
    """
    def __init__(self, export_dir: str = "results/reconstruction_18/"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        self.manifest = {
            "timestamp": time.time(),
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                "cuda_version": torch.version.cuda,
                "vram_total": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            },
            "software": {
                "torch_version": torch.version.__version__,
                "python_version": platform.python_version()
            },
            "model_loads": []
        }

    def record_load(self, model_id, quant_config, load_time):
        self.manifest["model_loads"].append({
            "model_id": model_id,
            "quantization": str(quant_config),
            "load_time_seconds": load_time
        })
        self.export()

    def export(self, filename: str = "runtime_manifest.json"):
        path = os.path.join(self.export_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=4)
        # print(f"[PHASE 18D] Runtime manifest exported to {path}")
        return path
