import platform
import torch
import psutil
import json
import os

class EnvironmentSnapshot:
    """
    Phase 18F: Captures the physical environment fingerprint for reproducibility.
    """
    def __init__(self, export_dir: str = "results/reconstruction_18/"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def capture(self):
        snapshot = {
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu": platform.processor(),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "gpu_vram_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": platform.python_version(),
            "torch_version": torch.version.__version__
        }
        
        path = os.path.join(self.export_dir, "environment_snapshot.json")
        with open(path, 'w') as f:
            json.dump(snapshot, f, indent=4)
            
        return snapshot, path

if __name__ == "__main__":
    es = EnvironmentSnapshot()
    snap, p = es.capture()
    print(f"Snapshot captured: {p}")
