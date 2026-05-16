import platform
import torch
import psutil
import json

class HardwareManifest:
    @staticmethod
    def capture_environment():
        manifest = {
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu": platform.processor(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpus": []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                manifest["gpus"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    "compute_capability": torch.cuda.get_device_capability(i)
                })
        return manifest

    @staticmethod
    def export_manifest(path="results/hardware_manifest.json"):
        manifest = HardwareManifest.capture_environment()
        with open(path, "w") as f:
            json.dump(manifest, f, indent=4)
        return manifest
