import torch
import psutil
import os
from runtime.runtime_config_manager import RuntimeConfigManager
from serving.real_sparse_serving_runtime import RealSparseServingRuntime

class SparseRuntimeLauncher:
    def __init__(self):
        self.config_manager = RuntimeConfigManager()
        self.runtime = None

    def detect_hardware(self):
        vram_total = 0
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        ram_total = psutil.virtual_memory().total / (1024**3)
        
        return {
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "vram_total_gb": vram_total,
            "ram_total_gb": ram_total
        }

    def launch(self, profile: Optional[str] = None):
        hw = self.detect_hardware()
        config = self.config_manager.get_config()
        if profile:
            config = self.config_manager.load_profile(profile)
            
        # Adjust config based on hardware
        if hw["vram_total_gb"] < config["vram_limit_gb"]:
            print(f"[WARNING] VRAM limit {config['vram_limit_gb']}GB exceeds hardware {hw['vram_total_gb']:.2f}GB. Capping.")
            config["vram_limit_gb"] = hw["vram_total_gb"] * 0.9
            
        print(f"[INFO] Launching DKV Runtime with config: {config}")
        self.runtime = RealSparseServingRuntime(model_name=config["model"])
        return self.runtime

if __name__ == "__main__":
    launcher = SparseRuntimeLauncher()
    hw_info = launcher.detect_hardware()
    print(f"Hardware Detected: {hw_info}")
    launcher.launch()
