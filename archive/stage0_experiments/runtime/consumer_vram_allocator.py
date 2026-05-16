import torch
import psutil

class ConsumerVRAMAllocator:
    """
    Manages VRAM allocation strategies for 12GB GPUs (like RTX 4070 Super).
    Ensures that 7B models + long context don't cause OOM.
    """
    def __init__(self, target_vram_gb: float = 12.0):
        self.target_vram_gb = target_vram_gb
        self.safety_margin = 0.5 # GB

    def get_optimal_quantization(self) -> str:
        """
        Suggests quantization based on available VRAM.
        For 12GB, 4bit is recommended for 7B models with long context.
        """
        if not torch.cuda.is_available():
            return "cpu"
            
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram >= 16.0:
            return "8bit"
        else:
            return "4bit"

    def check_memory_drift(self):
        """Monitor for fragmentation or leaks."""
        stats = torch.cuda.memory_stats()
        # active.all.current / (1024**3)
        active = stats.get("active.all.current", 0) / (1024**3)
        reserved = stats.get("reserved_bytes.all.current", 0) / (1024**3)
        
        return {
            "active_gb": active,
            "reserved_gb": reserved,
            "fragmentation_gb": reserved - active
        }

    def force_cleanup(self):
        torch.cuda.empty_cache()
        import gc
        gc.collect()
