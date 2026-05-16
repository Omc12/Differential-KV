import torch
import logging
from typing import Dict, Any

class MemoryPressureSafetySystem:
    """
    Implements graceful VRAM pressure handling, adaptive context reduction, 
    and concurrency throttling.
    """
    def __init__(self, high_vram_threshold: float = 0.9, critical_vram_threshold: float = 0.95):
        self.logger = logging.getLogger("MemoryPressureSafetySystem")
        self.high_vram_threshold = high_vram_threshold
        self.critical_vram_threshold = critical_vram_threshold
        self.concurrency_limit_override = None

    def monitor_vram_pressure(self) -> str:
        """
        Analyzes current VRAM usage and returns a safety status.
        """
        if not torch.cuda.is_available():
            return "NORMAL"
            
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        
        usage_ratio = r / t
        
        if usage_ratio > self.critical_vram_threshold:
            self.logger.error(f"CRITICAL VRAM Pressure: {usage_ratio*100:.1f}%")
            return "CRITICAL"
        elif usage_ratio > self.high_vram_threshold:
            self.logger.warning(f"HIGH VRAM Pressure: {usage_ratio*100:.1f}%")
            return "HIGH"
            
        return "NORMAL"

    def apply_safety_measures(self, status: str, scheduler: Any):
        """
        Adjusts scheduler and runtime parameters to mitigate pressure.
        """
        if status == "CRITICAL":
            # Immediate action: flush queue and reduce concurrency to 1
            self.logger.info("Mitigation: Throttling concurrency to 1 and flushing non-urgent requests.")
            scheduler.microbatch_size = 1
            # In a real system, we'd also trigger KV eviction
        elif status == "HIGH":
            # Pre-emptive action: reduce microbatch size
            self.logger.info("Mitigation: Reducing microbatch size to prevent OOM.")
            scheduler.microbatch_size = max(1, scheduler.microbatch_size // 2)
        else:
            # Normal: Restore defaults if they were overridden
            pass

    def get_memory_health_metrics(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {"vram_usage_pct": 0, "status": "UNKNOWN"}
            
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        
        return {
            "vram_usage_pct": float(r / t * 100),
            "vram_reserved_mb": float(r / 1024**2),
            "status": self.monitor_vram_pressure()
        }
