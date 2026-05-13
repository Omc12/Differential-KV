import psutil
from typing import Dict, Any

class HardwarePressureController:
    """
    Manages hardware load and triggers throttling or offloading.
    Monitors CPU, Memory, and GPU pressure (simulated/placeholder).
    """
    def __init__(self, mem_threshold: float = 85.0, cpu_threshold: float = 90.0):
        self.mem_threshold = mem_threshold
        self.cpu_threshold = cpu_threshold

    def get_load_status(self) -> str:
        """Returns the hardware load status."""
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        
        if cpu_usage > self.cpu_threshold or mem_usage > self.mem_threshold:
            return "CRITICAL"
        elif cpu_usage > self.cpu_threshold * 0.8 or mem_usage > self.mem_threshold * 0.8:
            return "HIGH"
        return "NORMAL"

    def get_stats(self) -> Dict[str, float]:
        return {
            "cpu_percent": psutil.cpu_percent(),
            "mem_percent": psutil.virtual_memory().percent
        }
