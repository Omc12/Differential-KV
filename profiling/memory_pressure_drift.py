import time

class MemoryPressureDrift:
    """
    Monitors RAM/VRAM pressure changes over hours.
    Detects if the system is gradually running out of resources.
    """
    def __init__(self):
        self.pressure_log = []

    def log_pressure(self, ram_usage: float, vram_usage: float):
        self.pressure_log.append({
            "timestamp": time.time(),
            "ram": ram_usage,
            "vram": vram_usage
        })

    def analyze_drift(self):
        if len(self.pressure_log) < 2:
            return 0.0
        # Check if vram is increasing over time for same workload
        vram_start = self.pressure_log[0]['vram']
        vram_end = self.pressure_log[-1]['vram']
        return vram_end - vram_start
