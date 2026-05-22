"""
validation/latency_plausibility_checker.py

Phase 12.5B: Latency Plausibility Checker
Rejects timings that are physically impossible given the constraints of
hardware (e.g., PCIe bandwidth, disk seek times, VRAM latency).
"""

from typing import Dict

class LatencyPlausibilityChecker:
    """
    Validates that reported benchmark latencies align with physical hardware limits.
    """
    
    # Absolute minimum physically possible latencies (conservative estimates)
    MIN_PLAUSIBLE_MS = {
        "vram_read_1mb": 0.05,    # High bandwidth VRAM
        "ram_to_vram_1mb": 0.2,   # PCIe transfer
        "disk_to_ram_1mb": 2.0,   # NVMe read
        "semantic_search_1k": 0.5,# GPU-accelerated similarity search
        "python_overhead": 0.1    # Python function call overhead
    }

    @classmethod
    def check_plausibility(cls, operation: str, measured_ms: float, size_mb: float = 1.0) -> bool:
        """
        Determines if a measured time is physically possible.
        If measured_ms is lower than the absolute theoretical minimum, it's a simulated bypass.
        """
        if operation not in cls.MIN_PLAUSIBLE_MS:
            return True # Unknown operation, can't verify

        min_time = cls.MIN_PLAUSIBLE_MS[operation] * size_mb
        
        if measured_ms < min_time:
            print(f"[WARNING] IMPLAUSIBLE LATENCY: {operation} reported {measured_ms:.4f}ms "
                  f"(Theoretical Min: {min_time:.4f}ms). Likely a simulated bypass.")
            return False
            
        return True
