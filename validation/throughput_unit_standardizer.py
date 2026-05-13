"""
validation/throughput_unit_standardizer.py

Standardizes units across all metrics (ops/sec, tokens/sec, etc).
Prevents apples-to-oranges comparisons in throughput reporting.
"""

from typing import Dict, Any
import logging

class ThroughputUnitStandardizer:
    """
    Normalizes metrics to a canonical set of units.
    """
    def __init__(self):
        self.logger = logging.getLogger("ThroughputUnitStandardizer")

    def standardize(self, name: str, value: float) -> tuple[float, str]:
        """
        Returns (standardized_value, unit_string).
        """
        name_lower = name.lower()
        
        if "tps" in name_lower:
            return value, "tokens/sec"
        if "latency" in name_lower or "time" in name_lower:
            if value > 1000: # Assuming it might be in microseconds
                self.logger.info(f"Normalizing {name} from us to ms")
                return value / 1000.0, "ms"
            return value, "ms"
        if "bandwidth" in name_lower or "traffic" in name_lower:
            return value, "GB/s"
            
        return value, "units/sec"
