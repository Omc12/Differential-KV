"""
memory/context_pressure_controller.py

Throttles or optimizes sparse retrieval based on total context pressure.
Prevents system-wide OOM or latency collapse at extreme scales.
"""

from typing import Dict, Any
import logging

class ContextPressureController:
    """
    Adaptive controller for long-context memory stability.
    """
    def __init__(self, max_context_tokens: int = 1048576): # 1M default
        self.max_tokens = max_context_tokens
        self.current_tokens = 0
        self.logger = logging.getLogger("ContextPressureController")

    def report_usage(self, token_count: int):
        """Updates the controller with current token usage."""
        self.current_tokens = token_count
        pressure = self.get_pressure_level()
        
        if pressure > 0.9:
            self.logger.warning(f"EXTREME PRESSURE: {self.current_tokens} tokens ({pressure*100:.1f}%)")
        elif pressure > 0.75:
            self.logger.info(f"HIGH PRESSURE: {self.current_tokens} tokens ({pressure*100:.1f}%)")

    def get_pressure_level(self) -> float:
        """Returns normalized pressure level [0, 1]."""
        return min(1.0, self.current_tokens / self.max_tokens)

    def get_sparse_budget_multiplier(self) -> float:
        """
        Returns a multiplier to reduce sparse density as pressure increases.
        If pressure > 0.8, start thinning out anchors.
        """
        pressure = self.get_pressure_level()
        if pressure < 0.5:
            return 1.0
        
        # Linear decay from 1.0 to 0.2
        return max(0.2, 1.0 - (pressure - 0.5) * 1.6)

    def should_allow_new_context(self, tokens: int) -> bool:
        """Checks if a new request will fit within the stability budget."""
        return (self.current_tokens + tokens) <= self.max_tokens
