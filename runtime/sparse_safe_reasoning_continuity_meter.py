"""
STAGE 2 - SDR: Sparse-Safe Reasoning Continuity Meter
Phase 39.4 - Semantic Drift Reduction

Measures the continuity of reasoning chains during sparse execution.
Detects 'reasoning collapse' when semantic drift disrupts logical token dependencies.
"""
import threading
from typing import Dict, Any, List


class SparseSafeReasoningContinuityMeter:
    """
    Tracks how many tokens are generated in a row without a 'semantic shock'
    (massive drift spike or recovery failure).
    """
    SHOCK_THRESHOLD = 0.25  # Instantaneous drift spike that breaks continuity

    def __init__(self):
        self._lock = threading.RLock()
        self._current_chain_len = 0
        self._max_chain_len = 0
        self._total_chains = 0
        self._collapse_count = 0
        self._chain_lengths: List[int] = []

    def record_step(self, max_layer_drift: float, recovery_failed: bool):
        """Processes a single token generation step."""
        with self._lock:
            if max_layer_drift >= self.SHOCK_THRESHOLD or recovery_failed:
                # Continuity broken
                if self._current_chain_len > 0:
                    self._chain_lengths.append(self._current_chain_len)
                    self._max_chain_len = max(self._max_chain_len, self._current_chain_len)
                    self._total_chains += 1
                self._current_chain_len = 0
                self._collapse_count += 1
            else:
                self._current_chain_len += 1

    def get_continuity_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_chain = sum(self._chain_lengths) / max(self._total_chains, 1)
            return {
                "max_reasoning_continuity_chain": self._max_chain_len,
                "avg_reasoning_continuity_chain": round(avg_chain, 2),
                "reasoning_collapse_count": self._collapse_count,
                "current_chain": self._current_chain_len
            }
