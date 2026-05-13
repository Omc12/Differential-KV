"""
validation/token_throughput_verifier.py

Verifies token throughput using explicit definitions.
Ensures 'Tokens/sec' are generated tokens, not just internal processing tokens.
"""

from typing import List, Dict, Any
import logging

class TokenThroughputVerifier:
    """
    Audits the definition of 'token' used in throughput metrics.
    """
    def __init__(self):
        self.logger = logging.getLogger("TokenThroughputVerifier")

    def verify_token_count(self, reported_tokens: int, raw_logs: List[Dict[str, Any]]) -> bool:
        """
        Cross-checks reported token count against raw generation logs.
        """
        actual_tokens = sum(l.get('generated_tokens', l.get('tokens', 0)) for l in raw_logs)
        if reported_tokens != actual_tokens:
            self.logger.warning(f"TOKEN COUNT MISMATCH: Reported {reported_tokens}, Actual {actual_tokens}")
            return False
        return True

    def calculate_effective_tps(self, total_tokens: int, duration: float) -> float:
        """
        Calculates ITL (Inter-Token Latency) based TPS.
        """
        if duration <= 0: return 0.0
        return total_tokens / duration
