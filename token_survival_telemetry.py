from typing import Dict, Any

class TokenSurvivalTelemetry:
    """
    Tracks real surviving tokens and compute savings for ATC.
    """
    def __init__(self):
        self.total_tokens_seen = 0
        self.active_tokens_seen = 0
        self.steps = 0

    def record_step(self, total: int, active: int):
        self.total_tokens_seen += total
        self.active_tokens_seen += active
        self.steps += 1

    def get_metrics(self) -> Dict[str, float]:
        if self.total_tokens_seen == 0:
            return {}
            
        ratio = self.active_tokens_seen / self.total_tokens_seen
        return {
            "active_token_ratio": ratio,
            "collapsed_token_ratio": 1.0 - ratio,
            "token_compute_reduction": (1.0 - ratio) * 100,
            "effective_sequence_length": self.active_tokens_seen / self.steps if self.steps > 0 else 0
        }

# Global singleton
atc_telemetry = TokenSurvivalTelemetry()
