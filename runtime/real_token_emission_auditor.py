import time
from typing import Dict, Any, List

class RealTokenEmissionAuditor:
    """
    Stage 4B.1.5 RTA: Real Token Emission Auditor.
    Tracks and counts only actual user-visible generated tokens emitted by the model,
    excluding prompts, speculative discarded tokens, and scheduler loop cycles.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.emitted_tokens = 0
        self.token_timestamps = []
        self.start_time = None
        self.end_time = None
        self.cadences = []

    def record_start(self):
        self.start_time = time.time()
        self.token_timestamps = []
        self.emitted_tokens = 0

    def record_token(self, token_text: str):
        """
        Record a single real token emission from tokenizer/generation loop.
        """
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        self.emitted_tokens += 1
        self.token_timestamps.append(now)
        
        if len(self.token_timestamps) > 1:
            interval = self.token_timestamps[-1] - self.token_timestamps[-2]
            self.cadences.append(interval)

    def record_end(self):
        self.end_time = time.time()

    def get_real_tps(self) -> float:
        """
        Returns true emitted tokens per second.
        """
        if self.start_time is None or self.emitted_tokens == 0:
            return 0.0
        end = self.end_time or time.time()
        duration = end - self.start_time
        if duration <= 0:
            return 0.0
        return float(self.emitted_tokens) / float(duration)

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns reality-verified token statistics.
        """
        real_tps = self.get_real_tps()
        avg_cadence_ms = (sum(self.cadences) / len(self.cadences)) * 1000.0 if self.cadences else 0.0
        return {
            "real_emitted_tokens": self.emitted_tokens,
            "real_tps": real_tps,
            "average_cadence_ms": avg_cadence_ms,
            "total_generation_seconds": (self.end_time or time.time()) - (self.start_time or time.time())
        }
