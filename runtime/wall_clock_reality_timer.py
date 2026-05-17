import time
import numpy as np
from typing import Dict, Any, List

class WallClockRealityTimer:
    """
    Stage 4B.1.5 RTA: Wall Clock Reality Timer.
    Strictly measures true wall-clock time elapsed for inference generation tasks.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.inter_token_latencies = []

    def start_timer(self):
        self.reset()
        self.start_time = time.monotonic()

    def record_first_token(self):
        self.first_token_time = time.monotonic()

    def record_token_step(self):
        now = time.monotonic()
        if self.first_token_time is None:
            self.first_token_time = now
            
        ref = self.inter_token_latencies[-1][0] if self.inter_token_latencies else self.first_token_time
        latency_sec = now - ref
        self.inter_token_latencies.append((now, latency_sec))

    def stop_timer(self):
        self.end_time = time.monotonic()

    def get_ttft_ms(self) -> float:
        if self.start_time is None or self.first_token_time is None:
            return 0.0
        return (self.first_token_time - self.start_time) * 1000.0

    def get_total_duration_sec(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.monotonic()
        return end - self.start_time

    def get_stream_duration_sec(self) -> float:
        if self.first_token_time is None:
            return 0.0
        end = self.end_time or time.monotonic()
        return end - self.first_token_time

    def get_percentiles_ms(self) -> Dict[str, float]:
        if not self.inter_token_latencies:
            return {"p50": 0.0, "p95": 0.0}
        latencies_ms = [lat * 1000.0 for _, lat in self.inter_token_latencies]
        return {
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95))
        }

    def get_telemetry(self) -> Dict[str, Any]:
        pcts = self.get_percentiles_ms()
        return {
            "ttft_ms": self.get_ttft_ms(),
            "total_duration_sec": self.get_total_duration_sec(),
            "stream_duration_sec": self.get_stream_duration_sec(),
            "p50_token_latency_ms": pcts["p50"],
            "p95_token_latency_ms": pcts["p95"]
        }
