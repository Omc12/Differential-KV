import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Any

class RealTailLatencyDistributionSystem:
    """
    RTS Stage 3C.5: Real Tail Latency Distribution System.
    Captures exact tail latencies across the serving horizon, persisting raw distributions
    and reporting p50, p90, p95, p99, p99.9, max latency, and successive step jitter.
    """
    def __init__(self, trace_dir: str = "traces/stage3c/phase_42_5_rts/"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.raw_latencies: List[float] = []
        self.step_durations: List[float] = []
        self.jitters: List[float] = []
        self.last_latency = 0.0

    def record_step_latency(self, latency_ms: float):
        """
        Record the raw latency of an individual decode step.
        No filters or percentile clipping.
        """
        self.raw_latencies.append(latency_ms)
        if len(self.raw_latencies) > 1:
            # Latency jitter measured as consecutive difference magnitude
            jitter = abs(latency_ms - self.last_latency)
            self.jitters.append(jitter)
        self.last_latency = latency_ms

    def compute_percentiles(self) -> Dict[str, float]:
        """
        Computes accurate percentiles including p99.9 and max.
        """
        if not self.raw_latencies:
            return {
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "p99.9": 0.0,
                "max": 0.0,
                "jitter": 0.0
            }
        
        arr = np.array(self.raw_latencies)
        jitters_arr = np.array(self.jitters) if self.jitters else np.array([0.0])
        
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "p99.9": float(np.percentile(arr, 99.9)),
            "max": float(np.max(arr)),
            "jitter": float(np.mean(jitters_arr))
        }

    def persist_trace(self, step: int):
        """
        Persists the current distribution stats to disk.
        """
        metrics = self.compute_percentiles()
        trace_record = {
            "timestamp": time.time(),
            "decode_step": step,
            "sample_count": len(self.raw_latencies),
            **metrics
        }
        
        trace_file = self.trace_dir / "latency_distribution_trace.jsonl"
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record) + "\n")
            
        # Also persist full raw distribution once at completion
        raw_file = self.trace_dir / "raw_latencies_distribution.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump({
                "raw_latencies": self.raw_latencies,
                "jitters": self.jitters
            }, f, indent=2)
