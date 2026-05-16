import time
import torch
from typing import Dict, List, Any

class RealEndToEndProfiler:
    """
    EOM MODULE 5: Tracks serving-aware performance metrics.
    Identifies true serving bottlenecks beyond kernel throughput.
    """
    def __init__(self):
        self.timings = {
            "decode_stage": [],
            "prefill": [],
            "fused_step": [],
            "sync_overhead": [],
            "launch_overhead": [],
            "queue_wait": [],
            "streaming": [],
            "serialization": []
        }
        self.start_times = {}

    def start_segment(self, session_id: str, segment: str):
        self.start_times[(session_id, segment)] = time.perf_counter()

    def end_segment(self, session_id: str, segment: str):
        key = (session_id, segment)
        if key in self.start_times:
            duration = (time.perf_counter() - self.start_times[key]) * 1000
            self.timings[segment].append(duration)
            del self.start_times[key]

    def get_profile_report(self) -> Dict[str, float]:
        report = {}
        for segment, values in self.timings.items():
            if values:
                report[f"avg_{segment}_ms"] = sum(values) / len(values)
                report[f"p95_{segment}_ms"] = sorted(values)[int(len(values) * 0.95)]
            else:
                report[f"avg_{segment}_ms"] = 0.0
        return report

    def reset(self):
        for k in self.timings:
            self.timings[k] = []
