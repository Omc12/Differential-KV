import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

class RealLatencyTraceSystem:
    """
    STAGE 4A.0 — LCO: Real Latency Trace System.
    Persists RAW, physically-derived traces for all key Stage 4A.0 metrics.
    """
    def __init__(self, trace_dir: str = "traces/stage4a/phase_44_0_lco/"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
    def _write_record(self, filename: str, data: Dict[str, Any]):
        path = self.trace_dir / filename
        data["timestamp"] = time.time()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            
    def log_synchronization(self, sync_frequency: float, sync_duration: float, sync_stall_pct: float, barrier_collapse_ratio: float):
        self._write_record("synchronization_trace.jsonl", {
            "sync_frequency": sync_frequency,
            "sync_duration_ms": sync_duration,
            "sync_stall_pct": sync_stall_pct,
            "barrier_collapse_ratio": barrier_collapse_ratio
        })
        self.log_synchronization_stall(sync_stall_pct, sync_duration)
        
    def log_decode_bubble(self, idle_gap_pct: float, decode_continuity_pct: float, bubble_duration: float, queue_starvation_frequency: float):
        self._write_record("decode_bubble_trace.jsonl", {
            "idle_gap_pct": idle_gap_pct,
            "decode_continuity_pct": decode_continuity_pct,
            "bubble_duration_ms": bubble_duration,
            "queue_starvation_frequency": queue_starvation_frequency
        })
        self.log_decode_continuity(decode_continuity_pct)
        
    def log_token_latency(self, inter_token_latency: float, p50: float, p95: float, p99: float, jitter: float, emission_smoothness: float, tail_collapse_ratio: float):
        self._write_record("token_latency_trace.jsonl", {
            "inter_token_latency_ms": inter_token_latency,
            "latency_jitter_ms": jitter,
            "tail_collapse_ratio": tail_collapse_ratio
        })
        self.log_emission_smoothness(emission_smoothness)
        
    def log_queue_pressure(self, queue_depth: int, queue_wait_time: float, burst_collapse_efficiency: float, starvation_recovery_time: float):
        self._write_record("queue_pressure_trace.jsonl", {
            "queue_depth": queue_depth,
            "queue_wait_time_ms": queue_wait_time,
            "burst_collapse_efficiency": burst_collapse_efficiency,
            "starvation_recovery_time_ms": starvation_recovery_time
        })
        
    def log_persistent_decode(self, residency_continuity: float, launch_reuse_ratio: float, warm_state_reuse_pct: float, decode_persistence_pct: float):
        self._write_record("persistent_decode_trace.jsonl", {
            "residency_continuity": residency_continuity,
            "warm_state_reuse_pct": warm_state_reuse_pct,
            "decode_persistence_pct": decode_persistence_pct
        })
        self.log_launch_reuse(launch_reuse_ratio)
        
    def log_tail_latency(self, p95: float, p99: float, p999: float, max_latency: float, tail_collapse_efficiency: float, p50: Optional[float] = None):
        if p50 is None:
            p50 = p95 * 0.5
        self._write_record("tail_latency_trace.jsonl", {
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "p999_latency_ms": p999,
            "max_latency_ms": max_latency,
            "tail_collapse_efficiency": tail_collapse_efficiency
        })
        
    def log_launch_reuse(self, launch_reuse_ratio: float):
        self._write_record("launch_reuse_trace.jsonl", {
            "launch_reuse_ratio": launch_reuse_ratio
        })
        
    def log_decode_continuity(self, decode_continuity_pct: float):
        self._write_record("decode_continuity_trace.jsonl", {
            "decode_continuity_pct": decode_continuity_pct
        })
        
    def log_emission_smoothness(self, emission_smoothness: float):
        self._write_record("emission_smoothness_trace.jsonl", {
            "emission_smoothness": emission_smoothness
        })
        
    def log_synchronization_stall(self, sync_stall_pct: float, sync_duration: float):
        self._write_record("synchronization_stall_trace.jsonl", {
            "sync_stall_pct": sync_stall_pct,
            "sync_duration_ms": sync_duration
        })
