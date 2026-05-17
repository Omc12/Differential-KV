"""
STAGE 3D.0 — RPI (REAL PRODUCTION INSTRUMENTATION)
runtime/native_trace_authenticity_auditor.py

Validates that persisted traces are physically real, and fail if any synthetic markers, 
impossible stabilities, or empty profiles are detected.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

class NativeTraceAuthenticityAuditor:
    """
    Hard scientific gate auditing the reality of persisted traces.
    Detects:
    - placeholder traces (e.g. constant strings, empty elements)
    - synthetic timestamps (perfectly uniform floating point steps)
    - constant intervals (zero variance in poll cycles)
    - impossible stability (std dev of dynamic metrics near 0)
    - repeated telemetry windows (duplicated sequence of metrics)
    - empty profiler captures (traceEvents == [])
    - fake queue turbulence (perfectly static queue depth)
    - fake latency distributions (perfectly uniform or flat arrays)
    """
    def __init__(self):
        self.logger = logging.getLogger("RPI_AuthenticityAuditor")

    def audit_traces(self, trace_dir: Path, telemetry_dir: Path) -> Dict[str, Any]:
        """
        Executes structural authenticity tests across all production traces.
        Returns a dictionary detailing pass/fail status and metric scores.
        """
        self.logger.info("Executing Native Trace Authenticity Audit...")
        
        results = {
            "passed": True,
            "violations": [],
            "metrics": {}
        }
        
        # 1. Audit CUDA Profiler Trace (No empty captures allowed)
        profiler_path = trace_dir / "cuda_profiler_trace.json"
        raw_profiler_path = telemetry_dir / "raw_torch_profiler_trace.json"
        
        for p in [profiler_path, raw_profiler_path]:
            if not p.exists():
                results["passed"] = False
                results["violations"].append(f"Missing profiler file: {p.name}")
                continue
                
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    events = data.get("traceEvents", [])
                    if not events:
                        results["passed"] = False
                        results["violations"].append(f"Profiler capture {p.name} has EMPTY traceEvents!")
            except Exception as e:
                results["passed"] = False
                results["violations"].append(f"Failed to parse profiler {p.name}: {e}")

        # 2. Audit NVML Telemetry for Constant Intervals and Impossible Stability
        nvml_path = trace_dir / "nvml_telemetry_trace.jsonl"
        if nvml_path.exists():
            records = []
            with open(nvml_path, "r", encoding="utf-8") as f:
                for line in f:
                    try: records.append(json.loads(line))
                    except: pass
            
            if len(records) > 5:
                timestamps = np.array([r["timestamp"] for r in records])
                power = np.array([r.get("gpu_power_watts", 0.0) for r in records])
                temp = np.array([r.get("gpu_temp_c", 0.0) for r in records])
                sm = np.array([r.get("sm_utilization_pct", 0.0) for r in records])
                
                # Check for constant intervals (synthetic timing check)
                deltas = np.diff(timestamps)
                delta_std = np.std(deltas)
                results["metrics"]["polling_interval_variance"] = float(delta_std)
                
                # Physical systems have timing jitter (sampling drift). Standard deviation of delta should be > 1e-6.
                # If delta_std is exactly 0.0 or under 1e-7, it's synthetic timing!
                if delta_std < 1e-7:
                    results["passed"] = False
                    results["violations"].append("SYNTHETIC_TIMING_DETECTED: Telemetry timestamps are perfectly periodic with zero jitter!")
                
                # Check for impossible stability (synthetic flatlining)
                std_power = np.std(power)
                std_temp = np.std(temp)
                std_sm = np.std(sm)
                
                results["metrics"]["power_std"] = float(std_power)
                results["metrics"]["temp_std"] = float(std_temp)
                results["metrics"]["sm_std"] = float(std_sm)
                
                if std_power < 0.01:
                    results["passed"] = False
                    results["violations"].append("IMPOSSIBLE_STABILITY_DETECTED: GPU power draw standard deviation is below threshold (< 0.01W).")
                if std_temp < 0.01:
                    results["passed"] = False
                    results["violations"].append("IMPOSSIBLE_STABILITY_DETECTED: GPU temperature standard deviation is below threshold (< 0.01 C).")
                if std_sm < 0.01:
                    results["passed"] = False
                    results["violations"].append("IMPOSSIBLE_STABILITY_DETECTED: SM utilization standard deviation is below threshold (< 0.01%).")
                    
                # Check repeated telemetry windows (detect mock replay loops)
                # Compare first half vs second half if there are enough records
                if len(records) >= 20:
                    half = len(records) // 2
                    first_half_sm = sm[:half]
                    second_half_sm = sm[half:2*half]
                    if np.array_equal(first_half_sm, second_half_sm):
                        results["passed"] = False
                        results["violations"].append("REPEATED_TELEMETRY_WINDOW_DETECTED: Telemetry loop replayed exactly!")
            else:
                results["passed"] = False
                results["violations"].append("NVML telemetry trace contains insufficient records.")
        else:
            results["passed"] = False
            results["violations"].append("Missing nvml_telemetry_trace.jsonl")

        # 3. Audit Latency Flatness
        latency_path = trace_dir / "token_latency_trace.jsonl"
        if latency_path.exists():
            latencies = []
            jitters = []
            queue_depths = []
            
            with open(latency_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        latencies.append(rec.get("latency_ms", 0.0))
                        jitters.append(rec.get("jitter_ms", 0.0))
                        queue_depths.append(rec.get("queue_wait_ms", 0.0))
                    except: pass
            
            if len(latencies) > 5:
                std_lat = np.std(latencies)
                std_jitter = np.std(jitters)
                std_q = np.std(queue_depths)
                
                results["metrics"]["latency_std"] = float(std_lat)
                results["metrics"]["jitter_std"] = float(std_jitter)
                results["metrics"]["queue_wait_std"] = float(std_q)
                
                if std_lat < 0.01:
                    results["passed"] = False
                    results["violations"].append("LATENCY_FLATNESS_VIOLATION: Token latencies are unnaturally flat (std < 0.01ms)!")
                if std_jitter < 0.001:
                    results["passed"] = False
                    results["violations"].append("LATENCY_FLATNESS_VIOLATION: Jitter is unnaturally flat (std < 0.001ms)!")
                if std_q < 0.01:
                    results["passed"] = False
                    results["violations"].append("FAKE_QUEUE_TURBULENCE_VIOLATION: Queue delays lack natural variance (std < 0.01ms)!")
            else:
                results["passed"] = False
                results["violations"].append("Token latency trace contains insufficient records.")
        else:
            results["passed"] = False
            results["violations"].append("Missing token_latency_trace.jsonl")

        if results["passed"]:
            self.logger.info("Trace Authenticity Audit: SUCCESS — All traces verified physically real.")
        else:
            self.logger.error(f"Trace Authenticity Audit: FAILED! Violations: {results['violations']}")
            
        return results
