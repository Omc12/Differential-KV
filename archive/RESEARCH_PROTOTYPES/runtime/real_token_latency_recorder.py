"""
STAGE 3D.0 — RPI (REAL PRODUCTION INSTRUMENTATION)
runtime/real_token_latency_recorder.py

Measures TRUE, unsmoothed token generation latencies on a token-by-token level.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

class RealTokenLatencyRecorder:
    """
    Measures true token generation latencies without smoothing, clipping, or compression.
    Tracks decode timestamps, inter-token latency, queue waiting time, and true jitter.
    """
    def __init__(self, trace_dir: str):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("RPI_TokenLatency")
        
        self.trace_path = self.trace_dir / "token_latency_trace.jsonl"
        
        # In-memory arrays for auditing and correlation
        self.session_timestamps: Dict[str, List[float]] = {}
        self.session_latencies: Dict[str, List[float]] = {}
        self.session_queue_waits: Dict[str, List[float]] = {}
        self.session_jitters: Dict[str, List[float]] = {}
        
        self.all_raw_latencies: List[float] = []
        self.all_raw_jitters: List[float] = []

    def record_token(self, session_id: str, token_index: int, latency_ms: float, queue_wait_ms: float):
        """
        Records a single token generation event.
        Calculates inter-token latency and true jitter.
        """
        now_ts = time.time()
        
        if session_id not in self.session_timestamps:
            self.session_timestamps[session_id] = []
            self.session_latencies[session_id] = []
            self.session_queue_waits[session_id] = []
            self.session_jitters[session_id] = []
            
        self.session_timestamps[session_id].append(now_ts)
        self.session_latencies[session_id].append(latency_ms)
        self.session_queue_waits[session_id].append(queue_wait_ms)
        
        self.all_raw_latencies.append(latency_ms)
        
        # Calculate true inter-token latency and jitter
        inter_token_ms = 0.0
        jitter_ms = 0.0
        
        if len(self.session_timestamps[session_id]) > 1:
            inter_token_ms = (self.session_timestamps[session_id][-1] - self.session_timestamps[session_id][-2]) * 1000.0
            
            # True jitter is absolute difference in consecutive inter-token latencies
            if len(self.session_latencies[session_id]) > 2:
                prev_inter_token = (self.session_timestamps[session_id][-2] - self.session_timestamps[session_id][-3]) * 1000.0
                jitter_ms = abs(inter_token_ms - prev_inter_token)
                self.session_jitters[session_id].append(jitter_ms)
                self.all_raw_jitters.append(jitter_ms)
                
        # Persist physically-derived token event to trace
        record = {
            "timestamp": now_ts,
            "session_id": session_id,
            "token_index": token_index,
            "latency_ms": latency_ms,
            "queue_wait_ms": queue_wait_ms,
            "inter_token_latency_ms": inter_token_ms,
            "jitter_ms": jitter_ms
        }
        self._persist_line(self.trace_path, record)

    def _persist_line(self, filepath: Path, data: Dict[str, Any]):
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            self.logger.debug(f"Failed to write to {filepath.name}: {e}")

    def check_flatness_violation(self) -> bool:
        """
        Validation check: If token latencies are too uniform (e.g. constant/synthetic delay),
        this will return True to fail validation.
        """
        if len(self.all_raw_latencies) < 5:
            return False  # Not enough data to judge
            
        std_val = np.std(self.all_raw_latencies)
        # Unnaturally flat latencies indicate synthetic shaping / simulated delay
        if std_val < 0.01:
            self.logger.error(f"FLATNESS_VIOLATION detected! Token latency standard deviation: {std_val:.4f}ms. Latencies are unnaturally flat.")
            return True
        return False

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Exposes raw unsmoothed lists for correlator and auditor."""
        return {
            "raw_latencies": self.all_raw_latencies,
            "raw_jitters": self.all_raw_jitters,
            "avg_latency_ms": np.mean(self.all_raw_latencies) if self.all_raw_latencies else 0.0,
            "std_latency_ms": np.std(self.all_raw_latencies) if self.all_raw_latencies else 0.0,
            "max_latency_ms": np.max(self.all_raw_latencies) if self.all_raw_latencies else 0.0,
            "avg_jitter_ms": np.mean(self.all_raw_jitters) if self.all_raw_jitters else 0.0
        }
