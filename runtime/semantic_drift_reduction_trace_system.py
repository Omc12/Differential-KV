"""
STAGE 2 - SDR: Semantic Drift Reduction Trace System
Phase 39.4 - Semantic Drift Reduction

Persists raw telemetry for SDR metrics including repair effectiveness,
dampening events, anchor reinforcement, and reasoning continuity.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any


class SemanticDriftReductionTrace:
    """
    Handles file-based logging of raw SDR events for offline analysis.
    """
    def __init__(self, run_id: str):
        self.trace_dir = Path("traces/stage2/phase_39_4_sdr") / run_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self._files = {
            "repair":      open(self.trace_dir / "repair_effectiveness_trace.jsonl",      "a", encoding="utf-8", buffering=1),
            "reduction":   open(self.trace_dir / "semantic_drift_reduction_trace.jsonl",  "a", encoding="utf-8", buffering=1),
            "continuity":  open(self.trace_dir / "reasoning_continuity_trace.jsonl",       "a", encoding="utf-8", buffering=1),
            "anchor":      open(self.trace_dir / "anchor_reinforcement_trace.jsonl",       "a", encoding="utf-8", buffering=1),
            "oscillation": open(self.trace_dir / "semantic_oscillation_trace.jsonl",      "a", encoding="utf-8", buffering=1),
        }

    def record_repair_event(self, step: int, layer_idx: int, drift_before: float, drift_after: float, effective: bool):
        self._write("repair", {
            "ts": time.time(), "step": step, "layer": layer_idx,
            "drift_before": round(drift_before, 6), "drift_after": round(drift_after, 6),
            "effective": effective
        })

    def record_reduction_metrics(self, step: int, global_drift: float, reduction_rate: float):
        self._write("reduction", {
            "ts": time.time(), "step": step,
            "global_drift": round(global_drift, 6),
            "reduction_rate": round(reduction_rate, 4)
        })

    def record_continuity(self, step: int, chain_len: int, collapsed: bool):
        self._write("continuity", {
            "ts": time.time(), "step": step,
            "chain_len": chain_len, "collapsed": collapsed
        })

    def record_anchor_reinforcement(self, step: int, layer_idx: int, impact: float):
        self._write("anchor", {
            "ts": time.time(), "step": step, "layer": layer_idx,
            "impact": round(impact, 6)
        })

    def record_oscillation(self, step: int, layer_idx: int, interval: int, window_size: int):
        self._write("oscillation", {
            "ts": time.time(), "step": step, "layer": layer_idx,
            "interval": interval, "window_size": window_size
        })

    def _write(self, key: str, data: Dict[str, Any]):
        if key in self._files:
            self._files[key].write(json.dumps(data) + "\n")

    def close(self):
        for f in self._files.values():
            f.close()
