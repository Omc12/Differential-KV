"""
STAGE 2 - ASS: Semantic Stability Forecast Trace
Phase 39.5 - Adaptive Semantic Scheduling

Persists RAW traces for predictive metrics and proactive interventions.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

class SemanticStabilityForecastTrace:
    def __init__(self, run_id: str):
        self.trace_dir = Path("traces/stage2/phase_39_5_ass") / run_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self._files = {
            "forecast":    open(self.trace_dir / "semantic_forecast_trace.jsonl",    "a", encoding="utf-8", buffering=1),
            "proactive":   open(self.trace_dir / "proactive_recovery_trace.jsonl",   "a", encoding="utf-8", buffering=1),
            "accuracy":    open(self.trace_dir / "forecast_accuracy_trace.jsonl",    "a", encoding="utf-8", buffering=1),
            "equilibrium": open(self.trace_dir / "semantic_equilibrium_trace.jsonl", "a", encoding="utf-8", buffering=1),
            "anchor":      open(self.trace_dir / "predictive_anchor_trace.jsonl",    "a", encoding="utf-8", buffering=1),
        }

    def record_forecast(self, step: int, layer_idx: int, pressure: float):
        self._write("forecast", {"ts": time.time(), "step": step, "layer": layer_idx, "pressure": round(pressure, 4)})

    def record_proactive_recovery(self, step: int, layer_idx: int):
        self._write("proactive", {"ts": time.time(), "step": step, "layer": layer_idx})

    def record_accuracy(self, step: int, accuracy: float, false_positives: int, missed: int, avoided: int):
        self._write("accuracy", {
            "ts": time.time(), "step": step, 
            "accuracy": round(accuracy, 4), "false_positives": false_positives,
            "missed_events": missed, "avoided_events": avoided
        })

    def record_equilibrium(self, step: int, score: float, in_fallback: bool):
        self._write("equilibrium", {"ts": time.time(), "step": step, "score": round(score, 4), "in_fallback": in_fallback})

    def record_predictive_anchor(self, step: int, layer_idx: int, half_life: float):
        self._write("anchor", {"ts": time.time(), "step": step, "layer": layer_idx, "half_life": round(half_life, 2)})

    def _write(self, key: str, data: Dict[str, Any]):
        if key in self._files:
            self._files[key].write(json.dumps(data) + "\n")

    def close(self):
        for f in self._files.values():
            f.close()
