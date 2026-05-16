"""
STAGE 2 - ARS: Adversarial Semantic Trace System
Phase 39.8 - Adversarial Reasoning Stability

Persists raw traces for adversarial reasoning evaluation.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

class AdversarialSemanticTraceSystem:
    def __init__(self, run_id: str):
        self.trace_dir = Path("traces/stage2/phase_39_8_ars") / run_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self._files = {
            "collapse":       open(self.trace_dir / "reasoning_collapse_trace.jsonl",      "a", encoding="utf-8", buffering=1),
            "contradiction":  open(self.trace_dir / "contradiction_trace.jsonl",           "a", encoding="utf-8", buffering=1),
            "multihop":       open(self.trace_dir / "multihop_trace.jsonl",                "a", encoding="utf-8", buffering=1),
            "delayed":        open(self.trace_dir / "delayed_dependency_trace.jsonl",      "a", encoding="utf-8", buffering=1),
            "perturbation":   open(self.trace_dir / "perturbation_robustness_trace.jsonl", "a", encoding="utf-8", buffering=1),
        }

    def record_collapse(self, step: int, collapse_events: int, rate: float):
        self._write("collapse", {"ts": time.time(), "step": step, "collapse_events": collapse_events, "collapse_rate": round(rate, 4)})

    def record_contradiction(self, step: int, rate: float):
        self._write("contradiction", {"ts": time.time(), "step": step, "contradiction_rate": round(rate, 4)})

    def record_multihop(self, step: int, stability: float):
        self._write("multihop", {"ts": time.time(), "step": step, "multihop_stability": round(stability, 4)})

    def record_delayed(self, step: int, fidelity: float):
        self._write("delayed", {"ts": time.time(), "step": step, "delayed_recall_fidelity": round(fidelity, 4)})

    def record_perturbation(self, step: int, robustness: float):
        self._write("perturbation", {"ts": time.time(), "step": step, "perturbation_robustness": round(robustness, 4)})

    def _write(self, key: str, data: Dict[str, Any]):
        if key in self._files:
            self._files[key].write(json.dumps(data) + "\n")

    def close(self):
        for f in self._files.values():
            f.close()
