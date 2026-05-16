"""
STAGE 2 - RBT: Rigorous Benchmark Trace System
Phase 39.9 - Rigorous Benchmark Triangulation

Persists raw traces for rigorous benchmark evaluation and failure mapping.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

class RigorousBenchmarkTraceSystem:
    def __init__(self, run_id: str):
        self.trace_dir = Path("traces/stage2/phase_39_9_rbt") / run_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self._files = {
            "boundary":    open(self.trace_dir / "failure_boundary_trace.jsonl",      "a", encoding="utf-8", buffering=1),
            "taxonomy":    open(self.trace_dir / "failure_taxonomy_trace.jsonl",      "a", encoding="utf-8", buffering=1),
            "horizon":     open(self.trace_dir / "long_horizon_trace.jsonl",          "a", encoding="utf-8", buffering=1),
            "domain":      open(self.trace_dir / "domain_fidelity_trace.jsonl",       "a", encoding="utf-8", buffering=1),
            "uncertainty": open(self.trace_dir / "benchmark_uncertainty_trace.jsonl", "a", encoding="utf-8", buffering=1),
        }

    def record_boundary(self, step: int, limits: Dict[str, Any]):
        data = {"ts": time.time(), "step": step}
        data.update(limits)
        self._write("boundary", data)

    def record_taxonomy(self, step: int, taxonomy: Dict[str, int]):
        data = {"ts": time.time(), "step": step}
        data.update(taxonomy)
        self._write("taxonomy", data)

    def record_horizon(self, step: int, horizon: int):
        self._write("horizon", {"ts": time.time(), "step": step, "max_stable_horizon_steps": horizon})

    def record_domain(self, step: int, fidelities: Dict[str, float]):
        data = {"ts": time.time(), "step": step}
        data.update(fidelities)
        self._write("domain", data)

    def record_uncertainty(self, step: int, conf: float, unsupp: int):
        self._write("uncertainty", {"ts": time.time(), "step": step, "confidence_score": round(conf, 4), "unsupported_regions_count": unsupp})

    def _write(self, key: str, data: Dict[str, Any]):
        if key in self._files:
            self._files[key].write(json.dumps(data) + "\n")

    def close(self):
        for f in self._files.values():
            f.close()
