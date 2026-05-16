"""
STAGE 2.5 - SRC: Scientific Trace Consolidation System
Phase 40.0 - Scientific Research Consolidation

Unifies all scientific outputs into a reproducible audit structure.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

class ScientificTraceConsolidationSystem:
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self._files = {
            "ablation":     open(self.trace_dir / "ablation_trace.jsonl",             "a", encoding="utf-8", buffering=1),
            "tradeoff":     open(self.trace_dir / "tradeoff_curve_trace.jsonl",       "a", encoding="utf-8", buffering=1),
            "degradation":  open(self.trace_dir / "degradation_curve_trace.jsonl",    "a", encoding="utf-8", buffering=1),
            "reproducibility": open(self.trace_dir / "reproducibility_trace.jsonl",   "a", encoding="utf-8", buffering=1),
            "envelope":     open(self.trace_dir / "operational_envelope_trace.jsonl", "a", encoding="utf-8", buffering=1),
        }

    def unify_previous_phases(self):
        """Copies key traces from previous phases into the current audit structure."""
        import shutil
        
        # Mapping of previous phase paths to current audit structure
        previous_sources = [
            Path("traces/stage2/phase_39_8_ars"),
            Path("traces/stage2/phase_39_9_rbt"),
        ]
        
        for src_root in previous_sources:
            if src_root.exists():
                # Find the latest run in that phase
                runs = sorted([d for d in src_root.iterdir() if d.is_dir()])
                if runs:
                    latest_run = runs[-1]
                    for f in latest_run.glob("*.jsonl"):
                        target_name = f"{src_root.name}_{f.name}"
                        shutil.copy2(f, self.trace_dir / target_name)

    def record_ablation(self, step: int, metrics: Dict[str, Any]):
        self._write("ablation", {"ts": time.time(), "step": step, **metrics})

    def record_tradeoff(self, step: int, metrics: Dict[str, Any]):
        self._write("tradeoff", {"ts": time.time(), "step": step, **metrics})

    def record_degradation(self, step: int, metrics: Dict[str, Any]):
        self._write("degradation", {"ts": time.time(), "step": step, **metrics})

    def record_reproducibility(self, step: int, metrics: Dict[str, Any]):
        self._write("reproducibility", {"ts": time.time(), "step": step, **metrics})

    def record_envelope(self, step: int, metrics: Dict[str, Any]):
        self._write("envelope", {"ts": time.time(), "step": step, **metrics})

    def _write(self, key: str, data: Dict[str, Any]):
        if key in self._files:
            self._files[key].write(json.dumps(data) + "\n")

    def close(self):
        for f in self._files.values():
            f.close()
