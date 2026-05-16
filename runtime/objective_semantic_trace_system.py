"""
STAGE 2 - OSE: Objective Semantic Trace System
Phase 39.7 - Objective Semantic Evaluation

Persists raw traces for objective reasoning evaluation.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

class ObjectiveSemanticTraceSystem:
    def __init__(self, run_id: str):
        self.trace_dir = Path("traces/stage2/phase_39_7_ose") / run_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self._files = {
            "reasoning":   open(self.trace_dir / "objective_reasoning_trace.jsonl", "a", encoding="utf-8", buffering=1),
            "divergence":  open(self.trace_dir / "semantic_divergence_trace.jsonl", "a", encoding="utf-8", buffering=1),
            "hallucination": open(self.trace_dir / "hallucination_trace.jsonl",       "a", encoding="utf-8", buffering=1),
            "recall":      open(self.trace_dir / "long_context_recall_trace.jsonl", "a", encoding="utf-8", buffering=1),
            "fidelity":    open(self.trace_dir / "fidelity_trace.jsonl",            "a", encoding="utf-8", buffering=1),
        }

    def record_reasoning(self, step: int, dense_acc: float, sparse_acc: float, agreement: float):
        self._write("reasoning", {"ts": time.time(), "step": step, "dense_acc": round(dense_acc, 4), "sparse_acc": round(sparse_acc, 4), "agreement": round(agreement, 4)})

    def record_divergence(self, step: int, kl_div: float, is_exact: float):
        self._write("divergence", {"ts": time.time(), "step": step, "kl_divergence": round(kl_div, 4), "is_exact": round(is_exact, 4)})

    def record_hallucination(self, step: int, hallucination_events: int):
        self._write("hallucination", {"ts": time.time(), "step": step, "hallucination_events": hallucination_events})

    def record_recall(self, step: int, recall_fidelity: float):
        self._write("recall", {"ts": time.time(), "step": step, "recall_fidelity": round(recall_fidelity, 4)})

    def record_fidelity(self, step: int, fidelity_score: float):
        self._write("fidelity", {"ts": time.time(), "step": step, "fidelity_score": round(fidelity_score, 4)})

    def _write(self, key: str, data: Dict[str, Any]):
        if key in self._files:
            self._files[key].write(json.dumps(data) + "\n")

    def close(self):
        for f in self._files.values():
            f.close()
