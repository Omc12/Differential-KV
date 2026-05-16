"""
STAGE 2 - ASI: Semantic Intelligence Trace System
Phase 39.6 - Adaptive Semantic Intelligence

Persists raw traces for learned fragility patterns, policy adaptations,
and boundary evolutions.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

class SemanticIntelligenceTraceSystem:
    def __init__(self, run_id: str):
        self.trace_dir = Path("traces/stage2/phase_39_6_asi") / run_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self._files = {
            "pattern":   open(self.trace_dir / "semantic_pattern_trace.jsonl",   "a", encoding="utf-8", buffering=1),
            "policy":    open(self.trace_dir / "policy_learning_trace.jsonl",    "a", encoding="utf-8", buffering=1),
            "strategy":  open(self.trace_dir / "recovery_strategy_trace.jsonl",  "a", encoding="utf-8", buffering=1),
            "fragility": open(self.trace_dir / "fragility_learning_trace.jsonl", "a", encoding="utf-8", buffering=1),
            "boundary":  open(self.trace_dir / "sparse_boundary_trace.jsonl",    "a", encoding="utf-8", buffering=1),
        }

    def record_pattern(self, step: int, layer_idx: int, outcome: str):
        self._write("pattern", {"ts": time.time(), "step": step, "layer": layer_idx, "outcome": outcome})

    def record_policy(self, step: int, best_policy: str, confidence: float):
        self._write("policy", {"ts": time.time(), "step": step, "best_policy": best_policy, "confidence": round(confidence, 4)})

    def record_strategy(self, step: int, layer_idx: int, best_strategy: str):
        self._write("strategy", {"ts": time.time(), "step": step, "layer": layer_idx, "best_strategy": best_strategy})

    def record_fragility(self, step: int, fragile_count: int, avg_score: float):
        self._write("fragility", {"ts": time.time(), "step": step, "fragile_count": fragile_count, "avg_fragility": round(avg_score, 4)})

    def record_boundary(self, step: int, safe_chain: float, safe_ratio: float):
        self._write("boundary", {"ts": time.time(), "step": step, "safe_chain": round(safe_chain, 2), "safe_ratio": round(safe_ratio, 4)})

    def _write(self, key: str, data: Dict[str, Any]):
        if key in self._files:
            self._files[key].write(json.dumps(data) + "\n")

    def close(self):
        for f in self._files.values():
            f.close()
