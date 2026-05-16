import json
import time
from pathlib import Path
from typing import Dict, Any

class SemanticDriftTraceSystem:
    """
    SGC Phase 39.1 RESET: Semantic Drift Trace System.
    Persists RAW traces of semantic equivalence, drift, and unsafe suppressions.
    """
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self.traces = {
            "semantic_equivalence": self.trace_dir / "semantic_equivalence_trace.jsonl",
            "semantic_drift": self.trace_dir / "semantic_drift_trace.jsonl",
            "unsafe_suppression": self.trace_dir / "unsafe_suppression_trace.jsonl",
            "governance_truth": self.trace_dir / "governance_truth_trace.jsonl"
        }

    def log_event(self, trace_key: str, data: Dict[str, Any]):
        if trace_key not in self.traces:
            return
            
        payload = {
            "ts": time.time(),
            **data
        }
        
        with open(self.traces[trace_key], "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def record_drift(self, step: int, kl_div: float, preservation_score: float):
        self.log_event("semantic_drift", {
            "step": step,
            "kl_divergence": kl_div,
            "preservation_score": preservation_score
        })

    def record_equivalence(self, step: int, tokens_match: bool, sparse_token: int, dense_token: int):
        self.log_event("semantic_equivalence", {
            "step": step,
            "tokens_match": tokens_match,
            "sparse_token": sparse_token,
            "dense_token": dense_token
        })

    def record_unsafe(self, step: int, event: Dict[str, Any]):
        self.log_event("unsafe_suppression", {
            "step": step,
            **event
        })

    def record_truth(self, metrics: Dict[str, float]):
        self.log_event("governance_truth", metrics)
