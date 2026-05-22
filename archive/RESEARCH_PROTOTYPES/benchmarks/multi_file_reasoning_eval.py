"""
benchmarks/multi_file_reasoning_eval.py

Phase 12.5C: Multi-File Reasoning Evaluation
Benchmarks the agent's ability to maintain context across many files
simultaneously during complex reasoning tasks.
"""

from typing import List, Dict, Any
from validation.cross_file_reasoning_validator import CrossFileReasoningValidator
from validation.real_memory_latency_meter import RealMemoryLatencyMeter
import time

class MultiFileReasoningEval:
    """
    Evaluates deep multi-file dependency reasoning.
    """
    def __init__(self, router):
        self.router = router
        self.validator = CrossFileReasoningValidator()
        self.meter = RealMemoryLatencyMeter()

    def evaluate_complex_refactor(self, refactor_intent: str, required_files: List[str]) -> Dict[str, Any]:
        self.validator.register_chain(refactor_intent, required_files)
        
        # Simulate retrieval using the meter to ensure realistic timing
        def _simulate_retrieval():
            anchors = self.router.route_query(refactor_intent)
            retrieved = set()
            for a in anchors:
                if "rel_path" in a.metadata:
                    retrieved.add(a.metadata["rel_path"])
            return list(retrieved)

        retrieved_files, latency = self.meter.measure("complex_refactor_retrieval", _simulate_retrieval)
        
        validation_result = self.validator.evaluate_retrieval(refactor_intent, retrieved_files)
        
        return {
            "intent": refactor_intent,
            "chain_completion": validation_result["chain_completion"],
            "latency_ms": latency,
            "missing_files": validation_result.get("missing", [])
        }
