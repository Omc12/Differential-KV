from typing import List, Dict, Any, Callable
from runtime.iterative_refinement_engine import IterativeRefinementEngine
import time

class IterativeReasoningSuite:
    """
    Suite for testing refinement loops and iterative reasoning stability.
    Benchmarks convergence, latency, and refinement quality.
    """
    def __init__(self, engine: IterativeRefinementEngine):
        self.engine = engine

    def run_benchmark(self, name: str, task_fn: Callable, initial_state: Any) -> Dict[str, Any]:
        """Runs a specific iterative reasoning benchmark."""
        print(f"Running benchmark: {name}...")
        
        # Simple convergence check: stop if state doesn't change much
        def convergence_fn(state):
            return state.get("delta", 1.0) < 0.01

        result = self.engine.refine(initial_state, task_fn, convergence_fn)
        
        return {
            "benchmark_name": name,
            "result": result,
            "timestamp": time.time()
        }

    def run_standard_suite(self) -> List[Dict[str, Any]]:
        """Runs a set of predefined benchmarks."""
        results = []
        
        # Example task: numeric refinement
        def numeric_refinement(state, i):
            val = state.get("value", 100.0)
            target = 10.0
            new_val = val + (target - val) * 0.5
            return {"value": new_val, "delta": abs(new_val - val)}

        results.append(self.run_benchmark("Numeric Converge", numeric_refinement, {"value": 100.0, "delta": 100.0}))
        
        return results
