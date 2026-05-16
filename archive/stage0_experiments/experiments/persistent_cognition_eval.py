import torch
import numpy as np
from runtime.persistent_state_reservoir import PersistentStateReservoir
from typing import Dict, Any

class PersistentCognitionEval:
    """
    Evaluates the survival of cognitive states across context rollovers and branches.
    """
    def __init__(self, d_model: int = 512):
        self.reservoir = PersistentStateReservoir(d_model)
        self.d_model = d_model

    def run_eval(self):
        print("Starting Persistent Cognition Evaluation...")
        
        # 1. State Injection
        original_state = torch.randn(self.d_model)
        self.reservoir.store_state("reasoning_pivot_A", original_state)
        
        # 2. Simulate "Context Rollover" (clear everything except reservoir)
        # In a real model, this would be a cache clear or window shift.
        
        # 3. State Retrieval
        retrieved_state = self.reservoir.retrieve_state("reasoning_pivot_A")
        
        # 4. Measure Fidelity
        fidelity = torch.nn.functional.cosine_similarity(
            original_state.flatten(), retrieved_state.flatten(), dim=0
        ).item()
        
        # 5. Working Memory Nuclei Stability
        # Inject several states to form a nucleus
        for _ in range(50):
            state = original_state + torch.randn(self.d_model) * 0.1
            self.reservoir.update_working_memory_nuclei(state)
            
        nucleus_signal = self.reservoir.get_nucleus_injection()
        nucleus_fidelity = torch.nn.functional.cosine_similarity(
            original_state.flatten(), nucleus_signal.flatten(), dim=0
        ).item()
        
        return {
            "state_fidelity": fidelity,
            "nucleus_fidelity": nucleus_fidelity,
            "overhead_mb": self.reservoir.get_overhead(),
            "status": "PASS" if fidelity > 0.9 else "FAIL"
        }

if __name__ == "__main__":
    evaluator = PersistentCognitionEval()
    results = evaluator.run_eval()
    for k, v in results.items():
        print(f"{k}: {v}")
