
import torch
from typing import Dict, Any, List, Optional
import time

class ExecutionSpecializationMatrix:
    """
    PHASE 22.2: ESM - Execution Specialization Matrix.
    Maps cognitive workloads to specialized execution strategies.
    """
    def __init__(self):
        self.active_mode = "semantic" # Default
        self.mode_weights = {
            "symbolic": 0.0,
            "semantic": 1.0,
            "topology": 0.0,
            "dormant": 0.0
        }
        self.switch_history = []
        self.last_switch_time = time.time()
        
    def determine_mode(self, 
                       symbolic_density: float, 
                       semantic_complexity: float, 
                       topology_drift: float,
                       inactivity_score: float) -> str:
        """
        Dynamically calculates the optimal execution mode based on cognitive signals.
        """
        # Calculate scores for each mode
        scores = {
            "symbolic": symbolic_density * 1.5, # Boost symbolic priority
            "semantic": semantic_complexity,
            "topology": topology_drift * 2.0, # High priority for repair
            "dormant": inactivity_score * 0.8
        }
        
        # Softmax-like selection or winner-takes-most
        best_mode = max(scores, key=scores.get)
        
        # Stability check: don't switch too frequently
        if best_mode != self.active_mode:
            if time.time() - self.last_switch_time > 0.5: # 500ms cooldown
                self.switch_history.append((self.active_mode, best_mode, time.time()))
                self.active_mode = best_mode
                self.last_switch_time = time.time()
                
        # Update mode weights for blending
        for mode in self.mode_weights:
            # Momentum-based weight update
            target = 1.0 if mode == self.active_mode else 0.0
            self.mode_weights[mode] = 0.7 * self.mode_weights[mode] + 0.3 * target
            
        return self.active_mode

    def get_execution_parameters(self) -> Dict[str, Any]:
        """
        Returns parameters for the current specialized execution state.
        """
        return {
            "mode": self.active_mode,
            "weights": self.mode_weights,
            "localization_factor": 1.0 - self.mode_weights["semantic"] * 0.5 # Semantic is more distributed
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "active_mode": self.active_mode,
            "mode_switch_count": len(self.switch_history),
            "mode_switch_stability": 1.0 / (1.0 + len(self.switch_history) * 0.01)
        }
