"""
STAGE 2 - ASI: Semantic Pattern Memory Engine
Phase 39.6 - Adaptive Semantic Intelligence

Learns recurring semantic instability patterns such as collapse trajectories,
repair-success patterns, and recurring drift signatures.
"""
import threading
from typing import Dict, Any, List

class SemanticPatternMemoryEngine:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        # Memory storage: hash of recent drift/repair state -> success/failure count
        self._pattern_memory: Dict[str, Dict[str, int]] = {}
        # layer -> recent drift trajectory
        self._recent_trajectory: Dict[int, List[float]] = {i: [] for i in range(num_layers)}
        self._trajectory_len = 5

    def record_state(self, step: int, layer_idx: int, drift: float):
        with self._lock:
            self._recent_trajectory[layer_idx].append(drift)
            if len(self._recent_trajectory[layer_idx]) > self._trajectory_len:
                self._recent_trajectory[layer_idx].pop(0)

    def record_outcome(self, layer_idx: int, action_taken: str, successful: bool):
        with self._lock:
            if len(self._recent_trajectory[layer_idx]) < self._trajectory_len:
                return
                
            # Discretize trajectory to create a hashable pattern
            pattern = "-".join([str(round(d, 1)) for d in self._recent_trajectory[layer_idx]])
            pattern_key = f"L{layer_idx}_{pattern}_{action_taken}"
            
            if pattern_key not in self._pattern_memory:
                self._pattern_memory[pattern_key] = {"success": 0, "failure": 0}
                
            if successful:
                self._pattern_memory[pattern_key]["success"] += 1
            else:
                self._pattern_memory[pattern_key]["failure"] += 1

    def recall_pattern_success_rate(self, layer_idx: int, action: str) -> float:
        with self._lock:
            if len(self._recent_trajectory[layer_idx]) < self._trajectory_len:
                return 0.5 # Unknown
                
            pattern = "-".join([str(round(d, 1)) for d in self._recent_trajectory[layer_idx]])
            pattern_key = f"L{layer_idx}_{pattern}_{action}"
            
            memory = self._pattern_memory.get(pattern_key, None)
            if not memory: return 0.5
            
            total = memory["success"] + memory["failure"]
            if total == 0: return 0.5
            return memory["success"] / total

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "learned_patterns": len(self._pattern_memory),
                "highly_confident_patterns": sum(1 for m in self._pattern_memory.values() if (m["success"] + m["failure"]) > 5)
            }
