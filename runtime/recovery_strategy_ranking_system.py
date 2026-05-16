"""
STAGE 2 - ASI: Recovery Strategy Ranking System
Phase 39.6 - Adaptive Semantic Intelligence

Ranks which recovery strategies work best for specific semantic conditions.
"""
import threading
from typing import Dict, Any, List

class RecoveryStrategyRankingSystem:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        self.strategies = ["anchor_reinforcement", "layer_densification", "hybrid_routing", "localized_window"]
        
        # Layer -> Strategy -> Effectiveness Score
        self._layer_strategy_scores: Dict[int, Dict[str, float]] = {
            i: {s: 0.5 for s in self.strategies} for i in range(num_layers)
        }
        self._sample_counts: Dict[int, Dict[str, int]] = {
            i: {s: 0 for s in self.strategies} for i in range(num_layers)
        }

    def record_outcome(self, layer_idx: int, strategy: str, drift_reduction: float, persistence: int):
        with self._lock:
            if strategy not in self.strategies: return
            
            # Heuristic score for the outcome
            score = min(1.0, (drift_reduction / 2.0) * 0.6 + (persistence / 20.0) * 0.4)
            
            current = self._layer_strategy_scores[layer_idx][strategy]
            samples = self._sample_counts[layer_idx][strategy]
            
            self._layer_strategy_scores[layer_idx][strategy] = (current * samples + score) / (samples + 1)
            self._sample_counts[layer_idx][strategy] += 1

    def get_best_strategy(self, layer_idx: int) -> str:
        with self._lock:
            scores = self._layer_strategy_scores[layer_idx]
            return max(scores.items(), key=lambda x: x[1])[0]

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            # Aggregate global preference
            global_scores = {s: 0.0 for s in self.strategies}
            for i in range(self.num_layers):
                for s in self.strategies:
                    global_scores[s] += self._layer_strategy_scores[i][s]
                    
            best_global = max(global_scores.items(), key=lambda x: x[1])[0]
            
            return {
                "top_global_strategy": best_global,
                "learning_samples": sum(sum(sc.values()) for sc in self._sample_counts.values())
            }
