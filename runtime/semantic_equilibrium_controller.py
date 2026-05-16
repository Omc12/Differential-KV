"""
STAGE 2 - ASS: Semantic Equilibrium Controller
Phase 39.5 - Adaptive Semantic Scheduling

Maintains stable operating regions and avoids oscillatory repair storms
by monitoring the global equilibrium state.
"""
import threading
from typing import Dict, Any

class SemanticEquilibriumController:
    """
    Acts as a high-level governor. If the system enters an unstable burst
    (many layers forecasting collapse or oscillating), it forces a temporary
    global fallback to re-establish the anchor foundation.
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        self._equilibrium_score = 1.0 # 1.0 = perfect equilibrium, 0.0 = complete collapse
        self._global_fallback_cooldown = 0
        self._fallback_events = 0

    def update(self, global_pressure: float, oscillation_count: int, average_chain: float):
        with self._lock:
            if self._global_fallback_cooldown > 0:
                self._global_fallback_cooldown -= 1
                # During cooldown, equilibrium artificially recovers
                self._equilibrium_score = min(1.0, self._equilibrium_score + 0.1)
                return True # Indicates forced global dense mode
                
            # Calculate equilibrium based on pressure (lower is better), 
            # oscillations (lower is better), and chain length (higher is better)
            
            pressure_penalty = global_pressure
            oscillation_penalty = min(oscillation_count / 20.0, 1.0)
            chain_bonus = min(average_chain / 50.0, 1.0)
            
            # Update score with smoothing
            target_score = 1.0 - (pressure_penalty * 0.5) - (oscillation_penalty * 0.5) + (chain_bonus * 0.2)
            target_score = max(0.0, min(1.0, target_score))
            
            self._equilibrium_score = (self._equilibrium_score * 0.8) + (target_score * 0.2)
            
            # If equilibrium breaks entirely, trigger a global reset
            if self._equilibrium_score < 0.3:
                self._global_fallback_cooldown = 10 # 10 steps of forced stability
                self._fallback_events += 1
                return True
                
            return False

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "equilibrium_score": round(self._equilibrium_score, 4),
                "global_fallback_events": self._fallback_events,
                "in_global_fallback": self._global_fallback_cooldown > 0
            }
