
import torch
from typing import Dict, List, Any, Optional

class CooperativeExecutionMemory:
    """
    PHASE 22.3: HEC - Cooperative Execution Memory.
    Remembers successful delegation and coordination patterns.
    """
    def __init__(self, memory_size: int = 64):
        self.memory_size = memory_size
        self.coordination_history: List[Dict[str, Any]] = []
        self.partnership_strengths: Dict[str, float] = {} # "modeA:modeB" -> strength

    def record_coordination(self, 
                             partners: List[str], 
                             success_score: float):
        """
        Records a coordination event and updates partnership strengths.
        """
        partners.sort()
        key = ":".join(partners)
        
        self.partnership_strengths[key] = (
            0.9 * self.partnership_strengths.get(key, 0.5) + 0.1 * success_score
        )
        
        self.coordination_history.append({
            "key": key,
            "score": success_score
        })
        
        if len(self.coordination_history) > self.memory_size:
            self.coordination_history.pop(0)

    def get_best_partner(self, mode_name: str) -> Optional[str]:
        """
        Suggests the best mode to cooperate with based on history.
        """
        best_partner = None
        max_strength = -1.0
        
        for key, strength in self.partnership_strengths.items():
            if mode_name in key:
                parts = key.split(":")
                if len(parts) > 1:
                    partner = parts[0] if parts[1] == mode_name else parts[1]
                    if strength > max_strength:
                        max_strength = strength
                        best_partner = partner
                    
        return best_partner

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "memory_utilization": len(self.coordination_history) / self.memory_size,
            "stable_partnerships": len([s for s in self.partnership_strengths.values() if s > 0.8])
        }
