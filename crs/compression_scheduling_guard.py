
import torch
from typing import Dict, Any, List, Optional

class CompressionSchedulingGuard:
    """
    PHASE 23.4a: CRS-ARC Integration Patch - Compression Scheduling Guard.
    Prevents thrashing and detects pathological compression/scheduling loops.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recent_rehydrations = []
        
        self.metrics = {
            "compression_scheduling_stability": 1.0,
            "thrash_suppression_active": False,
            "symbolic_continuity": 1.0
        }

    def validate_scheduling_step(self, 
                                 scheduled_rehydrations: List[int], 
                                 step: int) -> bool:
        """
        Detects if we are rehydrating the same regions too frequently (thrashing).
        """
        self.recent_rehydrations.append((step, scheduled_rehydrations))
        if len(self.recent_rehydrations) > 5:
            self.recent_rehydrations.pop(0)
            
        # Thrashing detection: same region rehydrated in >3 out of 5 steps
        counts = {}
        for _, regions in self.recent_rehydrations:
            for r in regions:
                counts[r] = counts.get(r, 0) + 1
                
        thrashing = any(count >= 3 for count in counts.values())
        
        if thrashing:
            self.metrics["thrash_suppression_active"] = True
            self.metrics["compression_scheduling_stability"] *= 0.9
            return False
            
        self.metrics["thrash_suppression_active"] = False
        self.metrics["compression_scheduling_stability"] = 0.99
        self.metrics["symbolic_continuity"] = 1.0
        
        return True

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
