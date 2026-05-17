import time
from typing import Dict, Any, List

class RuntimeLineageReconstructionEngine:
    """
    1. Runtime Lineage Reconstruction Engine
    
    Reconstructs end-to-end runtime flow, maps every emitted token to execution stages,
    traces all runtime layer participation, and verifies subsystem continuity.
    """
    def __init__(self):
        self.steps = []
        self.active_subsystems_log = []
        self.subsystem_counts = {}
        self.required_subsystems = {
            "CDBE", 
            "speculative_aware_batch_constructor", 
            "replay_affinity_routing", 
            "exl2_compatibility_engine", 
            "apix_runtime", 
            "uxr_runtime", 
            "reconstruction_layer"
        }

    def record_step(self, step: int, token_id: int, active_subsystems: List[str]) -> Dict[str, Any]:
        """
        Record and reconstruct execution lineage for a single token step.
        """
        timestamp = time.time()
        for sub in active_subsystems:
            self.subsystem_counts[sub] = self.subsystem_counts.get(sub, 0) + 1
            
        record = {
            "step": step,
            "token_id": token_id,
            "timestamp": timestamp,
            "active_subsystems": active_subsystems,
            "subsystem_participation_ratio": len(set(active_subsystems).intersection(self.required_subsystems)) / len(self.required_subsystems)
        }
        self.steps.append(record)
        self.active_subsystems_log.append(active_subsystems)
        return record

    def get_continuity_metric(self) -> float:
        """
        Returns the percentage of required subsystems that participated across the run.
        """
        if not self.steps:
            return 1.0
        
        all_participating = set()
        for step in self.steps:
            all_participating.update(step["active_subsystems"])
            
        participating_required = all_participating.intersection(self.required_subsystems)
        continuity = len(participating_required) / len(self.required_subsystems)
        return continuity * 100.0

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_steps": len(self.steps),
            "subsystem_counts": self.subsystem_counts,
            "runtime_continuity_percent": self.get_continuity_metric(),
            "status": "VALID" if self.get_continuity_metric() >= 99.0 else "DRIFTED"
        }
