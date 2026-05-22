import time
import torch
from typing import Dict, Any, List
from .autonomous_memory_consolidation import AutonomousMemoryConsolidation

class ManifoldSleepCycle:
    """
    Orchestrates periodic 'sleep cycles' for cognitive maintenance.
    During 'sleep', the system consolidates memories and prunes weak attractors.
    """
    def __init__(self, consolidation_engine: AutonomousMemoryConsolidation):
        self.engine = consolidation_engine
        self.last_sleep_time = time.time()
        self.sleep_count = 0

    def trigger_sleep_cycle(self, current_manifolds: torch.Tensor) -> Dict[str, Any]:
        """
        Executes a sleep cycle on the current cognitive state.
        """
        print(f"--- Initiating Manifold Sleep Cycle {self.sleep_count} ---")
        start_time = time.time()
        
        initial_redundancy = self.engine.compute_redundancy_score(current_manifolds)
        
        # 1. Consolidate attractors
        consolidated = self.engine.consolidate_manifolds(current_manifolds)
        
        # 2. Extract motifs
        motifs = self.engine.extract_long_term_motifs(consolidated)
        
        # 3. Prune low-energy manifolds (simulated by reduction in size)
        final_redundancy = self.engine.compute_redundancy_score(consolidated)
        
        self.sleep_count += 1
        self.last_sleep_time = time.time()
        
        duration = time.time() - start_time
        
        return {
            "duration": duration,
            "redundancy_reduction": initial_redundancy - final_redundancy,
            "preserved_motifs": motifs.shape[0],
            "manifold_compression": len(consolidated) / len(current_manifolds)
        }

    def should_sleep(self, token_count: int, threshold: int = 10000) -> bool:
        """
        Determines if it's time for a sleep cycle based on activity.
        """
        return token_count > threshold
