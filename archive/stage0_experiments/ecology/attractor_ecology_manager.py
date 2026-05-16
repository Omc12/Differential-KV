import torch
from typing import Dict, List, Set
from ecology.manifold_population_tracker import ManifoldPopulationTracker
from ecology.attractor_lifecycle_engine import AttractorLifecycleEngine

class AttractorEcologyManager:
    """
    Manages the birth, competition, and death of reasoning attractors in the cognitive manifold.
    Ensures sustainable population growth and suppresses parasitic manifolds.
    """
    def __init__(self, max_population: int = 50):
        self.max_population = max_population
        self.tracker = ManifoldPopulationTracker()
        self.lifecycle = AttractorLifecycleEngine()
        
        self.active_attractors = {} # ID -> Metadata
        self.parasitic_manifolds = set()
        
    def evolve_ecosystem(self, latent_trajectories: torch.Tensor):
        """
        Processes new trajectories to update attractor states.
        """
        # 1. Identify new potential attractors
        new_candidates = self.tracker.identify_clusters(latent_trajectories)
        
        # 2. Process lifecycle for each candidate
        for candidate in new_candidates:
            if candidate['id'] not in self.active_attractors:
                if len(self.active_attractors) < self.max_population:
                    self.active_attractors[candidate['id']] = self.lifecycle.birth(candidate)
            else:
                self.active_attractors[candidate['id']] = self.lifecycle.update_health(
                    self.active_attractors[candidate['id']], candidate
                )
        
        # 3. Competition and Merging
        self.active_attractors = self.lifecycle.handle_competition(self.active_attractors)
        
        # 4. Suppress parasitic (unstable or low-utility) manifolds
        self.parasitic_manifolds = self.tracker.detect_parasites(self.active_attractors)
        for pid in self.parasitic_manifolds:
            if pid in self.active_attractors:
                self.active_attractors[pid]['suppression'] = True
                
        # 5. Garbage Collection (Death)
        self.active_attractors = {
            aid: meta for aid, meta in self.active_attractors.items() 
            if not self.lifecycle.should_die(meta)
        }
        
    def get_ecology_stats(self) -> Dict:
        return {
            "population_count": len(self.active_attractors),
            "parasite_count": len(self.parasitic_manifolds),
            "reusable_basin_ratio": self.lifecycle.get_reusable_ratio(self.active_attractors),
            "system_carrying_capacity": self.max_population
        }
