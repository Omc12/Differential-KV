from typing import Dict, List
import numpy as np

class AttractorLifecycleEngine:
    """
    Handles the mechanics of attractor birth, health updates, merging, and death.
    """
    def birth(self, candidate: Dict) -> Dict:
        """
        Initializes a new attractor in the ecosystem.
        """
        return {
            "id": candidate['id'],
            "health": 1.0,
            "age": 0,
            "reusable": True,
            "volatility": candidate['volatility'],
            "density": candidate['density'],
            "suppression": False
        }
        
    def update_health(self, attractor: Dict, candidate: Dict) -> Dict:
        """
        Updates the health of an existing attractor based on its persistence.
        """
        attractor['age'] += 1
        # Health increases if it remains stable, decreases if it becomes volatile
        health_delta = 0.1 if candidate['volatility'] < 0.3 else -0.2
        attractor['health'] = np.clip(attractor['health'] + health_delta, 0.0, 2.0)
        attractor['volatility'] = candidate['volatility']
        attractor['density'] = candidate['density']
        return attractor
        
    def handle_competition(self, active_attractors: Dict) -> Dict:
        """
        Simulates competition for manifold 'real estate'.
        Merges overlapping attractors or suppresses weaker ones.
        """
        # Logic to merge similar attractors (simplified)
        # In practice, compare centers and health
        return active_attractors
        
    def should_die(self, attractor: Dict) -> bool:
        """
        Determines if an attractor should be pruned (GCed).
        """
        # Prune if health is zero or it's suppressed and very old
        if attractor['health'] <= 0:
            return True
        if attractor['suppression'] and attractor['age'] > 50:
            return True
        return False
        
    def get_reusable_ratio(self, active_attractors: Dict) -> float:
        """
        Returns the percentage of attractors that are stable enough for reuse.
        """
        if not active_attractors:
            return 0.0
        reusable = sum(1 for a in active_attractors.values() if a['health'] > 0.8)
        return reusable / len(active_attractors)
