from typing import Dict, List, Any, Set
import logging

class LocalityAwareDeviceMapper:
    """
    Keeps cognition regions near their dominant execution device and symbolic neighborhood.
    """
    def __init__(self, devices: List[str]):
        self.devices = devices
        self.affinity_scores: Dict[str, Dict[str, float]] = {} # segment_id -> {device: score}
        self.neighborhood_map: Dict[str, Set[str]] = {} # segment_id -> {neighbor_ids}
        self.logger = logging.getLogger("LocalityAwareDeviceMapper")

    def track_affinity(self, segment_id: str, device: str, weight: float = 1.0):
        """Tracks how often a segment is accessed by a device."""
        if segment_id not in self.affinity_scores:
            self.affinity_scores[segment_id] = {d: 0.0 for d in self.devices}
        self.affinity_scores[segment_id][device] += weight

    def register_neighborhood(self, segment_id: str, neighbors: List[str]):
        """Registers symbolic neighbors for locality optimization."""
        if segment_id not in self.neighborhood_map:
            self.neighborhood_map[segment_id] = set()
        self.neighborhood_map[segment_id].update(neighbors)

    def get_optimal_device(self, segment_id: str) -> str:
        """Determines the best device for a segment based on affinity and neighborhood."""
        if segment_id not in self.affinity_scores:
            return self.devices[0] # Default
            
        scores = self.affinity_scores[segment_id].copy()
        
        # Boost scores based on neighbor residency
        # (This would require access to fabric to check where neighbors are)
        # For now, we use the raw affinity scores
        best_device = max(scores, key=scores.get)
        return best_device

    def get_mapping_metrics(self) -> Dict[str, Any]:
        """Returns metrics related to placement stability."""
        # Simple metric: how many segments have a clear dominant device
        stable_count = 0
        for seg_id, scores in self.affinity_scores.items():
            max_score = max(scores.values())
            total_score = sum(scores.values())
            if max_score / max(1.0, total_score) > 0.7:
                stable_count += 1
        
        return {
            "placement_stability": stable_count / max(1, len(self.affinity_scores)),
            "migration_thrash_risk": 1.0 - (stable_count / max(1, len(self.affinity_scores)))
        }
