from typing import Dict, List, Tuple
import torch

class SemanticGeometryMapper:
    """
    PHASE 18.7C: Semantic Geometry Mapper.
    Maps spatial/geometric relationships between symbolic capsules.
    """
    def __init__(self):
        self.capsule_positions: Dict[str, int] = {}
        self.adjacency_matrix = None

    def update_positions(self, registry):
        for cid, capsule in registry.capsules.items():
            self.capsule_positions[cid] = (capsule.start_idx + capsule.end_idx) // 2

    def get_geometric_distance(self, cid1: str, cid2: str) -> int:
        pos1 = self.capsule_positions.get(cid1)
        pos2 = self.capsule_positions.get(cid2)
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1 - pos2)

    def find_nearest_neighbors(self, capsule_id: str, k: int = 3) -> List[str]:
        target_pos = self.capsule_positions.get(capsule_id)
        if target_pos is None:
            return []
            
        distances = []
        for cid, pos in self.capsule_positions.items():
            if cid != capsule_id:
                distances.append((cid, abs(pos - target_pos)))
                
        distances.sort(key=lambda x: x[1])
        return [d[0] for d in distances[:k]]
