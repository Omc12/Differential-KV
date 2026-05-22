"""
memory/universal_attractor_library.py
Phase 19: Universal Cognitive Geometry
Library for storing and retrieving reusable cognitive attractor motifs.
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

class UniversalAttractorLibrary:
    def __init__(self, storage_path: str = "results/phase19/attractor_library"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        self.library = {} # motif_id -> data

    def store_motif(self, motif_id: str, trajectory: np.ndarray, metadata: Dict[str, Any]):
        """
        Stores a cognitive motif (trajectory/basin).
        """
        path = os.path.join(self.storage_path, f"{motif_id}.npz")
        np.savez(path, trajectory=trajectory, metadata=json.dumps(metadata))
        self.library[motif_id] = metadata

    def retrieve_motif(self, motif_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        path = os.path.join(self.storage_path, f"{motif_id}.npz")
        if not os.path.exists(path):
            return None
        
        data = np.load(path, allow_pickle=True)
        trajectory = data["trajectory"]
        metadata = json.loads(str(data["metadata"]))
        return trajectory, metadata

    def search_by_metadata(self, query: Dict[str, Any]) -> List[str]:
        results = []
        for mid, meta in self.library.items():
            match = True
            for k, v in query.items():
                if meta.get(k) != v:
                    match = False
                    break
            if match:
                results.append(mid)
        return results

if __name__ == "__main__":
    lib = UniversalAttractorLibrary()
    lib.store_motif("logic_step_1", np.random.randn(5, 768), {"type": "reasoning", "model": "Qwen2"})
    
    traj, meta = lib.retrieve_motif("logic_step_1")
    print(f"Retrieved Motif: {meta}")
