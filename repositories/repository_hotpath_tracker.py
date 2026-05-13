"""
repositories/repository_hotpath_tracker.py

Phase 12B: Repository Hotpath Tracker
Tracks which parts of the repository are most frequently 'retrieved' or
modified, ensuring high-priority sparse residency for critical code paths.
"""

import time
from typing import Dict, List, Tuple

class RepositoryHotpathTracker:
    """
    Monitors access frequency and 'recency' for repository files.
    Informs the eviction and pre-fetching logic.
    """
    def __init__(self, decay_rate: float = 0.9):
        self.access_scores: Dict[str, float] = {} # rel_path -> score
        self.last_access: Dict[str, float] = {}   # rel_path -> timestamp
        self.decay_rate = decay_rate

    def record_access(self, rel_path: str):
        """Increments score and updates timestamp."""
        now = time.time()
        self.access_scores[rel_path] = self.access_scores.get(rel_path, 0) + 1.0
        self.last_access[rel_path] = now

    def get_hotpaths(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Returns the most frequently accessed files, with temporal decay."""
        now = time.time()
        decayed_scores = {}
        
        for path, score in self.access_scores.items():
            age_hours = (now - self.last_access[path]) / 3600
            # Apply decay: score * (decay_rate ^ age)
            decayed_scores[path] = score * (self.decay_rate ** age_hours)
            
        sorted_paths = sorted(decayed_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_paths[:top_k]

    def reset_score(self, rel_path: str):
        if rel_path in self.access_scores:
            self.access_scores[rel_path] = 0
