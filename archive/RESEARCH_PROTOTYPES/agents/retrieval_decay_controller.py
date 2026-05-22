"""
agents/retrieval_decay_controller.py

Manages the natural decay of retrieval scores over long agent sessions.
Prevents 'retrieval inertia' where old context overrides new relevance.
"""

import time
from typing import Dict, Any
import logging

class RetrievalDecayController:
    """
    Temporal decay manager for agent context.
    """
    def __init__(self, decay_rate: float = 0.99):
        self.decay_rate = decay_rate
        self.last_update = time.time()
        self.scores: Dict[int, float] = {} # shard_id -> retrieval_score
        self.logger = logging.getLogger("RetrievalDecayController")

    def update_scores(self, active_shards: Dict[int, float]):
        """
        Applies decay to all scores and boosts the current active set.
        """
        now = time.time()
        dt = now - self.last_update
        
        # Apply global decay
        for sid in self.scores:
            self.scores[sid] *= (self.decay_rate ** dt)
            
        # Boost active set
        for sid, score in active_shards.items():
            self.scores[sid] = max(self.scores.get(sid, 0.0), score)
            
        self.last_update = now

    def get_top_context(self, k: int = 10) -> List[int]:
        """Returns the IDs of the most relevant (least decayed) shards."""
        return sorted(self.scores, key=self.scores.get, reverse=True)[:k]
