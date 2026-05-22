"""
agents/context_decay_controller.py

Phase 12A: Context Decay Controller
Manages the temporal aspect of memory, implementing aging and relevance 
decay for sparse anchors.
"""

import time
from typing import Dict, List
from anchor_logic.semantic_anchor_system import SemanticAnchorMemory

class ContextDecayController:
    """
    Simulates the 'forgetting' process for less relevant anchors.
    Ensures long-horizon memory stability by preventing anchor bloat.
    """
    def __init__(self, memory: SemanticAnchorMemory, half_life_seconds: int = 3600):
        self.memory = memory
        self.half_life = half_life_seconds
        self.anchor_birthdays: Dict[int, float] = {
            pos: time.time() for pos in memory.anchors
        }

    def update_anchor_scores(self):
        """Applies exponential decay to importance scores based on age."""
        now = time.time()
        to_remove = []
        
        for pos, anchor in self.memory.anchors.items():
            age = now - self.anchor_birthdays.get(pos, now)
            # score = score * (0.5 ^ (age / half_life))
            decay_factor = 0.5 ** (age / self.half_life)
            anchor.importance_score *= decay_factor
            
            # If importance falls below a critical threshold, mark for removal
            if anchor.importance_score < 0.01:
                to_remove.append(pos)

        for pos in to_remove:
            del self.memory.anchors[pos]
            if pos in self.anchor_birthdays:
                del self.anchor_birthdays[pos]
        
        if to_remove:
            print(f"[ContextDecayController] Decayed and removed {len(to_remove)} low-relevance anchors.")

    def refresh_anchor(self, position: int):
        """Resets the age of an anchor when it is accessed."""
        self.anchor_birthdays[position] = time.time()
        if position in self.memory.anchors:
            # Reset importance to a base level
            self.memory.anchors[position].importance_score = max(
                self.memory.anchors[position].importance_score, 1.0
            )
