"""
agents/memory_lifecycle_manager.py

Phase 12A: Memory Lifecycle Manager
Manages the creation, promotion, and retirement of semantic anchors
throughout an agent's multi-session existence.
"""

from typing import List, Dict, Any
from anchor_logic.semantic_anchor_system import SemanticAnchor, SemanticAnchorMemory
from agents.persistent_memory_store import PersistentMemoryStore

class MemoryLifecycleManager:
    """
    Implements policies for long-term memory health.
    Handles 'consolidation' (moving frequent short-term anchors to long-term).
    """
    def __init__(self, memory: SemanticAnchorMemory, long_term_store: PersistentMemoryStore):
        self.memory = memory
        self.store = long_term_store
        self.access_counts: Dict[int, int] = {} # position -> count

    def record_access(self, position: int):
        """Tracks usage of an anchor to inform promotion/eviction."""
        self.access_counts[position] = self.access_counts.get(position, 0) + 1

    def consolidate_memory(self, session_id: str):
        """
        Identifies 'hot' anchors and ensures they are persisted in a 
        'global_memory' session for future use across all sessions.
        """
        hot_anchors = [
            pos for pos, count in self.access_counts.items() 
            if count > 5 and pos in self.memory.anchors
        ]
        
        if not hot_anchors:
            return

        print(f"[MemoryLifecycleManager] Consolidating {len(hot_anchors)} hot anchors for session '{session_id}'")
        
        # Load global memory
        global_mem = self.store.load_session("global_agent_memory")
        if not global_mem:
            global_mem = SemanticAnchorMemory(max_anchors=1024)

        for pos in hot_anchors:
            anchor = self.memory.anchors[pos]
            global_mem.add_anchor(anchor)

        self.store.save_session("global_agent_memory", global_mem)

    def prune_stale_anchors(self):
        """Removes anchors that haven't been accessed in a long time."""
        # Simplified: remove anchors with 0 access if memory is full
        if len(self.memory.anchors) >= self.memory.max_anchors:
            to_remove = [
                pos for pos in self.memory.anchors 
                if self.access_counts.get(pos, 0) == 0
            ]
            for pos in to_remove:
                del self.memory.anchors[pos]
            print(f"[MemoryLifecycleManager] Pruned {len(to_remove)} stale anchors.")
