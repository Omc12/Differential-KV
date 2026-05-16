"""
benchmarks/multi_session_reasoning.py

Phase 12D: Multi-Session Reasoning
Benchmarks the agent's ability to 'reason' about information provided in 
previous sessions.
"""

from typing import List, Dict, Any
from agents.persistent_memory_store import PersistentMemoryStore
from anchor_logic.semantic_anchor_system import SemanticAnchorMemory, SemanticAnchor

class MultiSessionReasoningBench:
    """
    Tests if the agent can use 'anchors' from Session 1 to answer a 
    question in Session 2.
    """
    def __init__(self, store: PersistentMemoryStore):
        self.store = store

    def setup_session_1(self, session_id: str, fact_pos: int, fact_token: int):
        """Stores a 'fact' as a semantic anchor in Session 1."""
        mem = SemanticAnchorMemory(max_anchors=128)
        anchor = SemanticAnchor(
            token_id=fact_token,
            position=fact_pos,
            reason="fact_storage",
            metadata={"fact": "The secret key is 42"}
        )
        mem.add_anchor(anchor)
        self.store.save_session(session_id, mem)

    def test_session_2_reasoning(self, session_id: str, fact_pos: int) -> bool:
        """Checks if the fact from Session 1 is available in Session 2."""
        mem = self.store.load_session(session_id)
        if not mem: return False
        
        anchor = mem.anchors.get(fact_pos)
        if anchor and anchor.metadata.get("fact") == "The secret key is 42":
            return True
        return False
