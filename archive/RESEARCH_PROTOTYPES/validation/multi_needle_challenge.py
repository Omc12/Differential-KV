"""
validation/multi_needle_challenge.py

Phase 12.5D: Multi-Needle Challenge
Evaluates retrieval robustness when multiple, highly similar "needles" are 
present in the context, but only one is the correct answer to the specific query.
"""

from typing import List, Dict, Any
from anchor_logic.semantic_anchor_system import SemanticAnchorMemory, SemanticAnchor
from validation.semantic_similarity_validator import SemanticSimilarityValidator
import torch

class MultiNeedleChallenge:
    """
    Stress-tests the retrieval system against semantic ambiguity.
    """
    def __init__(self, memory: SemanticAnchorMemory):
        self.memory = memory
        self.validator = SemanticSimilarityValidator(threshold=0.8)

    def inject_needles(self, base_position: int = 10000):
        """Injects the true needle and several similar decoys."""
        # True needle
        self.memory.add_anchor(SemanticAnchor(
            token_id=100, position=base_position,
            reason="true_needle", metadata={"text": "The server timeout is 45 seconds."}
        ))
        
        # Decoys
        decoys = [
            "The client timeout is 45 seconds.",
            "The server timeout is 30 seconds.",
            "The server connection limit is 45.",
            "A server timeout occurred after 45 seconds yesterday."
        ]
        
        for i, decoy in enumerate(decoys):
            self.memory.add_anchor(SemanticAnchor(
                token_id=101+i, position=base_position + (i+1)*1000,
                reason="decoy_needle", metadata={"text": decoy}
            ))

    def evaluate_retrieval(self, query: str = "What is the configured server timeout?") -> Dict[str, Any]:
        """
        Attempts to retrieve the answer and checks if it fell for a decoy.
        (Simulates retrieval by scanning metadata for this test)
        """
        best_match = None
        highest_sim = 0.0
        
        for anchor in self.memory.anchors.values():
            if "text" in anchor.metadata:
                val = self.validator.validate_match(query, anchor.metadata["text"])
                if val["similarity"] > highest_sim:
                    highest_sim = val["similarity"]
                    best_match = anchor

        success = best_match is not None and best_match.reason == "true_needle"
        
        return {
            "success": success,
            "retrieved_reason": best_match.reason if best_match else "none",
            "highest_similarity": highest_sim
        }
