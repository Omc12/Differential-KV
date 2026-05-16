"""
validation/cross_file_reasoning_validator.py

Phase 12.5A: Cross-File Reasoning Validator
Evaluates if the retrieval system can successfully pull together related
semantic concepts that are spread across multiple separate files.
"""

from typing import List, Dict, Set

class CrossFileReasoningValidator:
    """
    Validates "multi-hop" retrieval where answering a query requires
    context from File A, which points to File B, which contains the answer.
    """
    def __init__(self):
        self.dependency_chains = {} # target_concept -> [file_1, file_2, file_3]

    def register_chain(self, concept: str, required_files: List[str]):
        """Registers a known multi-hop path."""
        self.dependency_chains[concept] = required_files

    def evaluate_retrieval(self, concept: str, retrieved_files: List[str]) -> Dict[str, float]:
        """
        Scores how much of the dependency chain was successfully retrieved.
        """
        if concept not in self.dependency_chains:
            return {"chain_completion": 0.0, "files_found": 0}

        required = set(self.dependency_chains[concept])
        retrieved = set(retrieved_files)
        
        found = required.intersection(retrieved)
        completion_ratio = len(found) / len(required) if required else 1.0

        return {
            "chain_completion": completion_ratio,
            "files_found": len(found),
            "total_required": len(required),
            "missing": list(required - retrieved)
        }
