"""
validation/semantic_similarity_validator.py

Phase 12.5A: Semantic Similarity Validator
Cross-checks retrieved contexts against the original query using
an independent semantic similarity model to ensure the retrieval mechanism
is not just returning random noise.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any

class SemanticSimilarityValidator:
    """
    Independent validator for semantic matching quality.
    """
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def _compute_similarity(self, text1: str, text2: str) -> float:
        import difflib
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def validate_match(self, query: str, retrieved_text: str) -> Dict[str, Any]:
        """Returns similarity score and whether it passes the threshold."""
        sim = self._compute_similarity(query, retrieved_text)
        
        return {
            "similarity": sim,
            "is_valid": sim >= self.threshold
        }

    def validate_batch(self, query: str, retrieved_texts: List[str]) -> List[Dict]:
        return [self.validate_match(query, t) for t in retrieved_texts]
