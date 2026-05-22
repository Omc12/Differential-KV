"""
validation/retrieval_precision_tracker.py

Phase 12.5A: Retrieval Precision Tracker
Tracks the precision of retrieval, ensuring the system isn't just
dumping the entire context to achieve high recall.
"""

from typing import List, Set

class RetrievalPrecisionTracker:
    """
    Computes precision for sparse memory retrieval.
    Precision = (Relevant Items Retrieved) / (Total Items Retrieved)
    """
    
    @staticmethod
    def compute_precision(retrieved_ids: List[int], ground_truth_ids: List[int]) -> float:
        if not retrieved_ids:
            return 0.0 # If we retrieved nothing, precision is 0
            
        retrieved_set = set(retrieved_ids)
        truth_set = set(ground_truth_ids)
        
        intersection = retrieved_set.intersection(truth_set)
        
        precision = len(intersection) / len(retrieved_set)
        return precision

    @staticmethod
    def compute_f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
