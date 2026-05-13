"""
validation/retrieval_recall_measure.py

Phase 12.5A: Retrieval Recall Measure
Quantifies how many of the necessary ground-truth facts/anchors were
successfully retrieved during a query.
"""

from typing import List, Set

class RetrievalRecallMeasure:
    """
    Computes recall for sparse memory retrieval.
    Recall = (Relevant Items Retrieved) / (Total Relevant Items)
    """
    
    @staticmethod
    def compute_recall(retrieved_ids: List[int], ground_truth_ids: List[int]) -> float:
        if not ground_truth_ids:
            return 1.0 # If nothing was relevant, returning anything or nothing is 100% recall
            
        retrieved_set = set(retrieved_ids)
        truth_set = set(ground_truth_ids)
        
        intersection = retrieved_set.intersection(truth_set)
        
        recall = len(intersection) / len(truth_set)
        return recall

    @staticmethod
    def measure_batch_recall(queries_results: List[dict]) -> float:
        """
        Expects a list of dicts: {'retrieved': [1,2,3], 'truth': [2,3,4]}
        """
        recalls = []
        for res in queries_results:
            r = RetrievalRecallMeasure.compute_recall(res['retrieved'], res['truth'])
            recalls.append(r)
            
        return sum(recalls) / len(recalls) if recalls else 0.0
