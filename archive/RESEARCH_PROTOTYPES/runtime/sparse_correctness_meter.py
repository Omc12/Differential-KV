from typing import Dict, Any

class SparseCorrectnessMeter:
    """
    STAGE 2 - SRI: Sparse Correctness Meter
    Distinguishes between total sparse arithmetic and SAFE sparse arithmetic.
    """
    def __init__(self):
        self.stats = {
            "total_tokens": 0,
            "sparse_tokens": 0,
            "semantically_correct_tokens": 0,
            "safe_sparse_tokens": 0,
            "unsafe_sparse_tokens": 0
        }
        
    def record_step(self, is_sparse: bool, is_semantically_correct: bool):
        self.stats["total_tokens"] += 1
        if is_sparse:
            self.stats["sparse_tokens"] += 1
            if is_semantically_correct:
                self.stats["safe_sparse_tokens"] += 1
            else:
                self.stats["unsafe_sparse_tokens"] += 1
                
        if is_semantically_correct:
            self.stats["semantically_correct_tokens"] += 1
            
    def get_metrics(self) -> Dict[str, float]:
        total = max(self.stats["total_tokens"], 1)
        sparse = max(self.stats["sparse_tokens"], 1)
        return {
            "sparse_persistence": self.stats["sparse_tokens"] / total,
            "semantic_correctness": self.stats["semantically_correct_tokens"] / total,
            "safe_sparse_ratio": self.stats["safe_sparse_tokens"] / sparse,
            "unsafe_sparse_ratio": self.stats["unsafe_sparse_tokens"] / sparse
        }
