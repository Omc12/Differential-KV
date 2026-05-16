from typing import Dict, Any

class SparseGovernanceTruthLayer:
    """
    SGC Phase 39.1 RESET: Sparse Governance Truth Layer.
    Separates governance state from semantic correctness.
    """
    def __init__(self):
        self.stats = {
            "total_steps": 0,
            "sparse_governance_active": 0,
            "semantically_preserved_steps": 0,
            "correct_sparse_steps": 0,
            "false_sparse_persistence": 0
        }

    def record_step(self, 
                    is_sparse: bool, 
                    is_semantically_preserved: bool, 
                    confidence: float):
        """
        Correlates governance decision with semantic reality.
        """
        self.stats["total_steps"] += 1
        if is_sparse:
            self.stats["sparse_governance_active"] += 1
            
        if is_semantically_preserved:
            self.stats["semantically_preserved_steps"] += 1
            if is_sparse:
                self.stats["correct_sparse_steps"] += 1
        else:
            if is_sparse:
                # Governance kept it sparse, but the output drifted!
                self.stats["false_sparse_persistence"] += 1

    def get_truth_metrics(self) -> Dict[str, float]:
        total = self.stats["total_steps"] or 1
        return {
            "sparse_persistence_rate": self.stats["sparse_governance_active"] / total,
            "semantic_correctness_rate": self.stats["semantically_preserved_steps"] / total,
            "governance_accuracy": self.stats["correct_sparse_steps"] / (self.stats["sparse_governance_active"] or 1),
            "false_persistence_rate": self.stats["false_sparse_persistence"] / total
        }
