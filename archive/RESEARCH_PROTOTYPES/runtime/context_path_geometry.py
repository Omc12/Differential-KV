from typing import List, Dict, Any
import numpy as np

class ContextPathGeometry:
    """
    Analyzes structural context paths to predict retrieval requirements.
    Strictly structural, NO manifold narratives.
    """
    def __init__(self):
        self.paths = []

    def update_paths(self, current_retrieval_indices: List[int]):
        """Updates the internal model of retrieval paths."""
        self.paths.append(current_retrieval_indices)
        if len(self.paths) > 100:
            self.paths.pop(0)

    def predict_next_indices(self) -> List[int]:
        """
        Predicts which indices will be accessed next based on path history.
        Uses simple temporal extrapolation.
        """
        if len(self.paths) < 2:
            return []
            
        last_indices = set(self.paths[-1])
        prev_indices = set(self.paths[-2])
        
        # Simple prediction: indices that were newly added in the last step
        delta = last_indices - prev_indices
        return list(delta)

    def get_path_metrics(self) -> Dict[str, Any]:
        return {
            "history_length": len(self.paths),
            "avg_path_size": np.mean([len(p) for p in self.paths]) if self.paths else 0
        }
