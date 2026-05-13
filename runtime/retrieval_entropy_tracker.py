import numpy as np

class RetrievalEntropyTracker:
    """
    Measures entropy growth in attention patterns to detect collapse.
    High entropy = scattered attention. Zero entropy = collapse to single point.
    """
    def __init__(self):
        self.entropy_history = []

    def calculate_entropy(self, attention_weights: np.ndarray):
        """
        Calculates Shannon entropy of attention weights.
        """
        # Ensure weights are a distribution
        weights = attention_weights / (np.sum(attention_weights) + 1e-9)
        entropy = -np.sum(weights * np.log(weights + 1e-9))
        self.entropy_history.append(entropy)
        return entropy

    def detect_collapse(self, threshold: float = 0.1) -> bool:
        if not self.entropy_history:
            return False
        return self.entropy_history[-1] < threshold
