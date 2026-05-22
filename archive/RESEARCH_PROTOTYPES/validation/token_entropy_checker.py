import torch
import math
from collections import Counter

class TokenEntropyChecker:
    """
    Checks the entropy of generated tokens to detect deterministic synthetic loops.
    Real language has characteristic entropy ranges.
    """
    def __init__(self):
        pass

    def calculate_entropy(self, tokens: list) -> float:
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        total = len(tokens)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy

    def is_real_language(self, tokens: list) -> bool:
        entropy = self.calculate_entropy(tokens)
        # Synthetic loops like 0,1,2,0,1,2... have low entropy
        # Real language tokens usually have entropy > 3.0 for reasonable sequence lengths
        return entropy > 2.0 or len(set(tokens)) > len(tokens) * 0.1
