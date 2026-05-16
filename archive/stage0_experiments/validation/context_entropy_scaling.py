import numpy as np

class ContextEntropyScaling:
    def calculate_entropy(self, attention_distribution):
        """
        Calculates the entropy of the attention distribution to measure 'focus'.
        Higher entropy = less focused retrieval.
        """
        distribution = np.array(attention_distribution)
        distribution = distribution[distribution > 0]
        entropy = -np.sum(distribution * np.log2(distribution))
        return entropy

    def test_entropy_collapse(self, context_sizes):
        results = {}
        for ctx in context_sizes:
            # Simulate entropy growth with context size
            simulated_entropy = np.log2(ctx) * 0.4
            results[ctx] = simulated_entropy
        return results
