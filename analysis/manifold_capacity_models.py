import numpy as np

class ManifoldCapacityModels:
    """
    Models the 'carrying capacity' of the cognitive manifold.
    How many attractors can coexist before resonance collapse?
    """
    def estimate_capacity(self, d_model: int, sparsity: float) -> int:
        """
        Calculates theoretical capacity based on dimensionality and sparsity.
        N ~ d / log(d) * (1/sparsity)
        """
        capacity = (d_model / np.log(d_model)) * (1.0 / (sparsity + 1e-6))
        return int(capacity)
        
    def check_saturation(self, current_population: int, d_model: int) -> float:
        """
        Returns saturation percentage.
        """
        capacity = self.estimate_capacity(d_model, 0.1)
        return current_population / capacity
