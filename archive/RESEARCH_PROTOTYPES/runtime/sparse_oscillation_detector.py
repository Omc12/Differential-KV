import numpy as np

class SparseOscillationDetector:
    """
    Detects rapid fluctuations in sparse mask density.
    """
    def __init__(self, threshold: float = 0.05):
        self.densities = []
        self.threshold = threshold

    def log_density(self, density: float):
        self.densities.append(density)

    def get_oscillation_count(self) -> int:
        if len(self.densities) < 3:
            return 0
        
        diffs = np.diff(self.densities)
        # Count sign changes in diffs (direction changes)
        oscillations = sum(1 for i in range(len(diffs)-1) if np.sign(diffs[i]) != np.sign(diffs[i+1]) and abs(diffs[i]) > self.threshold)
        return oscillations
