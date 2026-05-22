class DensityDriftMonitor:
    """
    Tracks the sparsity ratio over time.
    """
    def __init__(self):
        self.densities = []

    def log_density(self, density: float):
        self.densities.append(density)

    def get_drift(self) -> float:
        if len(self.densities) < 10:
            return 0.0
        # Compare first 10 steps with last 10 steps
        start = sum(self.densities[:10]) / 10
        end = sum(self.densities[-10:]) / 10
        return abs(end - start) / (start + 1e-9)
