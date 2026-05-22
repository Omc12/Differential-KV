import numpy as np

class ReproducibilityEnforcer:
    def __init__(self, variance_threshold=0.05):
        self.variance_threshold = variance_threshold

    def check_variance(self, run_results: list) -> bool:
        """
        Takes a list of identical runs and ensures variance is within acceptable limits.
        """
        mean = np.mean(run_results)
        std_dev = np.std(run_results)
        coeff_var = std_dev / mean
        
        print(f"Mean: {mean:.4f}, StdDev: {std_dev:.4f}, CV: {coeff_var:.4f}")
        
        if coeff_var > self.variance_threshold:
            print(f"FAILED REPRODUCIBILITY: Variance {coeff_var:.4f} > {self.variance_threshold}")
            return False
        return True
