import numpy as np

class BenchmarkUncertaintyEstimator:
    """
    Estimates the uncertainty (error bars) for benchmark results.
    Uses standard deviation and standard error of the mean.
    """
    def __init__(self):
        pass

    def estimate_uncertainty(self, results: list):
        if not results:
            return 0
        std_err = np.std(results) / np.sqrt(len(results))
        return std_err
