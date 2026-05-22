class MetricConfidenceRanker:
    """
    Ranks the confidence of a metric based on the number of samples and variance.
    Identifies 'shaky' metrics that might be affected by noise.
    """
    def __init__(self):
        pass

    def rank_confidence(self, values: list):
        if not values:
            return "None"
        if len(values) < 5:
            return "Low (Insufficient Samples)"
        
        import numpy as np
        std = np.std(values)
        mean = np.mean(values)
        cv = std / mean if mean > 0 else 0
        
        if cv < 0.05:
            return "High (Stable)"
        if cv < 0.15:
            return "Medium (Variable)"
        return "Low (High Variance)"
