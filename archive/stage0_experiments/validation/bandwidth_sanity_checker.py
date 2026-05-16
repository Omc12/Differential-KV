import torch

class BandwidthSanityChecker:
    """
    Validates that reported bandwidth gains are REAL and not due to 
    hidden persistence or caching.
    """
    def __init__(self):
        pass

    def check_bandwidth(self, reported_gb: float, theoretical_min: float):
        """
        Rejects gains that exceed physical hardware limits.
        """
        # If reported bandwidth reduction is 99% but we still move 100MB, 
        # something is wrong.
        
        if reported_gb < theoretical_min:
            return "REJECTED: Gain exceeds physical limits (likely hidden caching)"
            
        return "PASS"
