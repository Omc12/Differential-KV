import torch

class ScalingSanityChecker:
    """
    Ensures that performance gains scale realistically with context length.
    Rejects 'magical' scaling that violates hardware limits.
    """
    def __init__(self):
        pass

    def verify_scaling(self, context_lengths: list, throughputs: list):
        """
        Gains should generally follow a sub-linear or linear trend relative to O(N^2) baseline.
        If throughput INCREASES with context length, something is wrong.
        """
        for i in range(1, len(throughputs)):
            if throughputs[i] > throughputs[i-1] * 1.1: # Allow some jitter
                print(f"REJECT: Unrealistic scaling detected @ {context_lengths[i]}. Throughput increased.")
                return False
        
        print("Scaling sanity: PASS")
        return True
