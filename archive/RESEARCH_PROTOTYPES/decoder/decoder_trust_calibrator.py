import torch

class DecoderTrustCalibrator:
    """PHASE 19.7A: Calibrates decoder trust based on symbolic confidence."""
    def __init__(self, base_trust: float = 1.0, max_trust: float = 5.0):
        self.base_trust = base_trust
        self.max_trust = max_trust

    def calibrate(self, confidence: float) -> float:
        # Sigmoid-like mapping of confidence to trust multiplier
        if confidence < 0.1:
            return self.base_trust
        # Scale trust based on confidence (0.0 to 1.0)
        trust_multiplier = self.base_trust + (self.max_trust - self.base_trust) * (confidence ** 2)
        return trust_multiplier
