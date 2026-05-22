class SNREfficiencyTracker:
    def __init__(self):
        self.snr_gain = 0.0
    def record_gain(self, gain: float):
        self.snr_gain += gain
