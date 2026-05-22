class ResonanceCostTracker:
    def __init__(self):
        self.activations = 0
    def add_activation(self):
        self.activations += 1
