class ReanchoringCostTracker:
    def __init__(self):
        self.reanchor_pulses = 0
    def add_pulse(self):
        self.reanchor_pulses += 1
