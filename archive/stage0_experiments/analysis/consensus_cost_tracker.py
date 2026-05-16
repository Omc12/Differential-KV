class ConsensusCostTracker:
    def __init__(self):
        self.consensus_events = 0
    def add_event(self):
        self.consensus_events += 1
