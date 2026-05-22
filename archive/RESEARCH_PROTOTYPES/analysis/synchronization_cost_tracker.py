class SynchronizationCostTracker:
    def __init__(self):
        self.sync_events = 0
    def add_sync(self):
        self.sync_events += 1
