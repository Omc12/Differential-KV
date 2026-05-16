class StableTransitionTracker:
    """
    PHASE 18.9A: Stable Transition Tracker.
    Identifies which anchors are stable markers of content transitions.
    """
    def __init__(self):
        self.transition_counts = {}

    def track_transition(self, anchor_type):
        self.transition_counts[anchor_type] = self.transition_counts.get(anchor_type, 0) + 1

    def is_stable(self, anchor_type, threshold=2):
        return self.transition_counts.get(anchor_type, 0) >= threshold
