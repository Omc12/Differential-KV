class CodeAnchorTracker:
    """
    Specifically monitors and protects structural code anchors (e.g., class defs, 
    key function signatures) from sparse eviction.
    """
    def __init__(self):
        self.protected_anchors = set()
        self.eviction_attempts = 0

    def protect_anchor(self, anchor_id: str):
        self.protected_anchors.add(anchor_id)

    def notify_eviction_attempt(self, anchor_id: str):
        if anchor_id in self.protected_anchors:
            self.eviction_attempts += 1
            # Re-insertion logic or block eviction
            return False # Blocked
        return True # Allowed
