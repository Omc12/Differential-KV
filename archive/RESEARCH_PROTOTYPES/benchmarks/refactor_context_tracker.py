class RefactorContextTracker:
    """
    Tracks context retention during iterative code refactoring sessions.
    Ensures structural code anchors survive multiple edit-apply loops.
    """
    def __init__(self):
        self.edit_history = []
        self.anchor_survival = {} # anchor_id -> bool

    def record_edit(self, file_id: str, diff_len: int):
        self.edit_history.append({"file": file_id, "size": diff_len})

    def audit_anchors(self, active_anchors: set):
        # Update survival status
        for aid in self.anchor_survival:
            if aid not in active_anchors:
                self.anchor_survival[aid] = False
                print(f"CRITICAL: Anchor {aid} collapsed during refactor!")
