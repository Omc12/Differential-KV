class QueueCollapseDestroyer:
    """
    Rejects scenarios where the orchestration queue collapses 
    due to starvation or deadlock.
    """
    def __init__(self, max_wait_s: float = 10.0):
        self.max_wait = max_wait_s

    def audit_queue(self, oldest_request_wait: float):
        if oldest_request_wait > self.max_wait:
            print(f"CRITICAL FAILURE: Queue collapse! Oldest request waited {oldest_request_wait}s")
            return False
        return True
