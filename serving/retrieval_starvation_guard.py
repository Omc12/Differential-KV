class RetrievalStarvationGuard:
    """
    Rejects scenarios where sparse retrieval is starved by heavy compute
    or bandwidth-intensive concurrent requests.
    """
    def __init__(self, min_retrieval_bandwidth: float = 100.0): # MB/s
        self.min_bw = min_retrieval_bandwidth

    def is_safe(self, current_available_bw: float):
        if current_available_bw < self.min_bw:
            print("CRITICAL: Retrieval starvation risk! Throttling requests.")
            return False
        return True
