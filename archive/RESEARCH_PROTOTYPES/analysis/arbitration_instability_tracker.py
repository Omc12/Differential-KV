class ArbitrationInstabilityTracker:
    """Tracks regions where decoder trust arbitration becomes unstable."""
    def __init__(self):
        self.instability_log = []

    def log_instability(self, ctx, domain, confidence_variance):
        if confidence_variance > 0.5:
            self.instability_log.append({
                "ctx": ctx,
                "domain": domain,
                "variance": confidence_variance
            })

    def get_instability_regions(self):
        return self.instability_log
