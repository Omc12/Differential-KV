class DomainGeneralizationTracker:
    """Tracks performance and success rates across different domains."""
    def __init__(self):
        self.stats = {}

    def record_result(self, domain, success, tps):
        if domain not in self.stats:
            self.stats[domain] = {"total": 0, "successes": 0, "tps_sum": 0.0}
        
        self.stats[domain]["total"] += 1
        if success:
            self.stats[domain]["successes"] += 1
        self.stats[domain]["tps_sum"] += tps

    def get_summary(self):
        summary = {}
        for domain, s in self.stats.items():
            summary[domain] = {
                "success_rate": s["successes"] / s["total"] if s["total"] > 0 else 0,
                "avg_tps": s["tps_sum"] / s["total"] if s["total"] > 0 else 0
            }
        return summary
