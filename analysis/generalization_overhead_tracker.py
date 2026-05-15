class GeneralizationOverheadTracker:
    """Tracks TPS and VRAM overhead introduced by generalization requirements."""
    def __init__(self):
        self.data = []

    def record_metrics(self, mode, ctx, tps, vram, overhead):
        self.data.append({
            "mode": mode,
            "ctx": ctx,
            "tps": tps,
            "vram": vram,
            "overhead": overhead
        })

    def get_overhead_summary(self):
        # Calculate average overhead per mode
        summary = {}
        for d in self.data:
            m = d["mode"]
            if m not in summary:
                summary[m] = {"count": 0, "overhead_sum": 0.0}
            summary[m]["count"] += 1
            summary[m]["overhead_sum"] += d["overhead"]
        
        return {m: v["overhead_sum"]/v["count"] for m, v in summary.items()}
