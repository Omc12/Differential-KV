class SymbolicDegradationProfiler:
    """Profiles which symbolic types fail first as context scales."""
    def __init__(self):
        self.profile = {}

    def record_attempt(self, domain, ctx, success):
        if domain not in self.profile:
            self.profile[domain] = {}
        if ctx not in self.profile[domain]:
            self.profile[domain][ctx] = {"success": 0, "total": 0}
        
        self.profile[domain][ctx]["total"] += 1
        if success:
            self.profile[domain][ctx]["success"] += 1

    def get_degradation_curves(self):
        curves = {}
        for domain, ctx_data in self.profile.items():
            curves[domain] = {ctx: d["success"]/d["total"] for ctx, d in ctx_data.items()}
        return curves
