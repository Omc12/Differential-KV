class StructuralOverheadMapper:
    """
    PHASE 18.9D: Structural Overhead Mapper.
    Visualizes how reinforcement overhead scales with context depth.
    """
    def __init__(self):
        self.data = []

    def add_data(self, ctx_len, overhead_pct):
        self.data.append((ctx_len, overhead_pct))

    def get_summary(self):
        if not self.data: return "No data."
        return f"Avg Overhead: {sum(o for c, o in self.data)/len(self.data):.2f}%"
