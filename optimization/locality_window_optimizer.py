class LocalityWindowOptimizer:
    def __init__(self, window_size=512):
        self.window_size = window_size
        self.current_window_anchors = set()

    def optimize_window(self, active_anchors):
        # Keep temporally and spatially local anchors active
        optimized_anchors = list(self.current_window_anchors.union(set(active_anchors)))
        # Evict oldest if exceeding capacity
        self.current_window_anchors = set(active_anchors)
        return optimized_anchors
