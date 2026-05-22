"""
Dynamic Batch Window.
Adjusts batch sizes dynamically based on retrieval latency and VRAM pressure.
"""

class DynamicBatchWindow:
    def get_window_size(self, current_vram, base_size=64):
        if current_vram > 0.9:
            return max(1, base_size // 2)
        return base_size
