import time

class KernelPipelineVisualizer:
    """
    PHASE 6G: Kernel Pipeline Visualizer
    Waterfall chart of kernel execution, launch overhead, and 
    memory stalls.
    Crucial for identifying synchronization bottlenecks.
    """
    def __init__(self):
        self.events = []

    def record_event(self, name: str, start_time: float, end_time: float, type: str):
        """Records an execution event."""
        self.events.append({
            "name": name,
            "start": start_time,
            "end": end_time,
            "type": type
        })

    def show_waterfall(self):
        """Prints a waterfall-style trace."""
        for e in self.events:
            print(f"[{e['type']}] {e['name']}: {e['end'] - e['start']:.6f}s")
