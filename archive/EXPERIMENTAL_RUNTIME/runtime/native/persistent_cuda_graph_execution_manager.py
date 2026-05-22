import torch
import time
import logging

class PersistentCUDAGraphExecutionManager:
    """
    STAGE 2 CDBE: Persistent CUDA Graph Execution Manager.
    Reduces micro-launch fragmentation via stable decode graph reuse and launch amortization.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.logger = logging.getLogger("CUBEGraphManager")
        self.persistence_active = True
        
        # Graph Storage
        self.graphs = {} # batch_size -> graph
        self.static_inputs = {}
        self.static_outputs = {}
        
        # Counters
        self.graph_reuse_count = 0
        self.graph_capture_count = 0
        self.launch_amortization_factor = 0
        
        # Metrics
        self.start_ts = time.time()
        self.last_launch_ts = 0

    def capture_persistent_graph(self, func, batch_size, *args, **kwargs):
        """
        Captures a function into a persistent CUDA graph for a specific batch size.
        """
        if batch_size in self.graphs:
            self.graph_reuse_count += 1
            return self.graphs[batch_size]

        self.logger.info(f"Capturing NEW persistent decode graph for batch_size={batch_size}")
        
        # Warm-up
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                func(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            func(*args, **kwargs)
            
        self.graphs[batch_size] = g
        self.graph_capture_count += 1
        return g

    def replay_graph(self, batch_size):
        """Replays a previously captured graph."""
        if batch_size in self.graphs:
            self.graphs[batch_size].replay()
            self.graph_reuse_count += 1
            
            # Update launch amortization metric
            now = time.time()
            if self.last_launch_ts > 0:
                interval = (now - self.last_launch_ts) * 1000
                # We want interval to be smaller than the actual compute time to show amortization
                self.launch_amortization_factor = 0.95 * self.launch_amortization_factor + 0.05 * (1.0 / max(0.1, interval))
            self.last_launch_ts = now
            return True
        return False

    def get_metrics(self):
        duration = time.time() - self.start_ts
        return {
            "graph_reuse_count": self.graph_reuse_count,
            "graph_capture_count": self.graph_capture_count,
            "graph_reuse_ratio": self.graph_reuse_count / max(1, self.graph_capture_count),
            "launch_amortization_factor": self.launch_amortization_factor,
            "persistence_duration_sec": duration,
            "active_graphs": len(self.graphs)
        }
