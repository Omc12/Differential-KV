"""
Dense Reconstruction Tracker

Tracks exactly how often dense tensors are reconstructed, where, how long, and FLOPs.
"""
import time

class DenseReconstructionTracker:
    def __init__(self):
        self.metrics = {
            "dense_reconstruction_count": 0,
            "reconstruction_latency_ms": 0.0,
            "reconstruction_flop_estimate": 0,
            "dense_decode_pct": 0.0,
            "reconstruction_sites": []
        }
        self.total_decode_steps = 0
        
    def log_reconstruction(self, site, duration_ms, flops):
        self.metrics["dense_reconstruction_count"] += 1
        self.metrics["reconstruction_latency_ms"] += duration_ms
        self.metrics["reconstruction_flop_estimate"] += flops
        if site not in self.metrics["reconstruction_sites"]:
            self.metrics["reconstruction_sites"].append(site)
            
    def step_decode(self, is_dense):
        self.total_decode_steps += 1
        if is_dense:
            self.metrics["dense_decode_pct"] = (
                (self.metrics["dense_decode_pct"] * (self.total_decode_steps - 1) + 100) 
                / self.total_decode_steps
            )
        else:
            self.metrics["dense_decode_pct"] = (
                (self.metrics["dense_decode_pct"] * (self.total_decode_steps - 1)) 
                / self.total_decode_steps
            )
            
    def get_report(self):
        return self.metrics
