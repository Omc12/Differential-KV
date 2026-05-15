
import json
from typing import Dict, List, Any

class SparseScalingCurveAnalyzer:
    """
    PHASE 24.3: Sparse Scaling Curve Analyzer (LCS).
    Analyzes TPS, VRAM, and Latency scaling as context length increases.
    """
    def __init__(self):
        self.scaling_data = [] # List of {context_len, dense_tps, sparse_tps, ...}
        
    def add_data_point(self, 
                       context_len: int, 
                       dense_metrics: Dict[str, float], 
                       sparse_metrics: Dict[str, float]):
        """
        Adds a scaling data point for a specific context length.
        """
        point = {
            "context_len": context_len,
            "dense": dense_metrics,
            "sparse": sparse_metrics,
            "tps_ratio": sparse_metrics["tps"] / dense_metrics["tps"] if dense_metrics["tps"] > 0 else 0.0,
            "vram_savings": (dense_metrics["vram_gb"] - sparse_metrics["vram_gb"]) / dense_metrics["vram_gb"] if dense_metrics["vram_gb"] > 0 else 0.0
        }
        self.scaling_data.append(point)
        return point

    def analyze_scaling_trends(self) -> Dict[str, Any]:
        """
        Determines the scaling advantage of sparse vs dense.
        """
        if len(self.scaling_data) < 2:
            return {"status": "insufficient_data"}
            
        # Scaling Advantage Ratio (SAR): Rate of change of sparse TPS / Rate of change of dense TPS
        # As context increases, dense TPS usually drops faster than sparse TPS.
        initial = self.scaling_data[0]
        final = self.scaling_data[-1]
        
        # Lower is "better" for TPS drop (closer to 1.0 means more stable)
        dense_stability = final["dense"]["tps"] / initial["dense"]["tps"]
        sparse_stability = final["sparse"]["tps"] / initial["sparse"]["tps"]
        
        sar = sparse_stability / dense_stability if dense_stability > 0 else 1.0
        
        return {
            "scaling_advantage_ratio": sar,
            "tps_cross_over_point": "detected" if sar > 1.2 else "not_yet",
            "vram_scalability": final["vram_savings"]
        }

    def save_curves(self, path: str):
        with open(path, "w") as f:
            json.dump(self.scaling_data, f, indent=4)
