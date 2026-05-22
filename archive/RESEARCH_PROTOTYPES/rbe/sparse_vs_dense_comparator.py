
from typing import Dict, Any

class SparseVsDenseComparator:
    """
    PHASE 24.2: Sparse vs Dense Comparator (RBE).
    Provides direct baseline comparisons.
    """
    def __init__(self):
        self.dense_stats = {}
        self.sparse_stats = {}
        
    def record_dense(self, metrics: Dict[str, Any]):
        self.dense_stats = metrics
        
    def record_sparse(self, metrics: Dict[str, Any]):
        self.sparse_stats = metrics
        
    def get_comparison(self) -> Dict[str, Any]:
        comparison = {}
        for key in ["tps", "vram_gb", "latency_ms"]:
            d_val = self.dense_stats.get(key, 1.0)
            s_val = self.sparse_stats.get(key, 1.0)
            
            if key == "tps":
                gain = (s_val / d_val) if d_val > 0 else 0.0
            else:
                gain = (d_val / s_val) if s_val > 0 else 0.0 # Reduction is gain for vram/latency
                
            comparison[f"{key}_gain"] = gain
            
        return comparison
