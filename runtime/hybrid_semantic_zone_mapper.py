"""
STAGE 2 - HSZ: Hybrid Semantic Zone Mapper
Phase 39.3 - Hybrid Semantic Zoning

Maps per-layer and per-head semantic safety.
Identifies sparse-safe regions vs dense-critical regions.
Does NOT enforce policy — only maps observed drift reality.
"""
import threading
import time
from collections import defaultdict
from typing import Dict, List, Any, Tuple


class HybridSemanticZoneMapper:
    """
    Maps which layers/heads are sparse-safe vs dense-critical
    based on observed KL divergence from live dense-reference comparisons.
    """
    # A layer whose mean drift consistently exceeds this threshold is dense-critical
    DENSE_CRITICAL_DRIFT  = 0.15
    # A layer whose mean drift is below this is sparse-safe
    SPARSE_SAFE_DRIFT     = 0.05
    # Minimum samples required before classifying a layer
    MIN_SAMPLES           = 4

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock() # Use RLock to avoid deadlock in get_zone_map
        # per-layer rolling drift history (list of floats)
        self._layer_drift: Dict[int, List[float]] = defaultdict(list)
        # per-head drift contribution: layer -> head -> list[float]
        self._head_drift: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    def record_layer_drift(self, layer_idx: int, kl_div: float):
        with self._lock:
            self._layer_drift[layer_idx].append(kl_div)

    def record_head_drift(self, layer_idx: int, head_idx: int, kl_div: float):
        with self._lock:
            self._head_drift[layer_idx][head_idx].append(kl_div)

    def get_dense_critical_heads(self, layer_idx: int) -> List[int]:
        """Returns list of head indices in a layer that are dense-critical."""
        critical_heads = []
        with self._lock:
            layer_heads = self._head_drift.get(layer_idx, {})
            for head_idx, hist in layer_heads.items():
                if len(hist) < self.MIN_SAMPLES:
                    continue
                mean = sum(hist) / len(hist)
                if mean >= self.DENSE_CRITICAL_DRIFT:
                    critical_heads.append(head_idx)
        return critical_heads

    def get_layer_mean_drift(self, layer_idx: int) -> float:
        with self._lock:
            hist = self._layer_drift.get(layer_idx, [])
            return sum(hist) / len(hist) if hist else 0.0

    def classify_layer(self, layer_idx: int) -> str:
        """Returns 'sparse_safe', 'dense_critical', or 'undetermined'."""
        with self._lock:
            hist = self._layer_drift.get(layer_idx, [])
            if len(hist) < self.MIN_SAMPLES:
                return "undetermined"
            mean = sum(hist) / len(hist)
            if mean >= self.DENSE_CRITICAL_DRIFT:
                return "dense_critical"
            elif mean <= self.SPARSE_SAFE_DRIFT:
                return "sparse_safe"
            else:
                return "repair_sensitive"

    def get_zone_map(self) -> Dict[str, Any]:
        """Returns the current classification of all layers and heads."""
        zones = {}
        critical_heads_map = {}
        with self._lock:
            for layer_idx in range(self.num_layers):
                zones[str(layer_idx)] = self.classify_layer(layer_idx)
                
                critical_heads = self.get_dense_critical_heads(layer_idx)
                if critical_heads:
                    critical_heads_map[str(layer_idx)] = critical_heads
                    
        return {
            "layers": zones,
            "dense_critical_heads": critical_heads_map
        }

    def get_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for layer_idx in range(self.num_layers):
                hist = self._layer_drift.get(layer_idx, [])
                mean = sum(hist) / len(hist) if hist else None
                
                crit_heads = []
                layer_heads = self._head_drift.get(layer_idx, {})
                for h_idx, h_hist in layer_heads.items():
                    if h_hist and (sum(h_hist)/len(h_hist)) >= self.DENSE_CRITICAL_DRIFT:
                        crit_heads.append(h_idx)

                rows.append({
                    "layer": layer_idx,
                    "samples": len(hist),
                    "mean_drift": round(mean, 6) if mean is not None else None,
                    "classification": self.classify_layer(layer_idx),
                    "dense_critical_heads": crit_heads
                })
            return rows
