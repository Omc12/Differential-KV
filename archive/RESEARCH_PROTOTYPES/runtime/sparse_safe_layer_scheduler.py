"""
STAGE 2 - HSZ: Sparse-Safe Layer Scheduler
Phase 39.3 - Hybrid Semantic Zoning

Assigns per-layer execution modes based on observed zone classification.
Sparse-safe layers stay sparse. Dense-critical layers get hybrid or dense.
No uniform policy.
"""
from typing import Dict, Optional

from runtime.hybrid_semantic_zone_mapper import HybridSemanticZoneMapper
from runtime.dense_criticality_detector import DenseCriticalityDetector


class SparseSafeLayerScheduler:
    """
    Provides per-layer mode recommendations based on zone map and criticality detector.
    This is the authoritative scheduling layer for HSZ phase.
    """
    def __init__(
        self,
        zone_mapper: HybridSemanticZoneMapper,
        criticality_detector: DenseCriticalityDetector,
    ):
        self.zone_mapper = zone_mapper
        self.criticality_detector = criticality_detector

    def get_mode(self, layer_idx: int) -> str:
        """
        Returns recommended execution mode for a given layer.
        Priority: dense_critical > repair_sensitive > sparse_safe > default hybrid.
        """
        # Hard override: if criticality detector flags this layer, force hybrid
        if self.criticality_detector.is_dense_critical(layer_idx):
            return "hybrid"

        classification = self.zone_mapper.classify_layer(layer_idx)

        if classification == "sparse_safe":
            return "sparse"
        elif classification == "dense_critical":
            return "hybrid"
        elif classification == "repair_sensitive":
            return "repair_hybrid"  # hybrid with mandatory repair pass
        else:
            # Undetermined: default conservative (hybrid) until data available
            return "hybrid"

    def get_full_schedule(self, num_layers: int) -> Dict[int, str]:
        return {layer_idx: self.get_mode(layer_idx) for layer_idx in range(num_layers)}

    def sparse_safe_count(self, num_layers: int) -> int:
        return sum(
            1 for l in range(num_layers) if self.get_mode(l) == "sparse"
        )
