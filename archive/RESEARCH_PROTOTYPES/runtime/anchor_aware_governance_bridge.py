import logging
from typing import Dict, Any, Optional

class AnchorAwareGovernanceBridge:
    """
    STAGE 2 - SRI: Anchor-Aware Governance Bridge
    Connects sparse governance with anchor integrity and semantic repair confidence.
    Ensures suppression decisions respect underlying reconstruction safety.
    """
    def __init__(self):
        self.logger = logging.getLogger("AnchorGovernanceBridge")
        self._anchor_stability: Dict[int, float] = {}
        self._repair_confidence: Dict[int, float] = {}
        
    def update_anchor_state(self, layer_idx: int, stability_score: float, repair_conf: float):
        """Called by the underlying KV manager/repair system to report anchor health."""
        self._anchor_stability[layer_idx] = stability_score
        self._repair_confidence[layer_idx] = repair_conf
        
    def get_safety_signals(self, layer_idx: int) -> Dict[str, float]:
        """Provides anchor and repair safety signals to the governance layer."""
        return {
            "anchor_stability": self._anchor_stability.get(layer_idx, 1.0),
            "repair_confidence": self._repair_confidence.get(layer_idx, 1.0)
        }
        
    def is_safe_to_suppress(self, layer_idx: int, threshold: float = 0.8) -> bool:
        """
        Determines if the anchor state is healthy enough to even consider 
        suppressing a dense fallback.
        """
        stability = self._anchor_stability.get(layer_idx, 1.0)
        repair_conf = self._repair_confidence.get(layer_idx, 1.0)
        # Both underlying anchor and the repair process must be healthy
        return (stability >= threshold) and (repair_conf >= threshold)
