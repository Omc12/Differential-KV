from typing import Dict, Any

class LayerSpecificSemanticRecovery:
    """
    STAGE 2 - SRI: Layer-Specific Semantic Recovery
    Applies stronger repair participation and tighter safety margins to later,
    more semantically fragile transformer layers.
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        
    def get_layer_safety_margin(self, layer_idx: int) -> float:
        """
        Returns a safety multiplier [1.0, 2.0]. 
        Later layers require higher confidence to suppress fallback.
        """
        depth = layer_idx / max(self.num_layers - 1, 1)
        # Deep layers are exponentially more fragile
        return 1.0 + (depth ** 2)
        
    def requires_forced_repair(self, layer_idx: int, base_confidence: float) -> bool:
        """
        Forces semantic repair activation on critical late layers 
        if base confidence is slightly shaky.
        """
        depth = layer_idx / max(self.num_layers - 1, 1)
        if depth > 0.75 and base_confidence < 0.90:
            return True
        return False
