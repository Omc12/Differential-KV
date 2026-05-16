class ProtectiveRadiusAllocator:
    """
    PHASE 18.9B: Protective Radius Allocator.
    Dynamically adjusts the protection window around anchors and symbolic regions.
    """
    def __init__(self, base_radius=16):
        self.base_radius = base_radius

    def get_radius(self, anchor_type, is_symbolic=False):
        if is_symbolic:
            return self.base_radius * 2 # Symbolic regions get double protection
        if "ANCHOR" in anchor_type:
            return self.base_radius
        return 4 # Default minimal protection
