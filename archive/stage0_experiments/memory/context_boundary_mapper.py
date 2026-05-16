class ContextBoundaryMapper:
    """
    PHASE 18.9A: Context Boundary Mapper.
    Maps detected anchors to semantic partitions to track context transitions.
    """
    def __init__(self):
        self.partitions = [] # List of (start_idx, end_idx, anchor_type)

    def map_boundaries(self, anchor_indices, seq_len):
        """
        Groups tokens between anchors into semantic partitions.
        """
        self.partitions = []
        if len(anchor_indices) == 0:
            self.partitions.append((0, seq_len - 1, "ROOT"))
            return self.partitions

        # First partition from start to first anchor
        if anchor_indices[0] > 0:
            self.partitions.append((0, anchor_indices[0].item() - 1, "PRE_ANCHOR"))

        for i in range(len(anchor_indices)):
            start = anchor_indices[i].item()
            end = anchor_indices[i+1].item() - 1 if i + 1 < len(anchor_indices) else seq_len - 1
            self.partitions.append((start, end, f"ANCHOR_{start}"))
            
        return self.partitions
