class SemanticPartitionRegistry:
    """
    PHASE 18.9A: Semantic Partition Registry.
    Persistent store for structural anchors across chunks.
    """
    def __init__(self):
        self.registry = {} # chunk_idx -> partitions

    def register_chunk(self, chunk_idx, partitions):
        self.registry[chunk_idx] = partitions

    def get_global_anchors(self):
        # Flatten registry into global indices if needed
        all_anchors = []
        for chunk_idx, partitions in self.registry.items():
            for p in partitions:
                all_anchors.append((chunk_idx, p))
        return all_anchors
