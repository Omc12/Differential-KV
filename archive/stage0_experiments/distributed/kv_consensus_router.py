import torch

class KVConsensusRouter:
    """
    Routes queries to the optimal node based on KV shard locality.
    Maintains a consensus of which nodes hold which sparse context blocks.
    """
    def __init__(self, cluster_map: Dict[int, List[int]]):
        self.cluster_map = cluster_map # node_id -> list of block_ids

    def route_query(self, block_ids: List[int]) -> int:
        """
        Finds the node with the highest coverage for the requested blocks.
        """
        best_node = -1
        max_coverage = -1
        
        for node_id, hosted_blocks in self.cluster_map.items():
            coverage = len(set(block_ids).intersection(set(hosted_blocks)))
            if coverage > max_coverage:
                max_coverage = coverage
                best_node = node_id
                
        return best_node

    def update_map(self, node_id: int, blocks: List[int]):
        self.cluster_map[node_id] = blocks
