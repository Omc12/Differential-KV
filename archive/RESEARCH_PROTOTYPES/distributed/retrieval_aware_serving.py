import torch
from typing import List, Dict

class RetrievalAwareServing:
    """
    Coordinates distributed sparse serving.
    Ensures that retrieval-critical KV blocks are synchronized across nodes.
    """
    def __init__(self, node_id: int, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.local_hotset = set()

    def handle_request(self, query: torch.Tensor, context_shards: List[int]):
        """
        Determines which nodes host the necessary KV shards for a request.
        """
        responsible_nodes = [shard % self.total_nodes for shard in context_shards]
        
        # If this node is responsible for any shard, it processes it
        # Otherwise, it routes the request or pre-fetches from other nodes
        
        return responsible_nodes

    def sync_hotset(self, global_hotset: Set[int]):
        """
        Synchronizes retrieval-critical indices across the cluster.
        """
        self.local_hotset = global_hotset
