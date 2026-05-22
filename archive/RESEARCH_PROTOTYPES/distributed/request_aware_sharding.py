from typing import List, Dict

class RequestAwareSharding:
    """
    Dynamically shards KV caches based on real-time request patterns.
    Optimizes for cases where multiple requests share the same long-context prefix.
    """
    def __init__(self):
        self.prefix_map: Dict[str, List[int]] = {}

    def register_request(self, request_id: int, prefix: str):
        """Track which requests share the same prefix."""
        if prefix not in self.prefix_map:
            self.prefix_map[prefix] = []
        self.prefix_map[prefix].append(request_id)

    def get_sharding_plan(self) -> Dict[str, List[int]]:
        """
        Returns a plan for which prefixes should be cached on which nodes.
        Requests sharing a prefix should ideally be routed to the same node 
        to leverage prefix-caching.
        """
        return self.prefix_map
