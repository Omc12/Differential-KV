import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class FederationIsolationLayer:
    """
    In a distributed swarm, prevents a crashing or corrupted agent from taking down
    the rest of the mesh. Severes connections dynamically if bad behavior is detected.
    """
    def __init__(self):
        self.node_trust_scores: Dict[str, float] = {}
        self.isolation_threshold = 0.3

    def register_node(self, node_id: str):
        self.node_trust_scores[node_id] = 1.0

    def report_anomaly(self, source_node: str, target_node: str):
        """Reduces the trust score of a node exhibiting anomalous behavior."""
        if target_node in self.node_trust_scores:
            self.node_trust_scores[target_node] -= 0.2
            logger.warning(f"Node {source_node} reported anomaly from {target_node}. Trust: {self.node_trust_scores[target_node]}")
            
            if self.node_trust_scores[target_node] < self.isolation_threshold:
                self._isolate_node(target_node)

    def _isolate_node(self, node_id: str):
        logger.critical(f"ISOLATION TRIGGERED: Node {node_id} severed from federation.")
        del self.node_trust_scores[node_id]

    def get_trusted_peers(self) -> List[str]:
        return [node for node, score in self.node_trust_scores.items() if score >= self.isolation_threshold]
