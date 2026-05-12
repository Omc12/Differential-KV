import logging

logger = logging.getLogger(__name__)

class DistributedManifoldTracing:
    """
    Traces the lineage and cross-node contamination of reasoning manifolds.
    Answers the question: 'Which agent originated this cognitive behavior?'
    """
    def __init__(self):
        self.manifold_lineage = {}

    def record_exchange(self, source_node: str, target_node: str, manifold_id: str):
        if manifold_id not in self.manifold_lineage:
            self.manifold_lineage[manifold_id] = []
            
        self.manifold_lineage[manifold_id].append({
            "from": source_node,
            "to": target_node,
            "timestamp": "now"
        })
        logger.debug(f"Traced: {manifold_id} from {source_node} to {target_node}")

    def get_lineage(self, manifold_id: str) -> list:
        return self.manifold_lineage.get(manifold_id, [])
