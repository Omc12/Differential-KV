import networkx as nx
from typing import List, Optional

class SemanticNavigationGraph:
    """
    PHASE 18.7B: Semantic Navigation Graph.
    Tracks transitions between semantic anchors and their associated symbolic capsules.
    Used for routing retrieval toward high-fidelity regions.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_anchor_node(self, anchor_idx: int, metadata: dict = None):
        self.graph.add_node(f"anchor_{anchor_idx}", type="anchor", idx=anchor_idx, **(metadata or {}))

    def add_capsule_node(self, capsule_id: str, metadata: dict = None):
        self.graph.add_node(capsule_id, type="capsule", **(metadata or {}))

    def add_transition(self, from_node: str, to_node: str, weight: float = 1.0):
        self.graph.add_edge(from_node, to_node, weight=weight)

    def get_route_to_fidelity(self, current_anchor_idx: int) -> List[str]:
        """Finds capsules reachable from the current anchor."""
        start_node = f"anchor_{current_anchor_idx}"
        if start_node not in self.graph:
            return []
        
        # Simple BFS to find nearby capsules
        reachable_capsules = []
        for node in nx.descendants(self.graph, start_node):
            if self.graph.nodes[node].get("type") == "capsule":
                reachable_capsules.append(node)
        return reachable_capsules

    def get_graph_stats(self):
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "anchors": len([n for n, d in self.graph.nodes(data=True) if d.get("type") == "anchor"]),
            "capsules": len([n for n, d in self.graph.nodes(data=True) if d.get("type") == "capsule"])
        }
