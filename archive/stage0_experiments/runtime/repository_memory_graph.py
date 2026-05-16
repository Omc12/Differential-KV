from typing import Dict, List

class RepositoryMemoryGraph:
    """
    Builds a topological map of KV retention across the repository.
    Visualizes which files are "hot" in the cache.
    """
    def __init__(self):
        self.nodes = {} # file_id -> retention_score

    def update_retention(self, file_id: str, score: float):
        self.nodes[file_id] = score

    def get_topology(self) -> Dict[str, float]:
        return self.nodes

    def get_average_retention(self):
        if not self.nodes: return 0.0
        return sum(self.nodes.values()) / len(self.nodes)
