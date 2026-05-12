import torch
import json
from typing import Dict, List
import time

class LiveManifoldEvolution:
    """
    Tracks attractor birth/death and manifold lineage in real-time.
    Generates stability lineage graphs.
    """
    def __init__(self):
        self.evolution_log = []
        self.active_manifolds = set()
        
    def record_event(self, event_type: str, manifold_id: str, metadata: Dict):
        """
        Records events: 'birth', 'death', 'stabilization', 'mutation'.
        """
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "id": manifold_id,
            "metadata": metadata
        }
        self.evolution_log.append(event)
        
        if event_type == "birth":
            self.active_manifolds.add(manifold_id)
        elif event_type == "death":
            self.active_manifolds.remove(manifold_id)
            
    def export_evolution_trace(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.evolution_log, f, indent=2)
            
    def get_lineage_stats(self) -> Dict:
        births = sum(1 for e in self.evolution_log if e["type"] == "birth")
        deaths = sum(1 for e in self.evolution_log if e["type"] == "death")
        return {
            "total_births": births,
            "total_deaths": deaths,
            "net_growth": births - deaths,
            "turnover_rate": deaths / (births + 1e-6)
        }
