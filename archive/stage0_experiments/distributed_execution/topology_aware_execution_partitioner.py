from typing import Dict, List, Any, Optional, Tuple
import logging

class TopologyAwareExecutionPartitioner:
    """
    Partitions sparse compute graphs based on hardware topology to minimize communication.
    """
    def __init__(self, devices: List[str], topology: Dict[str, List[str]]):
        self.devices = devices
        self.topology = topology # device -> connected_devices
        self.partition_map: Dict[str, str] = {} # task_id -> device
        self.logger = logging.getLogger("TopologyAwareExecutionPartitioner")

    def partition_graph(self, tasks: List[Dict[str, Any]], dependencies: List[Tuple[str, str]]) -> Dict[str, str]:
        """Partitions tasks across devices to maximize locality."""
        # Simple locality-preserving partitioning:
        # Group tasks with their parents if they are on the same device.
        # Otherwise, pick the best connected device.
        
        current_allocations = {d: 0 for d in self.devices}
        
        for task in tasks:
            tid = task["id"]
            # Simplified heuristic: pick device with least tasks for balance
            # In real DSE, we would look at dependency locality
            best_device = min(current_allocations, key=current_allocations.get)
            self.partition_map[tid] = best_device
            current_allocations[best_device] += 1
            
        self.logger.info(f"Partitioned {len(tasks)} tasks across {len(self.devices)} devices.")
        return self.partition_map

    def get_partition_efficiency(self, edges: List[Tuple[str, str]]) -> float:
        """Calculates efficiency based on local vs remote edges."""
        if not edges:
            return 1.0
        
        local_edges = 0
        for p, c in edges:
            if self.partition_map.get(p) == self.partition_map.get(c):
                local_edges += 1
        
        efficiency = local_edges / len(edges)
        return efficiency

