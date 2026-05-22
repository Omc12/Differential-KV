from typing import Dict, List, Any, Optional
import logging
import torch

class NetworkAwareSparseScheduler:
    """
    Communication-Aware Sparse Scheduler.
    Optimizes task placement based on KV residency and network topology.
    """
    def __init__(self, fabric: Any, topology: Dict[str, List[str]]):
        self.fabric = fabric
        self.topology = topology # device -> connected_devices
        self.bandwidth_usage: Dict[str, float] = {d: 0.0 for d in topology.keys()}
        self.max_bandwidth = 10.0 # GB/s
        self.logger = logging.getLogger("NetworkAwareSparseScheduler")

    def schedule_task(self, segment_id: str, requester_device: str) -> str:
        """Determines the best device to execute a task based on KV residency."""
        residency = self.fabric.get_residency(segment_id)
        
        if residency == requester_device:
            return requester_device # Local execution is best
        
        # Check if residency is directly connected to requester
        if residency in self.topology.get(requester_device, []):
            # Connected, but check bandwidth
            if self.bandwidth_usage[residency] < self.max_bandwidth:
                self.bandwidth_usage[residency] += 0.5 # Simulate bandwidth consumption
                return requester_device # Fetch remote and execute locally
        
        # If remote is busy or far, maybe migrate or execute there?
        # For now, prioritize execution where KV resides to minimize transfer
        return residency

    def balance_bandwidth(self):
        """Reduces simulated bandwidth usage over time."""
        for d in self.bandwidth_usage:
            self.bandwidth_usage[d] = max(0.0, self.bandwidth_usage[d] - 1.0)

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Returns scheduler stability and balancing metrics."""
        return {
            "network_scheduler_stability": 1.0 - (sum(self.bandwidth_usage.values()) / (len(self.bandwidth_usage) * self.max_bandwidth)),
            "bandwidth_pressure": self.bandwidth_usage
        }

class TopologyAwareRouter:
    """
    Routes execution requests through the network topology.
    """
    def __init__(self, scheduler: NetworkAwareSparseScheduler):
        self.scheduler = scheduler

    def route_execution(self, segment_id: str, source_device: str) -> str:
        target_device = self.scheduler.schedule_task(segment_id, source_device)
        self.scheduler.logger.info(f"Routing task for {segment_id} from {source_device} to {target_device}")
        return target_device
