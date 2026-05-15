import logging
import torch
from typing import List, Dict, Any
from distributed.distributed_kv_fabric import DistributedKVFabric, RemoteCognitionPool
from distributed.remote_hotzone_coordinator import RemoteHotzoneCoordinator
from distributed.cross_gpu_rehydration_engine import CrossGPURehydrationEngine, DistributedSparseWakePipeline
from distributed.network_aware_sparse_scheduler import NetworkAwareSparseScheduler, TopologyAwareRouter
from distributed.distributed_integrity_guard import DistributedIntegrityGuard

class DCSResolver:
    """
    Distributed Cognition Orchestrator (DCS Resolver).
    Coordinates distributed KV virtualization, residency, and scheduling.
    """
    def __init__(self, devices: List[str], topology: Dict[str, List[str]]):
        self.fabric = DistributedKVFabric(devices)
        self.guard = DistributedIntegrityGuard()
        self.hotzone_coord = RemoteHotzoneCoordinator(self.fabric)
        
        self.pools: Dict[str, RemoteCognitionPool] = {
            d: RemoteCognitionPool(self.fabric, d) for d in devices
        }
        
        self.rehydrators: Dict[str, CrossGPURehydrationEngine] = {
            d: CrossGPURehydrationEngine(self.fabric, self.pools[d]) for d in devices
        }
        
        self.scheduler = NetworkAwareSparseScheduler(self.fabric, topology)
        self.router = TopologyAwareRouter(self.scheduler)
        self.logger = logging.getLogger("DCSResolver")

    def register_remote_cognition(self, segment_id: str, device: str, initial_tensor: torch.Tensor):
        """Registers a new cognitive segment in the distributed fabric."""
        self.fabric.register_segment(segment_id, device)
        self.fabric.remote_pools[device][segment_id] = initial_tensor
        self.guard.register_integrity(segment_id, initial_tensor)

    async def resolve_distributed_access(self, segment_id: str, requester_device: str):
        """Resolves an access request for a distributed segment."""
        # 1. Track access for hotzone optimization
        self.hotzone_coord.track_access(segment_id, requester_device)
        
        # 2. Schedule execution or transfer
        target_device = self.router.route_execution(segment_id, requester_device)
        
        # 3. Handle rehydration if needed
        current_residency = self.fabric.get_residency(segment_id)
        if current_residency != requester_device and target_device == requester_device:
            rehydrator = self.rehydrators[requester_device]
            tensor = await rehydrator.rehydrate_remote_async(segment_id)
            
            # 4. Validate integrity
            self.guard.validate_continuity(segment_id, tensor)
            return tensor
        
        return None # In a real system, this would trigger remote execution or wait for migration

    def get_dcs_metrics(self) -> Dict[str, Any]:
        """Aggregates metrics from all DCS modules."""
        metrics = {}
        metrics.update(self.hotzone_coord.get_metrics())
        metrics.update(self.guard.get_integrity_metrics())
        metrics.update(self.scheduler.get_routing_metrics())
        
        # Aggregate rehydration metrics
        avg_latencies = [r.get_recovery_metrics()["avg_rehydration_latency"] for r in self.rehydrators.values()]
        metrics["cross_gpu_rehydration_latency"] = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0
        
        return metrics
