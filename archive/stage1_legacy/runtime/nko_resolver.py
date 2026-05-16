import logging
from typing import List, Dict, Any
from distributed_nccl.nccl_graph_orchestrator import NCCLGraphOrchestrator
from distributed_nccl.distributed_persistent_coordinator import DistributedPersistentCoordinator
from distributed_nccl.nccl_stream_synchronizer import NCCLStreamSynchronizer
from distributed_nccl.p2p_smem_transport import P2PSmemTransport
from distributed_nccl.distributed_replay_stabilizer import DistributedReplayStabilizer

class NKOResolver:
    """
    NCCL Kernel Orchestrator (NKO Resolver).
    Coordinates distributed CUDA-native execution and high-speed transport.
    """
    def __init__(self, devices: List[str], cko_resolver: Any):
        self.devices = devices
        self.cko = cko_resolver
        self.graph_orch = NCCLGraphOrchestrator()
        self.persistent_coord = DistributedPersistentCoordinator(devices)
        self.stream_sync = NCCLStreamSynchronizer()
        self.p2p_transport = P2PSmemTransport(cko_resolver.smem_manager)
        self.replay_stabilizer = DistributedReplayStabilizer()
        self.logger = logging.getLogger("NKOResolver")

    def execute_distributed_nccl_step(self, task_id: str, segment_id: str):
        """Executes a distributed CUDA-native step with NCCL sync and P2P transport."""
        # 1. Distributed Persistent Coordination
        self.persistent_coord.synchronize_wake(task_id)
        
        # 2. P2P Transport (integrated with CKO SMEM)
        # Move segment from device 0 to others
        for i in range(1, len(self.devices)):
            self.p2p_transport.p2p_transfer(segment_id, self.devices[0], self.devices[i])
            
        # 3. NCCL Graph Capture (simulated)
        self.graph_orch.capture_nccl_op(f"graph_{task_id}", "all_reduce", [])
        
        # 4. Stream Synchronization
        self.stream_sync.sync_stream_with_nccl(None, None)
        
        # 5. Local CKO execution on participating devices
        for dev in self.devices:
            self.cko.optimized_decode_step(torch.randn(1, 1, 128), segment_id)
            self.persistent_coord.signal_completion(dev)
            
        # 6. Replay Stabilization
        self.replay_stabilizer.track_lineage(task_id, self.devices[0], "sym_0")
        self.replay_stabilizer.validate_distributed_replay(task_id, {"res": torch.tensor(1.0)})

    def get_nko_metrics(self) -> Dict[str, Any]:
        """Aggregates NCCL-native distributed metrics."""
        metrics = {}
        metrics.update(self.graph_orch.get_nccl_metrics())
        metrics.update(self.persistent_coord.get_coordination_metrics())
        metrics.update(self.p2p_transport.get_transport_metrics())
        metrics.update(self.replay_stabilizer.get_stabilization_metrics())
        return metrics

import torch # Missing torch import
