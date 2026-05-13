import torch
from runtime.adaptive_anchor_system.structured_anchor_scheduler import StructuredAnchorScheduler
from runtime.adaptive_anchor_system.anchor_collision_reducer import AnchorCollisionReducer
from runtime.adaptive_anchor_system.hotspot_anchor_allocator import HotspotAnchorAllocator

class AdaptiveAnchorRecovery:
    def __init__(self):
        self.scheduler = StructuredAnchorScheduler()
        self.collision_reducer = AnchorCollisionReducer()
        self.hotspot_allocator = HotspotAnchorAllocator()

    def step(self, attn: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        self.scheduler.update_metrics(attn, indices, mask, seq_len)
        structured = self.scheduler.get_adaptive_anchors(seq_len)
        hotspots = self.scheduler.density_mapper.identify_hotspots()
        recovered = self.hotspot_allocator.allocate_hotspot_anchors(hotspots, structured)
        return self.collision_reducer.reduce_collisions(recovered)

    def get_status_report(self) -> dict:
        telemetry = self.scheduler.get_telemetry()
        return {
            "retrieval_stability": telemetry["avg_retrieval_density"],
            "active_hotspots": telemetry["hotspot_count"],
            "status": "STABLE" if telemetry["avg_retrieval_density"] > 0.9 else "RECOVERING"
        }
