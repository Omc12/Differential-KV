import torch
from .structured_anchor_scheduler import StructuredAnchorScheduler
from .anchor_collision_reducer import AnchorCollisionReducer
from .hotspot_anchor_allocator import HotspotAnchorAllocator

class AdaptiveAnchorRecovery:
    """
    Final recovery logic for Phase 7A.
    Combines structured scheduling, hotspot allocation, and collision reduction.
    """
    def __init__(self):
        self.scheduler = StructuredAnchorScheduler()
        self.collision_reducer = AnchorCollisionReducer()
        self.hotspot_allocator = HotspotAnchorAllocator()

    def step(self, 
             attn_weights: torch.Tensor, 
             retrieval_indices: torch.Tensor, 
             success_mask: torch.Tensor,
             total_seq_len: int) -> torch.Tensor:
        """
        Executes one recovery step:
        1. Update metrics (density, entropy)
        2. Identify hotspots
        3. Generate structured adaptive anchors
        4. Allocate extra anchors for hotspots
        5. Reduce collisions
        """
        # 1. Update hardware truth metrics
        self.scheduler.update_metrics(attn_weights, retrieval_indices, success_mask, total_seq_len)
        
        # 2. Get structured anchors based on regional modes
        structured_anchors = self.scheduler.get_adaptive_anchors(total_seq_len)
        
        # 3. Identify and recover hotspots
        hotspots = self.scheduler.density_mapper.identify_hotspots()
        recovered_anchors = self.hotspot_allocator.allocate_hotspot_anchors(hotspots, structured_anchors)
        
        # 4. Clean up collisions for GPU efficiency
        final_anchors = self.collision_reducer.reduce_collisions(recovered_anchors)
        
        # Ensure they are within bounds
        final_anchors = torch.clamp(final_anchors, 0, total_seq_len - 1)
        
        return final_anchors.unique()

    def get_status_report(self) -> dict:
        """Detailed status report for empirical validation."""
        telemetry = self.scheduler.get_telemetry()
        return {
            "retrieval_stability": telemetry["avg_retrieval_density"],
            "active_hotspots": telemetry["hotspot_count"],
            "spacing_distribution": telemetry["current_layout"],
            "status": "STABLE" if telemetry["avg_retrieval_density"] > 0.9 else "RECOVERING"
        }
