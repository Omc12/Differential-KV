import torch

class AttentionRunwayAllocator:
    """
    PHASE 19.0B: Attention Runway Allocator.
    Dynamically manages the budget for virtual runways, ensuring they 
    don't exceed memory limits.
    """
    def __init__(self, max_runway_budget: int = 2048):
        self.max_runway_budget = max_runway_budget
        self.current_usage = 0

    def allocate_runway(self, requested_indices: torch.Tensor) -> torch.Tensor:
        """
        Filters requested runway indices to fit within budget.
        Priority given to recent requests (or based on importance).
        """
        num_requested = requested_indices.numel()
        
        if num_requested <= self.max_runway_budget:
            self.current_usage = num_requested
            return requested_indices
            
        # If over budget, we must prioritize. 
        # Here we just take the most recent ones (highest indices)
        # Assuming requested_indices are sorted by time/position
        sorted_indices, _ = torch.sort(requested_indices, descending=True)
        allocated = sorted_indices[:self.max_runway_budget]
        
        self.current_usage = self.max_runway_budget
        return allocated

    def get_overhead_metrics(self):
        return {
            "runway_usage": self.current_usage,
            "runway_utilization": self.current_usage / self.max_runway_budget if self.max_runway_budget > 0 else 0
        }
