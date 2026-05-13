import torch

class HotspotAnchorAllocator:
    def allocate_hotspot_anchors(self, hotspots: torch.Tensor, current_anchors: torch.Tensor) -> torch.Tensor:
        if hotspots.numel() == 0: return current_anchors
        extra = []
        for idx in hotspots:
            extra.append(idx - 1); extra.append(idx + 1)
        extra_tensor = torch.tensor(extra, device=hotspots.device, dtype=torch.long)
        extra_tensor = extra_tensor[extra_tensor >= 0]
        return torch.cat([current_anchors, extra_tensor]).unique()
