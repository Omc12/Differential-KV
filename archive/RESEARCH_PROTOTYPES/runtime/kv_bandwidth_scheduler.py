import torch
import time
from typing import Dict, List

class KVBandwidthScheduler:
    """
    PHASE 6B: KV Bandwidth Scheduler
    Coordinates the migration of KV blocks between memory tiers (VRAM/RAM).
    Prioritizes 'retrieval-hot' blocks to ensure zero-stall execution.
    """
    def __init__(self, vram_limit_gb: float, pcie_bandwidth_gb_s: float = 16.0):
        self.vram_limit = vram_limit_gb * 1024**3
        self.pcie_bandwidth = pcie_bandwidth_gb_s * 1024**3
        self.current_vram_usage = 0
        self.transfer_queue = []

    def schedule_migration(self, blocks: List[Dict], priority_scores: torch.Tensor):
        """
        Schedules blocks for migration based on priority.
        Blocks with high priority scores stay in VRAM.
        Low priority blocks are offloaded to RAM.
        """
        # Sort blocks by priority
        sorted_indices = torch.argsort(priority_scores, descending=True)
        
        to_vram = []
        to_ram = []
        
        # Simple greedy allocation
        vram_budget = self.vram_limit
        for idx in sorted_indices:
            block_size = blocks[idx]['size']
            if vram_budget >= block_size:
                to_vram.append(idx)
                vram_budget -= block_size
            else:
                to_ram.append(idx)
                
        return to_vram, to_ram

    def estimate_migration_latency(self, size_bytes: int) -> float:
        """Estimates time required to move data over PCIe."""
        return size_bytes / self.pcie_bandwidth
