"""
memory/multi_tier_anchor_residency.py

Phase 12C: Multi-Tier Anchor Residency
Policy-driven management of where anchors reside in the hierarchical memory
system (VRAM, RAM, or Disk).
"""

import torch
from typing import List
from anchor_logic.semantic_anchor_system import SemanticAnchor

class MultiTierAnchorResidency:
    """
    Decides when to offload or pre-fetch anchors based on usage patterns
    and hardware pressure.
    """
    def __init__(self, vram_limit_gb: float = 2.0):
        self.vram_limit = vram_limit_gb
        self.current_vram_usage = 0.0

    def check_vram_pressure(self, anchors: List[SemanticAnchor]) -> bool:
        """Estimates VRAM usage of a list of anchors."""
        usage = 0
        for a in anchors:
            if a.kv_exact is not None:
                # [2, heads, dim] * 2 bytes (FP16)
                usage += a.kv_exact.numel() * 2
        
        self.current_vram_usage = usage / (1024**3)
        return self.current_vram_usage > self.vram_limit

    def enforce_residency(self, l1_anchors: List[SemanticAnchor]):
        """
        Moves low-priority anchors from VRAM to RAM if pressure is high.
        """
        if not self.check_vram_pressure(l1_anchors):
            return

        print(f"[MultiTierAnchorResidency] High VRAM pressure ({self.current_vram_usage:.2f} GB). Offloading...")
        
        # Sort by importance
        l1_anchors.sort(key=lambda x: x.importance_score)
        
        # Offload bottom 20%
        num_to_offload = len(l1_anchors) // 5
        for i in range(num_to_offload):
            anchor = l1_anchors[i]
            if anchor.kv_exact is not None and anchor.kv_exact.is_cuda:
                anchor.kv_exact = anchor.kv_exact.cpu()
                print(f"  Offloaded anchor at {anchor.position} to RAM.")
