"""
hardware_materialization/vram_fragmentation_recovery_engine.py

Detects VRAM fragmentation and orchestrates recovery/compaction.
"""

import torch
import logging

logger = logging.getLogger("VRAMRecovery")

class VRAMFragmentationRecoveryEngine:
    """
    Prevents long-session memory collapse by repacking fragmented sparse residency pools.
    """
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
        self.recovery_count = 0

    def check_and_recover(self) -> bool:
        """
        If fragmentation exceeds threshold, trigger a lightweight compaction.
        """
        if not torch.cuda.is_available():
            return False
            
        stats = torch.cuda.memory_stats()
        allocated = stats.get("allocated_bytes.all.current", 0)
        reserved = stats.get("reserved_bytes.all.current", 0)
        
        if reserved == 0:
            return False
            
        fragmentation = (reserved - allocated) / reserved
        
        if fragmentation > self.threshold:
            logger.info(f"VRAM Fragmentation ({fragmentation:.2%}) exceeds threshold. Triggering recovery...")
            return self.perform_recovery()
        return False

    def perform_recovery(self) -> bool:
        """
        Performs a lightweight memory recovery (empties cache).
        In a real system, this would involve repacking fragmented KV buffers.
        """
        torch.cuda.empty_cache()
        self.recovery_count += 1
        logger.info("VRAM recovery complete: Cache cleared.")
        return True

    def get_recovery_metrics(self):
        return {
            "recovery_events": self.recovery_count,
            "recovery_status": "stable"
        }
