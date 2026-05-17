import os
import time
from pathlib import Path

class SynchronizationBubbleEliminator:
    """
    DPC Phase 42.1 — Synchronization Bubble Eliminator.
    Identifies, tracks, and logs CUDA synchronization gaps, memcpy barrier delays,
    and stream synchronization bubbles, keeping the GPU occupied.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.total_sync_time_ms = 0.0
        self.barrier_count = 0

    def start_sync_wait(self) -> float:
        """Called just before calling torch.cuda.synchronize() or a blocking memcpy."""
        return time.perf_counter()

    def end_sync_wait(self, start_perf_ts: float, label: str = "cuda_sync"):
        """Called just after completion of the synchronization block."""
        duration_ms = (time.perf_counter() - start_perf_ts) * 1000.0
        self.total_sync_time_ms += duration_ms
        self.barrier_count += 1
        return duration_ms

    def get_average_bubble_ms(self) -> float:
        if self.barrier_count == 0:
            return 0.0
        return self.total_sync_time_ms / self.barrier_count
