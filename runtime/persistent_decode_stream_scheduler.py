import time
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class PersistentDecodeStreamScheduler:
    """
    SGC Stage 3C.4: Persistent Decode Stream Scheduler.
    Manages a persistent pool of CUDA streams, eliminating dynamic stream
    creation overhead and maximizing cross-request stream reuse.
    """
    def __init__(self, workspace_root: Path, num_streams: int = 4):
        self.workspace_root = Path(workspace_root)
        self.num_streams = num_streams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Warm pre-allocated CUDA stream pool
        self.stream_pool: List[torch.cuda.Stream] = []
        if self.device == "cuda":
            self.stream_pool = [torch.cuda.Stream() for _ in range(num_streams)]
            
        self.stream_assignments: Dict[str, torch.cuda.Stream] = {}
        self.last_used_time: Dict[int, float] = {i: time.perf_counter() for i in range(num_streams)}
        
        # Telemetry
        self.stream_continuity = 100.0  # continuous reuse efficiency percentage
        self.total_leases = 0
        self.total_hits = 0
        self.idle_gaps_ms = 0.0

    def lease_stream(self, session_id: str) -> Optional[torch.cuda.Stream]:
        """
        Leases a persistent CUDA stream from the pool for a given session.
        """
        self.total_leases += 1
        
        if self.device != "cuda":
            return None

        # Check if session already has a stream assigned (Hit)
        if session_id in self.stream_assignments:
            self.total_hits += 1
            self._update_continuity()
            return self.stream_assignments[session_id]

        # Find least-recently-used stream that is free
        stream_idx = 0
        min_time = float("inf")
        for idx in range(self.num_streams):
            # Check if this stream is assigned to any active session
            assigned = any(s == self.stream_pool[idx] for s in self.stream_assignments.values())
            if not assigned:
                if self.last_used_time[idx] < min_time:
                    min_time = self.last_used_time[idx]
                    stream_idx = idx
                    
        # Calculate idle gap duration before reuse
        gap = (time.perf_counter() - min_time) * 1000.0
        self.idle_gaps_ms = (self.idle_gaps_ms * 0.9) + (gap * 0.1)

        selected_stream = self.stream_pool[stream_idx]
        self.stream_assignments[session_id] = selected_stream
        self.last_used_time[stream_idx] = time.perf_counter()
        
        self._update_continuity()
        return selected_stream

    def release_stream(self, session_id: str):
        """
        Releases the stream back into the pool.
        """
        if session_id in self.stream_assignments:
            stream = self.stream_assignments.pop(session_id)
            # Find stream index to update its idle time anchor
            for idx, s in enumerate(self.stream_pool):
                if s == stream:
                    self.last_used_time[idx] = time.perf_counter()
                    break
        self._update_continuity()

    def _update_continuity(self):
        """
        Calculates stream continuity based on reuse hit ratio.
        """
        if self.total_leases > 0:
            self.stream_continuity = (self.total_hits / self.total_leases) * 100.0
        else:
            self.stream_continuity = 100.0

    def clear(self):
        """
        Resets stream scheduler states.
        """
        self.stream_assignments.clear()
        self.total_leases = 0
        self.total_hits = 0
        self.idle_gaps_ms = 0.0
        self.stream_continuity = 100.0
