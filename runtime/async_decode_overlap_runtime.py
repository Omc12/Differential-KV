import os
import time
import torch
from pathlib import Path

class AsyncDecodeOverlapRuntime:
    """
    DPC Phase 42.1 — Async Decode Overlap Runtime.
    Coordinates parallel CUDA streams to execute sparse attention and copy metadata/KV cache 
    in parallel execution windows, avoiding serialized bottlenecks.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.streams = [torch.cuda.Stream() for _ in range(2)] if torch.cuda.is_available() else []

    def execute_overlapped(self, compute_fn, transfer_fn):
        """
        Executes compute and copy transfers concurrently using background streams.
        """
        if not torch.cuda.is_available() or not self.streams:
            compute_fn()
            transfer_fn()
            return
            
        with torch.cuda.stream(self.streams[0]):
            compute_fn()
            
        with torch.cuda.stream(self.streams[1]):
            transfer_fn()
            
        # Parallel streams synchronization
        self.streams[0].synchronize()
        self.streams[1].synchronize()
