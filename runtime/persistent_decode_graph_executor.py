import os
import torch
import time
from pathlib import Path
from typing import Callable, Any, Dict, Optional

class PersistentDecodeGraphExecutor:
    """
    CGO Phase 42.0 — Persistent Decode Graph Executor.
    Provides CUDA Graph capture and replay capabilities for static decode steps.
    Eliminates per-token launch fragmentation and reduces CPU wakeup overheads.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[str, Any] = {}
        self.static_outputs: Dict[str, Any] = {}
        self.captured = False

    def capture_graph(self, key: str, forward_fn: Callable[..., Any], static_input: Any):
        """
        Captures a PyTorch forward function inside a CUDA Graph.
        Ensures execution path is compiled for persistent re-play.
        """
        if not torch.cuda.is_available():
            return
            
        print(f"[CGO Graph Executor] Capturing CUDA graph for: {key}...")
        
        # Warmup execution as required by PyTorch CUDA graph guidelines
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = forward_fn(static_input)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        
        # Initialize graph and capture
        graph = torch.cuda.CUDAGraph()
        self.static_inputs[key] = static_input
        
        with torch.cuda.graph(graph):
            output = forward_fn(static_input)
            
        self.static_outputs[key] = output
        self.graphs[key] = graph
        self.captured = True
        print(f"[CGO Graph Executor] Graph {key} captured successfully.")

    def replay(self, key: str) -> Optional[Any]:
        """
        Replays the captured CUDA Graph for high-occupancy execution.
        """
        if not torch.cuda.is_available() or key not in self.graphs:
            return None
            
        # Replay graph in current stream
        self.graphs[key].replay()
        return self.static_outputs[key]
