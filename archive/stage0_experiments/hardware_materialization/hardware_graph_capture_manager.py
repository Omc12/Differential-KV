"""
hardware_materialization/hardware_graph_capture_manager.py

Manages real CUDA graph capture and replay for Differential KV.
"""

import torch
import logging
from typing import Dict, Any, Callable, Tuple

logger = logging.getLogger("GraphCaptureManager")

class HardwareGraphCaptureManager:
    """
    Orchestrates CUDA graph lifecycle: capture, replay, and invalidation.
    """
    def __init__(self):
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[str, Tuple[torch.Tensor, ...]] = {}
        self.static_outputs: Dict[str, Any] = {}
        self.enabled = torch.cuda.is_available()

    def capture_graph(self, key: str, func: Callable, inputs: Tuple[torch.Tensor, ...]) -> bool:
        """
        Captures a function into a CUDA graph.
        Inputs must be static (allocated on GPU).
        """
        if not self.enabled:
            return False

        try:
            # 1. Warmup
            # Multiple iterations to ensure all kernels are JITed and memory is stabilized
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = func(*inputs)
            torch.cuda.current_stream().wait_stream(s)

            # 2. Capture
            g = torch.cuda.CUDAGraph()
            # We keep references to inputs to ensure they remain static
            self.static_inputs[key] = inputs
            
            with torch.cuda.graph(g):
                self.static_outputs[key] = func(*inputs)
            
            self.graphs[key] = g
            logger.info(f"Successfully captured CUDA graph for '{key}'.")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to capture CUDA graph for '{key}': {e}")
            return False

    def replay_graph(self, key: str) -> Any:
        """
        Replays a previously captured CUDA graph.
        """
        if key not in self.graphs:
            logger.warning(f"Graph '{key}' not found for replay.")
            return None
            
        self.graphs[key].replay()
        return self.static_outputs[key]

    def invalidate_graph(self, key: str):
        """Removes a captured graph."""
        if key in self.graphs:
            del self.graphs[key]
            del self.static_inputs[key]
            del self.static_outputs[key]
            logger.info(f"Invalidated CUDA graph '{key}'.")

    def is_captured(self, key: str) -> bool:
        return key in self.graphs
