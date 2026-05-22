import time
import torch

class PersistentCudaGraphReplayEngine:
    """
    NDX Phase 42.1.5 — Persistent CUDA Graph Replay Engine.
    Captures and replays PyTorch CUDA Graphs of the decode forward step
    to completely eliminate sequential host-device launch latency.
    """
    def __init__(self):
        self.captured = False
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        self.replay_count = 0
        self.warmup_runs = 3

    def capture_graph(self, model_fn, static_input_tensor: torch.Tensor):
        """
        Captures the decode step graph securely under standard CUDA stream synchronization.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("[NDX Cuda Graph Error] CUDA is unavailable. Cannot capture graph.")
            
        print("[Persistent Graph] Capturing persistent decode execution graph...")
        device = static_input_tensor.device
        self.static_input = static_input_tensor
        
        # Warmup phase to populate PyTorch's internal JIT/CUDA memory caches
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_runs):
                self.static_output = model_fn(self.static_input)
        torch.cuda.synchronize(device=device)
        
        # Capture phase
        self.cuda_graph = torch.cuda.CUDAGraph()
        
        # Avoid static memory pools allocation conflicts
        with torch.cuda.graph(self.cuda_graph, stream=stream):
            self.static_output = model_fn(self.static_input)
            
        torch.cuda.synchronize(device=device)
        self.captured = True
        self.replay_count = 0
        print("[Persistent Graph] Decode CUDAGraph captured successfully.")

    def replay(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Replays the captured graph natively.
        """
        if not self.captured or self.cuda_graph is None:
            raise RuntimeError("[NDX Violation] Attempted to replay uncaptured CUDA Graph!")
            
        # Copy input into static memory location
        self.static_input.copy_(input_tensor)
        
        # Replay graph completely natively
        self.cuda_graph.replay()
        self.replay_count += 1
        
        return self.static_output

    def get_replay_count(self) -> int:
        return self.replay_count
