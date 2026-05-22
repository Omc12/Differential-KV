"""
runtime/static_decode_graph.py

Phase 17 - Static Sparse Execution Graphs

Wraps the sparse decode sequence in a replayable CUDA graph. This eliminates
Python dispatch overhead, dynamic graph rebuilds, and repeated launch scheduling
for the critical seq=1 continuous batching hot path.

Because CUDA graphs require static memory addresses, we use the `metadata_pool.py`
to ensure the block metadata (U, V, indices) remain at stable pointers.
"""

import torch

class StaticSparseDecodeGraph:
    def __init__(self, decode_fn, max_batch_size: int, head_dim: int, device: str = "cuda"):
        self.decode_fn = decode_fn
        self.device = device
        
        # Static input buffers for graph replay
        self.static_q = torch.zeros((max_batch_size, 32, head_dim), dtype=torch.float16, device=device)
        self.static_session_ids = torch.zeros((max_batch_size,), dtype=torch.int32, device=device)
        self.static_out = torch.zeros((max_batch_size, 32, head_dim), dtype=torch.float16, device=device)
        
        self.graph = None
        self.is_captured = False
        
    def capture(self, q: torch.Tensor, session_ids: torch.Tensor):
        """Captures the Triton Sparse Decode kernel execution."""
        # Warmup
        self.static_q[:q.size(0)].copy_(q)
        self.static_session_ids[:session_ids.size(0)].copy_(session_ids)
        
        torch.cuda.synchronize()
        
        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            # We call the external decode_fn which uses the static metadata pools internally
            out = self.decode_fn(self.static_q, self.static_session_ids)
            self.static_out.copy_(out)
            
        self.is_captured = True
        
    def replay(self, q: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
        """Replays the static graph with new queries."""
        if not self.is_captured:
            self.capture(q, session_ids)
            
        bsz = q.size(0)
        self.static_q[:bsz].copy_(q)
        self.static_session_ids[:bsz].copy_(session_ids)
        
        self.graph.replay()
        
        return self.static_out[:bsz].clone()
