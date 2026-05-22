"""
runtime/metadata_pool.py

Phase 17 - Persistent Sparse Metadata Pools

Replaces dynamic `torch.stack()` and lists of block metadata with 
pre-allocated GPU-resident buffers. This eliminates CPU-GPU synchronization 
and python list comprehension overhead during the continuous batching decode step.
"""

import torch

class PersistentMetadataPool:
    def __init__(self, max_blocks: int, max_sessions: int, head_dim: int, rank: int, device: str = "cuda"):
        self.max_blocks = max_blocks
        self.max_sessions = max_sessions
        
        # Pre-allocated GPU buffers
        # [max_blocks, rank, head_dim] (example shape for U/V)
        self.U_pool = torch.zeros((max_blocks, 64, rank), dtype=torch.float16, device=device)
        self.V_pool = torch.zeros((max_blocks, rank, head_dim * 2), dtype=torch.float16, device=device)
        
        # Block mapping for each session
        # [max_sessions, max_blocks_per_session]
        self.session_block_indices = torch.full((max_sessions, max_blocks), -1, dtype=torch.int32, device=device)
        self.session_num_blocks = torch.zeros((max_sessions,), dtype=torch.int32, device=device)
        
        self.next_free_block = 0
        self.device = device
        
    def allocate_block(self) -> int:
        if self.next_free_block >= self.max_blocks:
            raise RuntimeError("Metadata pool exhausted")
        idx = self.next_free_block
        self.next_free_block += 1
        return idx
        
    def write_block_metadata(self, block_idx: int, U: torch.Tensor, V: torch.Tensor):
        """Directly writes U and V matrices into the persistent buffer."""
        # No new allocations, in-place copy
        self.U_pool[block_idx].copy_(U, non_blocking=True)
        self.V_pool[block_idx].copy_(V, non_blocking=True)
        
    def append_to_session(self, session_idx: int, block_idx: int):
        num = self.session_num_blocks[session_idx].item()
        self.session_block_indices[session_idx, num] = block_idx
        self.session_num_blocks[session_idx] += 1

    def get_session_indices(self, session_idx: int) -> torch.Tensor:
        """Returns the pre-allocated index tensor for Triton to read from."""
        num = self.session_num_blocks[session_idx]
        return self.session_block_indices[session_idx, :num]
