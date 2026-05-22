"""
runtime/async_tiered_ffn.py

Phase 17 - CUDA Stream Prefetching for Hierarchical FFN Residency

PCIe stalls during `seq=1` decoding destroy sparse compute gains. This module
introduces a dedicated CUDA stream for Host-to-Device weight transfers. 

Mechanism:
1. When routing predicts a required FFN block, we issue a non-blocking copy 
   on a background stream.
2. The main compute stream records an event and waits ONLY if the compute
   reaches the FFN layer before the transfer completes.
3. This completely overlaps the memory movement of layer L with the compute
   of layer L-1.
"""

import torch
import torch.nn.functional as F

class AsyncTieredFFN:
    def __init__(self, hidden_dim: int, d_ff: int, block_size: int, device: str = "cuda"):
        self.hidden_dim = hidden_dim
        self.d_ff = d_ff
        self.block_size = block_size
        self.device = device
        
        self.num_blocks = d_ff // block_size
        
        # Pinned CPU memory (fast PCIe transfer)
        self.cpu_gate = torch.zeros((d_ff, hidden_dim), dtype=torch.float16, pin_memory=True)
        self.cpu_up = torch.zeros((d_ff, hidden_dim), dtype=torch.float16, pin_memory=True)
        
        # GPU Cache Residency
        self.gpu_gate = torch.zeros((d_ff, hidden_dim), dtype=torch.float16, device=device)
        self.gpu_up = torch.zeros((d_ff, hidden_dim), dtype=torch.float16, device=device)
        
        self.block_resident = torch.zeros(self.num_blocks, dtype=torch.bool, device=device)
        
        # Async CUDA Stream Setup
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.transfer_events = {i: torch.cuda.Event() for i in range(self.num_blocks)}
        
        self.stats = {
            "sync_stalls": 0,
            "async_hits": 0,
            "prefetch_issued": 0
        }

    def issue_prefetch(self, block_idx: int):
        """Issues an async H2D transfer on the background stream."""
        if self.block_resident[block_idx]:
            return
            
        start_row = block_idx * self.block_size
        end_row = start_row + self.block_size
        
        with torch.cuda.stream(self.transfer_stream):
            # Non-blocking copy from pinned CPU RAM to VRAM
            self.gpu_gate[start_row:end_row].copy_(self.cpu_gate[start_row:end_row], non_blocking=True)
            self.gpu_up[start_row:end_row].copy_(self.cpu_up[start_row:end_row], non_blocking=True)
            
            # Record event on transfer stream
            self.transfer_events[block_idx].record(self.transfer_stream)
            
        self.stats["prefetch_issued"] += 1
        self.block_resident[block_idx] = True

    def forward_block(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Executes the FFN block, waiting on the event ONLY if necessary."""
        start_row = block_idx * self.block_size
        end_row = start_row + self.block_size
        
        # If we issued a prefetch, synchronize with the main stream
        if self.block_resident[block_idx]:
            # This will block the main stream ONLY if the transfer isn't finished yet
            if not self.transfer_events[block_idx].query():
                self.stats["sync_stalls"] += 1
            else:
                self.stats["async_hits"] += 1
                
            self.transfer_events[block_idx].wait()
        else:
            # Synchronous fallback (should never happen if routing predicted correctly)
            self.gpu_gate[start_row:end_row].copy_(self.cpu_gate[start_row:end_row])
            self.gpu_up[start_row:end_row].copy_(self.cpu_up[start_row:end_row])
            self.block_resident[block_idx] = True
            self.stats["sync_stalls"] += 1
            
        gate = F.linear(x, self.gpu_gate[start_row:end_row])
        up = F.linear(x, self.gpu_up[start_row:end_row])
        
        return F.silu(gate) * up
