"""
runtime/native_block_pool.py

Phase 10: Native Block Pool (vLLM style block tables)

Pre-allocates large contiguous GPU memory pools for all sparse block components
(U, V_K, V_V, anchors, scales, seq_lens). When blocks are compressed, their data
is copied into an assigned slot in this pool.

During inference, we completely bypass `torch.stack`. We simply pass a 1D tensor
of `block_indices` to the Triton kernel, which does the gather natively in SRAM.
"""

import torch

class NativeBlockPool:
    def __init__(
        self,
        max_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        rank: int,
        max_seq_len: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.max_blocks = max_blocks
        self.device = device
        self.dtype = dtype

        # Allocate pools
        self.U          = torch.zeros((max_blocks, max_seq_len, rank), device=device, dtype=self.dtype)
        self.V_K        = torch.zeros((max_blocks, rank, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.V_V        = torch.zeros((max_blocks, rank, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.anchors_K  = torch.zeros((max_blocks, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.anchors_V  = torch.zeros((max_blocks, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.scales     = torch.zeros((max_blocks,), device=device, dtype=self.dtype)
        self.seq_lens   = torch.zeros((max_blocks,), device=device, dtype=torch.int32)
        
        # Block allocator state
        self._free_indices = list(range(max_blocks - 1, -1, -1))
        self._ref_counts = [0] * max_blocks
        
    def allocate_block(self) -> int:
        if not self._free_indices:
            raise RuntimeError("NativeBlockPool is out of memory! Increase max_blocks.")
        idx = self._free_indices.pop()
        self._ref_counts[idx] = 1
        return idx

    def allocate_blocks(self, count: int) -> list:
        if len(self._free_indices) < count:
            raise RuntimeError(f"NativeBlockPool is out of memory! Requested {count}, available {len(self._free_indices)}")
        allocated = self._free_indices[-count:]
        del self._free_indices[-count:]
        for idx in allocated:
            self._ref_counts[idx] = 1
        return allocated
        
    def increment_ref(self, pool_idx: int):
        if pool_idx is not None and 0 <= pool_idx < self.max_blocks:
            self._ref_counts[pool_idx] += 1

    def free_block(self, pool_idx: int):
        if pool_idx is not None and 0 <= pool_idx < self.max_blocks:
            self._ref_counts[pool_idx] -= 1
            if self._ref_counts[pool_idx] <= 0:
                self._ref_counts[pool_idx] = 0
                self._free_indices.append(pool_idx)
        
    def write_block(
        self, 
        pool_idx: int, 
        U: torch.Tensor, 
        V: torch.Tensor, 
        anchor_K: torch.Tensor, 
        anchor_V: torch.Tensor, 
        scale: float, 
        seq_len: int
    ):
        """
        Copies compressed data directly into the contiguous pool.
        This happens in the background (AsyncCompressor) or once per block,
        NEVER during the decode hot-path.
        """
        self.U[pool_idx, :seq_len, :U.shape[1]] = U.to(self.dtype)
        
        rank = V.shape[0]
        num_kv = self.V_K.shape[2]
        h_dim = self.V_K.shape[3]
        
        vk = V[:, :num_kv * h_dim].view(rank, num_kv, h_dim)
        vv = V[:, num_kv * h_dim:].view(rank, num_kv, h_dim)
        
        self.V_K[pool_idx, :rank] = vk.to(self.dtype)
        self.V_V[pool_idx, :rank] = vv.to(self.dtype)
        self.anchors_K[pool_idx] = anchor_K.to(self.dtype)
        self.anchors_V[pool_idx] = anchor_V.to(self.dtype)
        self.scales[pool_idx] = scale
        self.seq_lens[pool_idx] = seq_len

    def reset(self):
        """Completely reset the pool to its initial state, zeroing out all tensors."""
        self._free_indices = list(range(self.max_blocks - 1, -1, -1))
        self._ref_counts = [0] * self.max_blocks
        self.U.zero_()
        self.V_K.zero_()
        self.V_V.zero_()
        self.anchors_K.zero_()
        self.anchors_V.zero_()
        self.scales.zero_()
        self.seq_lens.zero_()
