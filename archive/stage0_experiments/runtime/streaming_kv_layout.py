import torch

class StreamingKVLayout:
    """
    Optimizes KV memory layout for streaming access and cache locality.
    Prevents fragmentation during aggressive sparse pruning.
    """
    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def pack_sparse_kv(self, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        """
        Packs sparse KV tokens into contiguous memory blocks to improve 
        DRAM/VRAM burst read efficiency.
        """
        # mask: [batch, heads, seq_len]
        # k, v: [batch, heads, seq_len, head_dim]
        
        # In a real implementation, this would involve a custom CUDA kernel 
        # for zero-copy packing.
        
        packed_k = []
        packed_v = []
        
        # Mocking the packing process
        for b in range(k.shape[0]):
            for h in range(k.shape[1]):
                indices = torch.where(mask[b, h])[0]
                packed_k.append(k[b, h, indices])
                packed_v.append(v[b, h, indices])
                
        return packed_k, packed_v

    def get_block_aligned_indices(self, indices: torch.Tensor):
        """
        Ensures indices are aligned to memory blocks for efficient IO.
        """
        start = (indices // self.block_size) * self.block_size
        end = start + self.block_size
        return start, end
