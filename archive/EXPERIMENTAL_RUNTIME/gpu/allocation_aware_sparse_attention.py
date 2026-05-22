import torch
import torch.nn as nn
import math

class AllocationAwareSparseAttention(nn.Module):
    """
    PHASE 18.3A: Custom attention path that avoids dense O(n^2) allocation.
    Intercepts the prefill phase to process attention in bounded blocks.
    """
    def __init__(self, config, block_size=1024):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(self, query, key, value, attention_mask=None):
        """
        Input: 
            query: [batch, heads, q_len, head_dim]
            key, value: [batch, heads, kv_len, head_dim]
        """
        batch_size, num_heads, q_len, head_dim = query.shape
        kv_len = key.shape[2]
        
        # MANDATORY: Avoid full dense allocation
        # If q_len > block_size, we enter allocation-aware mode
        if q_len > self.block_size:
            return self.block_sparse_forward(query, key, value, attention_mask)
        
        # Dense fallback for small chunks (only if strictly necessary)
        attn = (query @ key.transpose(-1, -2)) / math.sqrt(head_dim)
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = torch.softmax(attn, dim=-1)
        return attn @ value

    def block_sparse_forward(self, query, key, value, attention_mask=None):
        """
        Processes attention in tiles to keep VRAM allocation O(block_size * kv_len).
        """
        batch_size, num_heads, q_len, head_dim = query.shape
        out = torch.zeros_like(query)
        
        for i in range(0, q_len, self.block_size):
            q_block = query[:, :, i:i+self.block_size, :]
            
            # Compute attention for this block only
            # Resulting matrix is [batch, heads, block_size, kv_len]
            # This is significantly smaller than [batch, heads, q_len, kv_len]
            attn_block = (q_block @ key.transpose(-1, -2)) / math.sqrt(head_dim)
            
            if attention_mask is not None:
                attn_block = attn_block + attention_mask[:, :, i:i+self.block_size, :]
            
            attn_block = torch.softmax(attn_block, dim=-1)
            out[:, :, i:i+self.block_size, :] = attn_block @ value
            
            # Record telemetry point
            # print(f"[MEASURED] Block {i//self.block_size} allocation: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            
        return out
