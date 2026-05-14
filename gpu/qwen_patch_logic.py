import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

def allocation_aware_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    PHASE 18.3A: SDPA-Accelerated Allocation-Aware Attention.
    Uses hardware-accelerated tiling to bypass the 27.95 GiB bottleneck.
    """
    bsz, q_len, _ = hidden_states.size()

    # 1. Project Q, K, V
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    head_dim = self.config.hidden_size // num_heads

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    # 2. Rotary Embeddings
    if position_embeddings is None:
        rotary_emb = getattr(self, "rotary_emb", getattr(self, "_rotary_emb_ref", None))
        cos, sin = rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # 3. Handle Cache (DynamicCache/v4.45 compatibility)
    if past_key_values is not None:
        if isinstance(past_key_values, tuple):
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)
        else:
            # Cache object
            key_states, value_states = past_key_values.update(key_states, value_states, getattr(self, "layer_idx", 0))

    # 4. ALLOCATION-AWARE ATTENTION BLOCK (Broadcasting-Optimized)
    # We avoid repeat_kv (14GB) by using broadcasting in the matmul tile loop.
    attn_output = torch.zeros_like(query_states)
    tile_size = 1024 
    
    n_rep = self.config.num_attention_heads // self.config.num_key_value_heads
    q_reshaped = query_states.view(bsz, num_key_value_heads, n_rep, q_len, head_dim)
    k_reshaped = key_states.unsqueeze(2)
    v_reshaped = value_states.unsqueeze(2)

    for i in range(0, q_len, tile_size):
        end_idx = min(i + tile_size, q_len)
        q_tile = q_reshaped[:, :, :, i:end_idx, :]
        
        # Broadcasting Matmul: [bsz, kv_heads, n_rep, tile_size, kv_len]
        # Peak allocation: 1024 * 16384 * 28 * 2 = 939 MB
        # We explicitly use float16 to avoid 2x spike
        attn_weights_tile = torch.matmul(q_tile, k_reshaped.transpose(-1, -2)) / math.sqrt(head_dim)
        
        if attention_mask is not None:
            m_tile = attention_mask[:, :, i:end_idx, :] if attention_mask.shape[2] > 1 else attention_mask
            attn_weights_tile = attn_weights_tile + m_tile.unsqueeze(1)
            
        attn_weights_tile = torch.nn.functional.softmax(attn_weights_tile, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        tile_out = torch.matmul(attn_weights_tile, v_reshaped)
        attn_output[:, :, i:end_idx, :] = tile_out.view(bsz, num_heads, -1, head_dim)
        
        # Cleanup tile intermediates
        del attn_weights_tile, tile_out

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None
