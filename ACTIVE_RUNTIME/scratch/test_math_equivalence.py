import os
import sys
import math
import torch

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from native_core.compression.lowrank import compress_lowrank
from runtime.native_block_pool import NativeBlockPool
from native_core.sparse_decode.triton_sparse_attn import _pytorch_vectorized_sparse_attn_decode

def main():
    print("=" * 60)
    print("  DiffKV Sparse Attention Math Equivalence Test")
    print("=" * 60)
    
    device = "cpu"
    dtype = torch.float32
    
    num_heads = 8
    num_kv_heads = 4
    num_key_value_groups = num_heads // num_kv_heads
    D = 64
    R = 8
    S_MAX = 32
    
    # 1. Generate keys, values, query
    torch.manual_seed(42)
    # Generate keys around a base anchor to make them highly low-rank
    anchor_k = torch.randn(1, num_kv_heads, 1, D, dtype=dtype) * 2.0
    anchor_v = torch.randn(1, num_kv_heads, 1, D, dtype=dtype) * 2.0
    
    # Generate delta tokens
    delta_k = torch.randn(1, num_kv_heads, S_MAX - 1, D, dtype=dtype) * 0.1
    delta_v = torch.randn(1, num_kv_heads, S_MAX - 1, D, dtype=dtype) * 0.1
    
    k = torch.cat([anchor_k, anchor_k + delta_k], dim=2) # [1, num_kv_heads, S_MAX, D]
    v = torch.cat([anchor_v, anchor_v + delta_v], dim=2) # [1, num_kv_heads, S_MAX, D]
    
    q = torch.randn(1, num_heads, 1, D, dtype=dtype) * 1.5
    
    # 2. Compute standard dense attention
    k_rep = k.repeat_interleave(num_key_value_groups, dim=1) # [1, num_heads, S_MAX, D]
    v_rep = v.repeat_interleave(num_key_value_groups, dim=1) # [1, num_heads, S_MAX, D]
    
    # SDPA
    dense_out = torch.nn.functional.scaled_dot_product_attention(
        q, k_rep, v_rep, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    
    # 3. Compress keys/values using our runtime code
    # We construct the flat delta matrix
    stacked = torch.stack([k[0].transpose(0, 1), v[0].transpose(0, 1)], dim=1)
    flat_tokens = stacked.reshape(S_MAX, 2 * num_kv_heads * D).float()
    anchor_flat = torch.stack([k[0, :, 0], v[0, :, 0]], dim=0).reshape(-1).float()
    
    deltas = flat_tokens - anchor_flat.unsqueeze(0)
    lr_delta = compress_lowrank(deltas, R)
    
    # 4. Write to NativeBlockPool
    pool = NativeBlockPool(
        max_blocks=16, num_kv_heads=num_kv_heads, head_dim=D,
        rank=R, max_seq_len=S_MAX,
        device=device, dtype=dtype, initial_blocks=4,
    )
    
    pool_idx = pool.allocate_block()
    pool.write_block(
        pool_idx=pool_idx,
        U=lr_delta.U,
        V=lr_delta.V,
        anchor_K=k[0, :, 0],
        anchor_V=v[0, :, 0],
        scale=lr_delta.scale,
        seq_len=S_MAX,
    )
    
    # 5. Run our sparse decode attention
    block_indices = torch.tensor([pool_idx], dtype=torch.int32)
    
    sparse_out = _pytorch_vectorized_sparse_attn_decode(
        q=q,
        block_indices=block_indices,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=num_key_value_groups,
        R=R,
        S_MAX=S_MAX,
    )
    
    # Compare outputs
    diff = (sparse_out - dense_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Check reconstruction of deltas
    recon_deltas = (lr_delta.U @ lr_delta.V) * lr_delta.scale
    recon_err = (recon_deltas - deltas).norm() / deltas.norm()
    print(f"SVD reconstruction error: {recon_err.item():.6f}")

if __name__ == "__main__":
    main()
