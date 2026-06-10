import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from runtime.native_block_pool import NativeBlockPool

def test_pool():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    num_layers = 1
    num_kv_heads = 2
    head_dim = 64
    max_seq_len = 256
    pool_rank = 32
    
    pool = NativeBlockPool(
        max_blocks=100,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rank=pool_rank,
        max_seq_len=max_seq_len,
        device=device,
        dtype=torch.float16,
    )
    
    # Create mock inputs
    seq_len = 48
    U = torch.randn(seq_len, pool_rank, device=device) * 2.0
    V = torch.randn(pool_rank, 2 * num_kv_heads * head_dim, device=device)
    anchor_K = torch.randn(num_kv_heads, head_dim, device=device)
    anchor_V = torch.randn(num_kv_heads, head_dim, device=device)
    scale = 0.5
    
    # Write to pool
    pool_idx = pool.allocate_block()
    pool.write_block(
        pool_idx=pool_idx,
        U=U,
        V=V,
        anchor_K=anchor_K,
        anchor_V=anchor_V,
        scale=scale,
        seq_len=seq_len,
    )
    
    # Read back
    indices = torch.tensor([pool_idx], device=device)
    
    # Reconstruct U
    U_recon = pool.U[indices].to(torch.float32) * pool.U_scale[indices].view(-1, 1, 1).to(torch.float32)
    # Reconstruct V_K
    V_K_recon = pool.V_K[indices].to(torch.float32)
    # Reconstruct Anchor K
    anchor_K_recon = pool.anchors_K[indices].to(torch.float32)
    
    # Check shape & values
    print("Shapes:")
    print(f"  U_recon shape      : {U_recon.shape}")
    print(f"  V_K_recon shape    : {V_K_recon.shape}")
    print(f"  anchor_K_recon shape: {anchor_K_recon.shape}")
    
    # Quantization error on U
    err_u = (U_recon[0, :seq_len] - U.float()).abs().max().item()
    print(f"U quantization max abs error: {err_u:.6f}")
    
    # Check V_K values
    vk_orig = V[:, :num_kv_heads * head_dim].view(pool_rank, num_kv_heads, head_dim)
    err_v = (V_K_recon[0] - vk_orig.float()).abs().max().item()
    print(f"V_K max abs error: {err_v:.6f}")
    
    # Check anchor K values
    err_anc = (anchor_K_recon[0] - anchor_K.float()).abs().max().item()
    print(f"Anchor K max abs error: {err_anc:.6f}")

if __name__ == "__main__":
    test_pool()
