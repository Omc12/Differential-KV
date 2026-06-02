import torch
import numpy as np
from native_core.mac_utils import mlx_svd_lowrank, mlx_available

print(f"MLX available: {mlx_available()}")

# Create a random matrix
torch.manual_seed(42)
x = torch.randn(64, 256)
rank = 8

# Compute SVD via PyTorch
U_py, S_py, Vh_py = torch.linalg.svd(x, full_matrices=False)
U_py_k = U_py[:, :rank] * S_py[:rank].unsqueeze(0)
Vh_py_k = Vh_py[:rank, :]
recon_py = U_py_k @ Vh_py_k

# Compute SVD via MLX SVD lowrank
mlx_res = mlx_svd_lowrank(x, rank)
if mlx_res is not None:
    U_mlx, S_mlx, Vh_mlx = mlx_res
    print(f"U_mlx shape: {U_mlx.shape}")
    print(f"S_mlx shape: {S_mlx.shape}")
    print(f"Vh_mlx shape: {Vh_mlx.shape}")
    
    U_mlx_k = U_mlx[:, :rank] * S_mlx[:rank].unsqueeze(0)
    Vh_mlx_k = Vh_mlx[:rank, :]
    recon_mlx = U_mlx_k @ Vh_mlx_k
    
    # Calculate reconstruction error
    diff_py = torch.abs(x - recon_py).mean().item()
    diff_mlx = torch.abs(x - recon_mlx).mean().item()
    diff_between = torch.abs(recon_py - recon_mlx).mean().item()
    
    print(f"PyTorch recon error: {diff_py:.4f}")
    print(f"MLX recon error: {diff_mlx:.4f}")
    print(f"Diff between PyTorch and MLX recon: {diff_between:.4f}")
else:
    print("MLX SVD returned None!")
