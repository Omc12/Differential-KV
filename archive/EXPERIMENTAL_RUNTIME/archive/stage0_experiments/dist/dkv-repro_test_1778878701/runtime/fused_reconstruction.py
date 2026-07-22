"""
runtime/fused_reconstruction.py

Optimized, vectorized reconstruction kernels for Differential KV.
Eliminates token-by-token loops in Python.
"""

import torch
from typing import Optional, List, Tuple

def fused_lowrank_reconstruct(
    anchor_kv: torch.Tensor,    # [2, heads, head_dim] or [B, 2, heads, head_dim]
    U: torch.Tensor,            # [num_deltas, rank]
    V: torch.Tensor,            # [rank, feat_dim] (feat_dim = 2 * heads * head_dim)
    scale: float = 1.0,
    sparse_indices: Optional[torch.Tensor] = None, # [num_sparse]
    sparse_values: Optional[torch.Tensor] = None,  # [num_sparse, feat_dim]
    out_dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Fused low-rank reconstruction: ΔKV = U @ V.T + anchor_kv
    
    Args:
        anchor_kv: Base KV tensor to add deltas to.
        U: Low-rank U matrix.
        V: Low-rank V matrix.
        scale: Scaling factor for deltas.
        sparse_indices: (Optional) Indices within the delta block for sparse repair.
        sparse_values: (Optional) Sparse repair values.
        out_dtype: Output tensor dtype.
        
    Returns:
        reconstructed_deltas: [num_deltas, 2, heads, head_dim]
    """
    # 1. Low-rank reconstruction
    # [num_deltas, rank] @ [rank, feat_dim] -> [num_deltas, feat_dim]
    deltas = torch.matmul(U.float(), V.float())
    
    if scale != 1.0:
        deltas = deltas * scale
        
    # 2. Sparse repair (if provided)
    if sparse_indices is not None and sparse_values is not None:
        # deltas[sparse_indices] += sparse_values
        # Using index_add_ for in-place speed if possible, 
        # but deltas is newly created so we can just use it.
        deltas.index_add_(0, sparse_indices, sparse_values.float())
        
    # 3. Add anchor
    # anchor_kv shape: [2, heads, head_dim]
    # feat_dim = 2 * heads * head_dim
    num_deltas = U.shape[0]
    feat_dim = V.shape[1]
    
    # Reshape deltas to [num_deltas, 2, heads, head_dim]
    heads = anchor_kv.shape[-2]
    head_dim = anchor_kv.shape[-1]
    deltas = deltas.view(num_deltas, 2, heads, head_dim)
    
    # Vectorized addition: [num_deltas, 2, heads, head_dim] + [1, 2, heads, head_dim]
    reconstructed = deltas + anchor_kv.float().unsqueeze(0)
    
    return reconstructed.to(out_dtype)

def batched_reconstruct_periodic(
    anchors: torch.Tensor,      # [num_anchors, 2, heads, head_dim]
    U_list: List[torch.Tensor], # List of [block_len, rank]
    V_list: List[torch.Tensor], # List of [rank, feat_dim]
    scales: List[float],
    block_assignments: torch.Tensor, # [seq_len] -> index into anchors/U_list/V_list
    out_dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Reconstruct an entire sequence where multiple blocks exist.
    This is still a loop over blocks, but each block is internally vectorized.
    """
    # In a real runtime, we'd probably have these pre-concatenated or 
    # use a specialized kernel.
    pass

class FusedKVReconstructor:
    """
    Optimized reconstructor that avoids token-by-token processing.
    """
    @staticmethod
    def reconstruct_block(
        anchor_kv: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        scale: float,
        sparse_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        indices, values = sparse_data if sparse_data else (None, None)
        return fused_lowrank_reconstruct(anchor_kv, U, V, scale, indices, values)
