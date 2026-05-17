import torch
from pathlib import Path

class TensorCoreSparseAlignmentRuntime:
    """
    SGC Stage 3C.2: Tensor Core Sparse Alignment Runtime.
    Ensures sparse matrix shapes, alignments, and precisions are 
    100% compatible with Tensor Core GEMM operations (multiples of 8/16/32).
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.alignment_size = 16  # standard alignment for Tensor Cores

    def align_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forces a tensor's innermost dimensions to align with Tensor Core friendly layouts 
        (e.g., multiples of 8/16) and enforces contiguous half-precision storage (FP16).
        """
        # Ensure half precision (FP16 or BF16)
        if tensor.dtype not in [torch.float16, torch.bfloat16]:
            tensor = tensor.to(torch.float16)

        # Ensure layout contiguous in memory
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Audit and enforce Tensor Core alignment on the innermost dimension (head_dim or hidden)
        dim = tensor.shape[-1]
        if dim % self.alignment_size != 0:
            # Pad tensor along the last dimension to align with self.alignment_size
            pad_amount = self.alignment_size - (dim % self.alignment_size)
            padding = [0] * (2 * (tensor.ndim))
            padding[1] = pad_amount  # Pad end of the last dimension
            
            # Pad tensor with zeros
            tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0.0)
            
        return tensor.contiguous()

    def get_aligned_batch_indices(self, sparse_indices: torch.Tensor) -> torch.Tensor:
        """
        Converts irregular sparse block indexes into a contiguous, 
        Tensor-Core friendly grouped representation.
        """
        # Tensor Cores perform best when block allocations are aligned on 8-element boundaries
        B, S, N = sparse_indices.shape
        if N % 8 != 0:
            pad_amount = 8 - (N % 8)
            padding = [0, pad_amount, 0, 0, 0, 0]
            # Pad sparse block indexes with -1 (representing inactive blocks)
            sparse_indices = torch.nn.functional.pad(sparse_indices, padding, mode='constant', value=-1)
            
        return sparse_indices.contiguous()

    def get_tensor_core_utilization_estimate(self, Q: torch.Tensor, sparse_indices: torch.Tensor) -> float:
        """
        Calculates theoretical Tensor Core efficiency based on layout alignments.
        """
        innermost_dim = Q.shape[-1]
        num_blocks = sparse_indices.shape[-1]
        
        # Innermost dimension must be multiple of 16, block counts multiple of 8
        dim_score = 100.0 if innermost_dim % 16 == 0 else (100.0 * (innermost_dim // 16 * 16) / innermost_dim)
        block_score = 100.0 if num_blocks % 8 == 0 else (100.0 * (num_blocks // 8 * 8) / num_blocks)
        
        # Returns the layout alignment utilization score
        return float(min(dim_score, block_score))
