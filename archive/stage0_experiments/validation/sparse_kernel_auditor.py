import torch

class SparseKernelAuditor:
    """
    Audits sparse kernels for correctness and numerical stability.
    Ensures fused logic doesn't introduce silent errors or drift.
    """
    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def audit(self, sparse_output: torch.Tensor, dense_reference: torch.Tensor, mask: torch.Tensor):
        """
        Validates that sparse output matches dense reference where mask is active.
        """
        # In a sparse kernel, the output might slightly differ due to floating point 
        # accumulation order, but it should be within tolerance.
        
        diff = (sparse_output - dense_reference).abs()
        max_diff = diff.max().item()
        
        if max_diff > self.tolerance:
            return False, f"Numerical Drift Detected: {max_diff}"
            
        return True, "Audit Passed"
