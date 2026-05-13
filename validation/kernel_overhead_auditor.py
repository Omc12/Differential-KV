import torch

class KernelOverheadAuditor:
    """
    PHASE 6H: Kernel Overhead Auditor
    Measures the ratio of 'Launch Overhead' vs 'Compute Time'.
    Rejects optimizations that significantly increase kernel fragmentation.
    """
    def __init__(self):
        pass

    def audit_kernel(self, kernel_name: str, wall_time: float, gpu_time: float):
        """
        Calculates overhead ratio.
        """
        overhead = wall_time - gpu_time
        ratio = overhead / wall_time
        if ratio > 0.3:
            return False, f"Kernel {kernel_name} has excessive launch overhead ({ratio:.2%})."
        return True, "Overhead is acceptable."
