import torch
import triton
import triton.language as tl
from profiling.gpu_hardware_truth import CUDAEventPipeline

class TritonKernelTruth:
    """
    Profiles Triton sparse kernels with hardware-level timing.
    """
    def __init__(self, logger_name: str = "triton_truth"):
        from empirical.runtime_truth_logger import RuntimeTruthLogger
        self.logger = RuntimeTruthLogger(logger_name)
        self.pipeline = CUDAEventPipeline()

    def profile_kernel(self, kernel_fn, *args, **kwargs):
        """Runs a kernel with precise timing."""
        self.pipeline.start_event("kernel_execution")
        kernel_fn(*args, **kwargs)
        self.pipeline.end_event("kernel_execution")
        
        results = self.pipeline.collect_results()
        self.logger.log("kernel_truth", {
            "kernel_name": kernel_fn.__name__,
            "duration_ms": results["kernel_execution"],
            "config": {k: str(v) for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
        })
        return results["kernel_execution"]

if __name__ == "__main__":
    # Simple test if triton is available
    print("Triton Truth Profiler Ready.")
