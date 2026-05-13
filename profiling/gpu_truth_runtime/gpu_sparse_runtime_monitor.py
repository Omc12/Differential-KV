from .cuda_sparse_timer import CUDASparseTimer
from .triton_kernel_inspector import TritonKernelInspector
from .real_gpu_memory_map import RealGPUMemoryMap
from .kernel_bandwidth_analyzer import KernelBandwidthAnalyzer
from .sparse_execution_trace import SparseExecutionTrace

class GPUSparseRuntimeMonitor:
    """
    Main monitor for Phase 7.5B.
    Provides total hardware visibility for the sparse runtime.
    """
    def __init__(self):
        self.timer = CUDASparseTimer()
        self.inspector = TritonKernelInspector()
        self.memory_map = RealGPUMemoryMap()
        self.bandwidth = KernelBandwidthAnalyzer()
        self.trace = SparseExecutionTrace()

    def start_profile(self, name: str):
        self.trace.log_event(name, "kernel", "B")
        self.timer.start(name)

    def stop_profile(self, name: str, bytes_transferred: int = 0):
        self.timer.stop(name)
        self.trace.log_event(name, "kernel", "E")
        
        # Immediate stats if synced
        # (Usually we sync at the end of the batch)

    def get_full_telemetry(self) -> dict:
        """Collects all hardware truth metrics."""
        latencies = self.timer.sync_and_collect()
        vram = self.memory_map.get_detailed_map()
        
        return {
            "latencies_ms": latencies,
            "vram_mb": vram,
            "system_health": "STABLE" if max(latencies.values(), default=0) < 50 else "CONGESTED"
        }
