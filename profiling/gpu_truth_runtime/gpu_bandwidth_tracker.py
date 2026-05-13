import torch
import time

class GPUBandwidthTracker:
    """
    Measures real GPU bandwidth consumption during sparse retrieval.
    Verifies if Differential KV is truly reducing memory traffic.
    """
    def __init__(self):
        self.start_time = 0
        self.total_bytes = 0

    def start_transaction(self):
        self.start_time = time.perf_counter()
        self.total_bytes = 0

    def record_transfer(self, num_elements: int, element_size: int):
        """Records a memory transfer of N elements."""
        self.total_bytes += num_elements * element_size

    def end_transaction(self) -> float:
        """Returns the bandwidth in GB/s."""
        duration = time.perf_counter() - self.start_time
        if duration == 0:
            return 0.0
        gbps = (self.total_bytes / (1024**3)) / duration
        return gbps

    def calculate_sparse_reduction(self, dense_size: int, sparse_size: int) -> float:
        """Calculates the bandwidth reduction ratio."""
        if dense_size == 0:
            return 1.0
        return 1.0 - (sparse_size / dense_size)
