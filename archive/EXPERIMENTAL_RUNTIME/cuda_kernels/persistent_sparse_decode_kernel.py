import time
from typing import Dict, Any
import logging

class PersistentSparseDecodeKernel:
    """
    Manages persistent GPU kernels for low-latency sparse decoding.
    Simulates threads that stay resident in GPU SMs.
    """
    def __init__(self):
        self.is_active = False
        self.uptime = 0.0
        self.logger = logging.getLogger("PersistentSparseDecode")

    def start_kernel(self):
        """Signals the GPU to start the persistent kernel."""
        self.is_active = True
        self.start_time = time.time()
        self.logger.info("Persistent decode kernel started on GPU.")

    def stop_kernel(self):
        self.is_active = False
        self.uptime = time.time() - self.start_time
        self.logger.info(f"Persistent decode kernel stopped. Uptime: {self.uptime:.2f}s")

    def dispatch_work(self, task_id: str):
        """Dispatches work to the already-running kernel via signaling."""
        if not self.is_active:
            raise RuntimeError("Persistent kernel not active.")
        self.logger.info(f"Dispatched task {task_id} to persistent GPU threads.")
        return True

    def get_persistent_metrics(self) -> Dict[str, Any]:
        return {
            "persistent_decode_active": self.is_active,
            "persistent_decode_uptime": self.uptime if not self.is_active else time.time() - self.start_time
        }
