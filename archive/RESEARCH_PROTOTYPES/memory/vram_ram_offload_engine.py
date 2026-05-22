import torch

class VRAMRAMOffloadEngine:
    """
    Handles asynchronous KV offloading and onloading between VRAM and RAM.
    Uses CUDA streams to hide transfer latency.
    """
    def __init__(self):
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def offload(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        """
        Moves KV tensor to CPU RAM asynchronously.
        """
        if self.stream:
            with torch.cuda.stream(self.stream):
                cpu_kv = kv_tensor.to("cpu", non_blocking=True)
                return cpu_kv
        return kv_tensor.to("cpu")

    def onload(self, cpu_kv: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Moves KV tensor back to GPU VRAM asynchronously.
        """
        if self.stream:
            with torch.cuda.stream(self.stream):
                gpu_kv = cpu_kv.to(device, non_blocking=True)
                return gpu_kv
        return cpu_kv.to(device)

    def synchronize(self):
        if self.stream:
            self.stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
