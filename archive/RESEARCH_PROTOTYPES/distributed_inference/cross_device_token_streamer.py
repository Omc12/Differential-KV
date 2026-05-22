import collections
from typing import Dict, List, Any
import logging

class CrossDeviceTokenStreamer:
    """
    Manages high-throughput sparse token streaming across GPU devices.
    """
    def __init__(self, buffer_size: int = 100):
        self.device_buffers: Dict[str, collections.deque] = {}
        self.buffer_size = buffer_size
        self.stream_log: List[Dict] = []
        self.logger = logging.getLogger("CrossDeviceTokenStreamer")

    def stream_token(self, token_id: str, source_device: str, target_device: str):
        """Streams a token from one device to another."""
        if target_device not in self.device_buffers:
            self.device_buffers[target_device] = collections.deque(maxlen=self.buffer_size)
        
        self.device_buffers[target_device].append(token_id)
        
        self.stream_log.append({
            "token_id": token_id,
            "from": source_device,
            "to": target_device
        })
        self.logger.info(f"Streamed token {token_id} from {source_device} to {target_device}")

    def get_stream_stability(self) -> float:
        """Returns the stability of the token streams."""
        # Simple check: no buffer overflows
        return 1.0 if all(len(b) < self.buffer_size for b in self.device_buffers.values()) else 0.8
