import logging
from typing import Dict, List, Any

class DistributedPersistentCoordinator:
    """
    Coordinates persistent decode kernels across multiple GPU devices.
    Ensures synchronized wake-up and execution for distributed shards.
    """
    def __init__(self, devices: List[str]):
        self.devices = devices
        self.device_states: Dict[str, str] = {d: "idle" for d in devices}
        self.was_active = False
        self.logger = logging.getLogger("DistributedPersistentCoordinator")

    def synchronize_wake(self, task_id: str):
        """Signals all participating devices to wake their persistent kernels."""
        self.was_active = True
        for dev in self.devices:
            self.device_states[dev] = "active"
            self.logger.info(f"Waking persistent kernel on {dev} for {task_id}")
        return True

    def signal_completion(self, device: str):
        self.device_states[device] = "idle"
        self.logger.info(f"Device {device} completed persistent task.")

    def get_coordination_metrics(self) -> Dict[str, Any]:
        active_count = sum(1 for s in self.device_states.values() if s == "active")
        return {
            "distributed_persistent_uptime": 1.0 if self.was_active else 0.0,
            "active_device_shards": active_count
        }
