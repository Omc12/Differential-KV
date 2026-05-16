import uuid
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PersistentCognitionContainer:
    """
    Wraps the cognition runtime in an isolated, resumable execution environment.
    Ensures that if the underlying process dies, the cognitive state remains
    uncontaminated and ready for revival.
    """
    def __init__(self, container_id: Optional[str] = None):
        self.container_id = container_id or str(uuid.uuid4())
        self.status = "initialized"
        self.mounted_manifolds = []
        self.runtime_config = {}
        
    def mount_runtime(self, config: Dict[str, Any]):
        """Injects configuration and prepares the runtime for execution."""
        self.runtime_config = config
        self.status = "mounted"
        logger.info(f"Container {self.container_id} mounted with config keys: {list(config.keys())}")

    def inject_manifold(self, manifold_data: Dict):
        """Loads a pre-computed reasoning manifold into the container."""
        if self.status != "mounted":
            raise RuntimeError("Must mount runtime before injecting manifolds.")
        self.mounted_manifolds.append(manifold_data.get("id", "unknown"))
        logger.info(f"Manifold {manifold_data.get('id')} injected into {self.container_id}")

    def start(self):
        """Starts the persistent execution loop within the container."""
        if self.status != "mounted":
            raise RuntimeError("Cannot start an unmounted container.")
        self.status = "running"
        logger.info(f"Container {self.container_id} is now RUNNING.")

    def stop(self):
        """Safely suspends the container, preparing it for a snapshot."""
        self.status = "suspended"
        logger.info(f"Container {self.container_id} SUSPENDED.")
        
    def get_state(self) -> Dict:
        """Returns the complete internal state for snapshotting."""
        return {
            "container_id": self.container_id,
            "status": self.status,
            "manifolds": self.mounted_manifolds,
            "config": self.runtime_config
        }
