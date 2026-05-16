import logging
import os
from typing import Optional
from containers.persistent_cognition_container import PersistentCognitionContainer
from containers.session_snapshot_engine import SessionSnapshotEngine

logger = logging.getLogger(__name__)

class RuntimeRestoreManager:
    """
    Orchestrates the recovery of a cognitive session from a snapshot.
    Handles container recreation, state injection, and runtime resumption.
    """
    def __init__(self, snapshot_engine: SessionSnapshotEngine):
        self.snapshot_engine = snapshot_engine

    def restore_from_snapshot(self, snapshot_path: str) -> Optional[PersistentCognitionContainer]:
        """Restores a full runtime container from a disk snapshot."""
        if not os.path.exists(snapshot_path):
            logger.error(f"Snapshot file not found: {snapshot_path}")
            return None

        logger.info(f"Initiating restore from {snapshot_path}...")
        
        # 1. Load metadata
        metadata = self.snapshot_engine.load_snapshot_metadata(snapshot_path)
        if not metadata:
            return None
            
        # 2. Recreate container
        container = PersistentCognitionContainer(container_id=metadata.get("container_id"))
        container.mount_runtime(metadata.get("config", {}))
        
        # 3. Inject saved manifolds
        for m_id in metadata.get("manifolds", []):
            container.inject_manifold({"id": m_id, "restored": True})
            
        logger.info(f"Successfully restored container {container.container_id}.")
        return container
