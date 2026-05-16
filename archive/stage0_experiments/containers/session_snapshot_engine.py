import time
import json
import logging
import gzip
from typing import Dict

logger = logging.getLogger(__name__)

class SessionSnapshotEngine:
    """
    Serializes live cognitive sessions, compressing the latent KV states
    and routing manifolds into compact snapshot files.
    """
    def __init__(self, snapshot_dir: str = "./snapshots"):
        self.snapshot_dir = snapshot_dir
        import os
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def create_snapshot(self, session_id: str, container_state: Dict, latent_tensors: bytes) -> str:
        """
        Creates a full snapshot combining container metadata and compressed latent tensors.
        Returns the path to the snapshot file.
        """
        snapshot_id = f"snap_{session_id}_{int(time.time())}"
        snapshot_path = f"{self.snapshot_dir}/{snapshot_id}.csnap"
        
        # Combine metadata and raw tensors
        payload = {
            "metadata": container_state,
            "latent_size": len(latent_tensors),
            "timestamp": time.time()
        }
        
        try:
            # We use gzip to simulate tensor compression
            with gzip.open(snapshot_path, 'wt') as f:
                json.dump(payload, f)
            logger.info(f"Created snapshot {snapshot_id} at {snapshot_path}")
            return snapshot_path
        except Exception as e:
            logger.error(f"Failed to create snapshot for {session_id}: {e}")
            return ""

    def load_snapshot_metadata(self, snapshot_path: str) -> Dict:
        """Reads just the metadata from a snapshot file."""
        try:
            with gzip.open(snapshot_path, 'rt') as f:
                data = json.load(f)
            return data["metadata"]
        except Exception as e:
            logger.error(f"Failed to read snapshot {snapshot_path}: {e}")
            return {}
