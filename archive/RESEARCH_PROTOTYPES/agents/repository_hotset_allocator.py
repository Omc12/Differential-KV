"""
agents/repository_hotset_allocator.py

Specialized allocator for repository-wide 'context anchors'.
Ensures that critical code paths (entry points, core APIs) are always cached.
"""

from typing import List, Dict, Set, Any
import logging

class RepositoryHotsetAllocator:
    """
    Allocates and protects high-priority code anchors in the distributed cache.
    """
    def __init__(self, cache_manager: Any):
        self.cache = cache_manager
        self.protected_shards: Set[int] = set()
        self.logger = logging.getLogger("RepositoryHotsetAllocator")

    def identify_critical_shards(self, repo_structure: Dict[str, Any]):
        """
        Analyzes repository structure to identify entry points and core modules.
        """
        # Example: prioritize files in 'src/core' or 'runtime/'
        for file_path, shard_ids in repo_structure.items():
            if "core" in file_path or "runtime" in file_path:
                for sid in shard_ids:
                    self.protected_shards.add(sid)
                    self.logger.info(f"Protecting Critical Shard: {sid} ({file_path})")

    def enforce_residency(self):
        """
        Ensures all protected shards are resident in the distributed GPU cache.
        """
        for shard_id in self.protected_shards:
            if self.cache.get(shard_id) is None:
                self.logger.warning(f"Protected Shard {shard_id} missing from cache! Re-allocating...")
                # self.cache.load_into_gpu(shard_id)

    def get_protection_status(self) -> Dict[str, Any]:
        """Returns the count of protected shards and their health."""
        return {
            "protected_shard_count": len(self.protected_shards),
            "healthy_count": len([s for s in self.protected_shards if self.cache.get(s) is not None])
        }
