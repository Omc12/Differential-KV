"""
agents/repository_drift_tracker.py

Tracks changes in the repository to invalidate or update stale sparse anchors.
Ensures agent memory reflects the current state of the code.
"""

import os
import hashlib
from typing import Dict, Set, List
import logging

class RepositoryDriftTracker:
    """
    Monitor for repository mutations.
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.file_hashes: Dict[str, str] = {}
        self.logger = logging.getLogger("RepositoryDriftTracker")

    def scan_for_drift(self) -> Set[str]:
        """
        Scans the repository and returns a set of changed files.
        """
        changed = set()
        for root, _, files in os.walk(self.repo_path):
            for f in files:
                if f.endswith('.py'):
                    path = os.path.join(root, f)
                    h = self._get_file_hash(path)
                    if path in self.file_hashes and self.file_hashes[path] != h:
                        changed.add(path)
                    self.file_hashes[path] = h
        
        if changed:
            self.logger.info(f"Drift detected in {len(changed)} files.")
        return changed

    def _get_file_hash(self, path: str) -> str:
        """Calculates MD5 hash of a file."""
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def get_stale_shards(self, file_to_shards: Dict[str, List[int]]) -> List[int]:
        """
        Returns shard IDs that are associated with changed files.
        """
        changed_files = self.scan_for_drift()
        stale_shards = []
        for f in changed_files:
            if f in file_to_shards:
                stale_shards.extend(file_to_shards[f])
        return stale_shards
