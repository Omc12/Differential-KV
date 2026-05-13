"""
agents/repository_memory_index.py

Phase 12A: Repository Memory Index
Builds a persistent index of a code repository to provide long-lived
workspace memory for coding agents.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set
from anchor_logic.semantic_anchor_system import SemanticAnchor, SemanticAnchorMemory
import torch

class RepositoryMemoryIndex:
    """
    Indexes files within a repository and maps them to semantic anchors.
    This allows agents to 'remember' file contents across sessions.
    """
    def __init__(self, repo_root: str, memory: SemanticAnchorMemory):
        self.repo_root = Path(repo_root)
        self.memory = memory
        self.indexed_files: Dict[str, str] = {}  # rel_path -> hash
        self.file_to_anchors: Dict[str, List[int]] = {} # rel_path -> [positions]

    def index_repository(self, extensions: List[str] = [".py", ".md", ".txt"]):
        """Scans the repository and creates anchors for critical files."""
        print(f"[RepositoryMemoryIndex] Indexing repository at {self.repo_root}...")
        
        for file_path in self.repo_root.rglob("*"):
            if file_path.suffix in extensions and ".git" not in str(file_path):
                self._index_file(file_path)

    def _index_file(self, file_path: Path):
        rel_path = str(file_path.relative_to(self.repo_root))
        
        with open(file_path, "rb") as f:
            content = f.read()
            file_hash = hashlib.md5(content).hexdigest()

        if self.indexed_files.get(rel_path) == file_hash:
            return # No change

        self.indexed_files[rel_path] = file_hash
        
        # In a real system, we'd tokenize and pass through a model to get KV states.
        # Here we simulate by creating "structural anchors" for the file.
        # We use negative positions or special encoding to distinguish repo anchors.
        
        # Simulate creating an anchor for the file start
        anchor_pos = hash(rel_path) % 1000000 + 1000000 # Offset to avoid collision with session tokens
        
        anchor = SemanticAnchor(
            token_id=0, # Placeholder
            position=anchor_pos,
            kv_exact=torch.randn(2, 32, 128), # Mock KV
            reason="repo_structural",
            metadata={"rel_path": rel_path, "type": "file_boundary"}
        )
        
        self.memory.add_anchor(anchor)
        self.file_to_anchors.setdefault(rel_path, []).append(anchor_pos)

    def get_anchors_for_file(self, rel_path: str) -> List[int]:
        return self.file_to_anchors.get(rel_path, [])

    def sync_changes(self):
        """Re-indexes only changed files."""
        self.index_repository()
