"""
repositories/hierarchical_repo_index.py

Phase 12B: Hierarchical Repository Index
Scales sparse retrieval across very large repositories by organizing 
anchors into a hierarchical structure (Repo -> Module -> File).
"""

from typing import Dict, List, Optional
from pathlib import Path

class HierarchicalRepoIndex:
    """
    Manages a multi-level index of a codebase.
    Allows for coarse-to-fine retrieval: 
    1. Search modules
    2. Search files within modules
    3. Retrieve specific semantic anchors within files
    """
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.index = {} # {module_path: {file_path: [anchor_ids]}}

    def build_index(self):
        """Walks the repository and builds the hierarchy."""
        print(f"[HierarchicalRepoIndex] Building index for {self.repo_root}...")
        for path in self.repo_root.rglob("*.py"):
            if ".git" in str(path) or "__pycache__" in str(path):
                continue
            
            rel_path = path.relative_to(self.repo_root)
            module = str(rel_path.parent)
            filename = str(rel_path.name)
            
            if module not in self.index:
                self.index[module] = {}
            
            self.index[module][filename] = [] # To be filled with anchor IDs

    def register_anchor(self, file_rel_path: str, anchor_id: int):
        """Associates an anchor with a specific file in the hierarchy."""
        path = Path(file_rel_path)
        module = str(path.parent)
        filename = str(path.name)
        
        if module in self.index and filename in self.index[module]:
            self.index[module][filename].append(anchor_id)

    def search(self, query: str) -> List[str]:
        """Simple keyword search across the index (simulated)."""
        results = []
        for module, files in self.index.items():
            if query.lower() in module.lower():
                results.append(module)
            for filename in files:
                if query.lower() in filename.lower():
                    results.append(f"{module}/{filename}")
        return results

    def get_anchors_for_module(self, module_path: str) -> List[int]:
        """Flattens all anchors within a module/directory."""
        anchors = []
        if module_path in self.index:
            for file_anchors in self.index[module_path].values():
                anchors.extend(file_anchors)
        return anchors
