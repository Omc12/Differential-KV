"""
repositories/cross_file_memory_router.py

Phase 12B: Cross-File Memory Router
Routes sparse retrieval requests across multiple files, enabling reasoning
that spans an entire repository.
"""

from typing import List, Dict, Optional
from anchor_logic.semantic_anchor_system import SemanticAnchor, SemanticAnchorMemory
from repositories.hierarchical_repo_index import HierarchicalRepoIndex

class CrossFileMemoryRouter:
    """
    Coordinates retrieval across different files.
    When a query is made, it decides which files' anchors should be 
    activated and injected into the current context.
    """
    def __init__(self, index: HierarchicalRepoIndex, memory: SemanticAnchorMemory):
        self.index = index
        self.memory = memory

    def route_query(self, query: str, active_file: Optional[str] = None) -> List[SemanticAnchor]:
        """
        Decision logic for cross-file retrieval.
        1. Find files related to the query.
        2. Filter anchors by relevance/importance.
        3. Return a list of anchors to inject.
        """
        # Search index for related files
        related_files = self.index.search(query)
        
        # Also include the active file if provided
        if active_file and active_file not in related_files:
            related_files.append(active_file)

        anchors_to_inject = []
        for rel_path in related_files:
            # Retrieve anchor IDs for these files
            module = str(self.index.repo_root / rel_path).replace(str(self.index.repo_root), "").strip("/")
            # This is a simplification; index structure depends on implementation
            # For now, let's assume we can get anchors by rel_path
            anchor_ids = self.index.get_anchors_for_module(rel_path) # Mock usage
            
            for aid in anchor_ids:
                if aid in self.memory.anchors:
                    anchors_to_inject.append(self.memory.anchors[aid])
        
        # Sort by importance and limit to avoid context overflow
        anchors_to_inject.sort(key=lambda x: x.importance_score, reverse=True)
        return anchors_to_inject[:128] # Max budget for cross-file injection
