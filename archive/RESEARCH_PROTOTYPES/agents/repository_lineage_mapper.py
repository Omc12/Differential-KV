"""
agents/repository_lineage_mapper.py

Tracks anchors across branches and commits.
Allows agents to maintain context when switching between git branches.
"""

from typing import Dict, List, Set, Any
import logging

class RepositoryLineageMapper:
    """
    Lineage tracker for cross-branch retrieval.
    """
    def __init__(self):
        self.branch_anchors: Dict[str, List[int]] = {} # branch -> shard_ids
        self.common_anchors: Set[int] = set()
        self.logger = logging.getLogger("RepositoryLineageMapper")

    def register_branch_context(self, branch_name: str, shard_ids: List[int]):
        """Records the anchors active on a specific branch."""
        self.branch_anchors[branch_name] = shard_ids
        
        # Update common set (anchors that appear in all branches)
        if not self.common_anchors:
            self.common_anchors = set(shard_ids)
        else:
            self.common_anchors &= set(shard_ids)

    def get_transferable_context(self, from_branch: str, to_branch: str) -> List[int]:
        """
        Returns anchors from 'from_branch' that are safe to use in 'to_branch'.
        """
        if from_branch not in self.branch_anchors or to_branch not in self.branch_anchors:
            return list(self.common_anchors)
            
        # Return intersection
        return list(set(self.branch_anchors[from_branch]) & set(self.branch_anchors[to_branch]))
