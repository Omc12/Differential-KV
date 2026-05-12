import torch
import os
import pickle
from typing import Dict, List, Optional

class HierarchicalReasoningArchive:
    """
    Long-term storage for hierarchical reasoning motifs and compressed attractors.
    """
    def __init__(self, archive_path: str = "reasoning_archive"):
        self.archive_path = archive_path
        os.makedirs(archive_path, exist_ok=True)
        self.motif_index = {}

    def archive_motif(self, category: str, motif: torch.Tensor, metadata: Dict):
        """
        Stores a reasoning motif under a specific category.
        """
        category_dir = os.path.join(self.archive_path, category)
        os.makedirs(category_dir, exist_ok=True)
        
        motif_id = f"motif_{len(self.motif_index)}"
        file_path = os.path.join(category_dir, f"{motif_id}.pt")
        
        torch.save({
            "motif": motif,
            "metadata": metadata
        }, file_path)
        
        self.motif_index[motif_id] = {
            "path": file_path,
            "category": category,
            "metadata": metadata
        }

    def query_motifs(self, category: Optional[str] = None) -> List[torch.Tensor]:
        """
        Retrieves archived motifs, optionally filtered by category.
        """
        results = []
        for m_id, info in self.motif_index.items():
            if category is None or info["category"] == category:
                data = torch.load(info["path"])
                results.append(data["motif"])
        return results

    def save_index(self):
        """
        Saves the archive index to disk.
        """
        with open(os.path.join(self.archive_path, "index.pkl"), "wb") as f:
            pickle.dump(self.motif_index, f)

    def load_index(self):
        """
        Loads the archive index from disk.
        """
        index_path = os.path.join(self.archive_path, "index.pkl")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self.motif_index = pickle.load(f)
