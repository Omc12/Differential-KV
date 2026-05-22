"""
memory/compressed_attractor_library.py
Phase 26: Cognitive Energy Minimization (CEM)
Storage for compressed attractor signatures and centroids.
"""

import torch
from typing import Dict, List, Any, Optional

class CompressedAttractorLibrary:
    """
    Persistent store for compressed cognitive attractors.
    Allows the runtime to quickly retrieve stable manifolds for reconstruction.
    """
    def __init__(self):
        # Keyed by layer_idx or a semantic hash
        self.library: Dict[str, Dict[str, Any]] = {}

    def store_attractor(self, identifier: str, compressed_data: Dict[str, Any]):
        """Stores a compressed attractor representation."""
        self.library[identifier] = compressed_data

    def get_attractor(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieves a compressed attractor if it exists."""
        return self.library.get(identifier)

    def list_known_attractors(self) -> List[str]:
        """Returns identifiers of all stored attractors."""
        return list(self.library.keys())

    def clear(self):
        """Wipes the library."""
        self.library = {}

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the library usage and overhead."""
        total_entries = len(self.library)
        # Estimate size in bytes (approximate)
        total_bytes = 0
        for entry in self.library.values():
            if "centroid_compressed" in entry:
                total_bytes += entry["centroid_compressed"].nbytes
        
        return {
            "total_attractors": total_entries,
            "estimated_memory_kb": total_bytes / 1024.0,
            "avg_attractor_size_bytes": (total_bytes / total_entries) if total_entries > 0 else 0
        }
