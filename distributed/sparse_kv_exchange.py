import torch
from typing import List, Dict, Tuple

class SparseKVExchange:
    """
    Handles efficient KV exchange between distributed nodes.
    Only synchronizes 'anchor' tokens and high-priority subsets.
    """
    def __init__(self, node_id: int, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes

    def prepare_sync_package(self, layer_id: int, kv_cache: Tuple[torch.Tensor, torch.Tensor], anchor_indices: torch.Tensor) -> Dict:
        """
        Extract anchors for synchronization.
        """
        k, v = kv_cache
        sync_k = k[:, :, anchor_indices, :]
        sync_v = v[:, :, anchor_indices, :]
        
        return {
            "node_id": self.node_id,
            "layer_id": layer_id,
            "indices": anchor_indices.tolist(),
            "k": sync_k,
            "v": sync_v
        }

    def merge_remote_kv(self, local_kv: Tuple[torch.Tensor, torch.Tensor], remote_package: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge received KV anchors into local cache.
        """
        # In a real distributed system, this would involve complex index mapping.
        # Here we just show the logic of integrating remote anchors.
        k, v = local_kv
        rk, rv = remote_package["k"], remote_package["v"]
        
        # Simple concatenation for demonstration
        return torch.cat([k, rk], dim=-2), torch.cat([v, rv], dim=-2)
