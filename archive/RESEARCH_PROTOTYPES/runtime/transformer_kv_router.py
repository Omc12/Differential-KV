import torch
from typing import Optional, Tuple

class TransformerKVRouter:
    """
    Routes KV requests between the dense (standard) and sparse (DKV) paths.
    Ensures that the model receives the correct KV state regardless of sparsity level.
    """
    def __init__(self, manager):
        self.manager = manager

    def route_kv(self, 
                 layer_idx: int, 
                 dense_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines which KV state to provide to the attention layer.
        Can return dense KV, reconstructed sparse KV, or a hybrid.
        """
        if self.manager.is_sparse_active(layer_idx):
            return self.manager.get_sparse_kv(layer_idx)
        return dense_kv
