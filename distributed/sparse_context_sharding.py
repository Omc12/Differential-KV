import torch

class SparseContextSharding:
    """
    Shards sparse KV contexts across multiple GPUs/nodes.
    Optimizes for balanced retrieval load.
    """
    def __init__(self, num_shards: int):
        self.num_shards = num_shards

    def shard_context(self, kv_cache: torch.Tensor, mask: torch.Tensor):
        """
        Splits the sparse context into shards.
        Attempts to distribute high-importance (mask=True) tokens evenly.
        """
        # mask: [seq_len]
        active_indices = torch.where(mask)[0]
        
        shards = []
        shard_size = len(active_indices) // self.num_shards
        
        for i in range(self.num_shards):
            start = i * shard_size
            end = (i + 1) * shard_size if i < self.num_shards - 1 else len(active_indices)
            shards.append(active_indices[start:end])
            
        return shards
