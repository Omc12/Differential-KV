import torch

class RetrievalBridgeTokens:
    """
    Implements 'bridge tokens' that summarize blocks of context.
    These are inserted into the sequence to allow long-range retrieval without full KV.
    """
    def __init__(self, bridge_frequency: int = 1024):
        self.bridge_frequency = bridge_frequency

    def should_insert_bridge(self, current_pos: int) -> bool:
        return current_pos > 0 and current_pos % self.bridge_frequency == 0

    def generate_bridge_token(self, block_kv: torch.Tensor) -> torch.Tensor:
        """
        Generates a bridge token representation from a block of KV.
        This is a grounded heuristic: mean or max pooling of the block.
        """
        # [batch, heads, block_len, head_dim] -> [batch, heads, 1, head_dim]
        bridge = torch.mean(block_kv, dim=-2, keepdim=True)
        return bridge
