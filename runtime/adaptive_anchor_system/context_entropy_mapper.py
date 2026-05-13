import torch
import torch.nn.functional as F

class ContextEntropyMapper:
    def map_sequence_entropy(self, attn_weights: torch.Tensor, bucket_size: int = 128) -> torch.Tensor:
        mean_attn = attn_weights.mean(dim=0)
        p = F.softmax(mean_attn, dim=-1)
        token_entropy = -(p * torch.log(p + 1e-9)).sum(dim=0)
        num_buckets = (token_entropy.size(0) + bucket_size - 1) // bucket_size
        buckets = []
        for i in range(num_buckets):
            start = i * bucket_size
            end = min((i + 1) * bucket_size, token_entropy.size(0))
            buckets.append(token_entropy[start:end].mean())
        return torch.stack(buckets)
