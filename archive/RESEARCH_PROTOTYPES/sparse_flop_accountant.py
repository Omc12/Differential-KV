import torch
import time
from typing import Dict, Any

class SparseFlopAccountant:
    """
    Real FLOP accounting for Differential KV.
    Tracks dense, sparse, and skipped operations.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.stats = {
            "total_flops_dense": 0,
            "total_flops_sparse": 0,
            "total_flops_skipped": 0,
            "active_tokens": 0,
            "total_tokens": 0,
            "participation_count": 0
        }

    def record_attention(self, q_len: int, k_len: int, d_model: int, n_heads: int, sparse_k_len: int):
        """
        Record FLOPs for an attention operation.
        Standard Attention FLOPs (QK^T V): 4 * q_len * k_len * d_model * n_heads
        """
        dense_flops = 4 * q_len * k_len * d_model * n_heads
        sparse_flops = 4 * q_len * sparse_k_len * d_model * n_heads
        skipped_flops = dense_flops - sparse_flops

        self.stats["total_flops_dense"] += dense_flops
        self.stats["total_flops_sparse"] += sparse_flops
        self.stats["total_flops_skipped"] += skipped_flops
        self.stats["active_tokens"] += sparse_k_len
        self.stats["total_tokens"] += k_len
        self.stats["participation_count"] += 1

    def record_mlp(self, bsz: int, seq_len: int, d_model: int, d_ff: int, is_sparse: bool = False, sparse_ratio: float = 1.0):
        """
        Record FLOPs for MLP.
        Dense MLP (2 * MatMul): 2 * (2 * bsz * seq_len * d_model * d_ff)
        """
        dense_flops = 4 * bsz * seq_len * d_model * d_ff
        if is_sparse:
            sparse_flops = int(dense_flops * sparse_ratio)
            skipped_flops = dense_flops - sparse_flops
        else:
            sparse_flops = dense_flops
            skipped_flops = 0

        self.stats["total_flops_dense"] += dense_flops
        self.stats["total_flops_sparse"] += sparse_flops
        self.stats["total_flops_skipped"] += skipped_flops

    def get_metrics(self) -> Dict[str, Any]:
        dense = self.stats["total_flops_dense"]
        sparse = self.stats["total_flops_sparse"]
        reduction = (1.0 - (sparse / dense)) * 100 if dense > 0 else 0
        
        active_ratio = (self.stats["active_tokens"] / self.stats["total_tokens"]) * 100 if self.stats["total_tokens"] > 0 else 0

        return {
            "total_flops_dense": dense,
            "total_flops_sparse": sparse,
            "total_flops_skipped": self.stats["total_flops_skipped"],
            "real_compute_reduction_percent": reduction,
            "active_attention_ratio": active_ratio
        }

    def report(self):
        metrics = self.get_metrics()
        print("\n" + "="*40)
        print("SPARSE FLOP ACCOUNTING REPORT")
        print("="*40)
        print(f"Dense FLOPs:   {metrics['total_flops_dense']:,}")
        print(f"Sparse FLOPs:  {metrics['total_flops_sparse']:,}")
        print(f"Skipped FLOPs: {metrics['total_flops_skipped']:,}")
        print(f"Reduction:     {metrics['real_compute_reduction_percent']:.2f}%")
        print(f"Active Tokens: {metrics['active_attention_ratio']:.2f}%")
        print("="*40 + "\n")

# Global singleton for runtime accounting
accountant = SparseFlopAccountant()
