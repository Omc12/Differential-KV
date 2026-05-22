"""
runtime/sparse_prefill_mlp.py

Phase 13 — Prefill-Aware MLP Sparsity (Sequence-Aware Routing)

During long-context prefill, tokens exhibit high semantic locality. Nearby tokens 
(e.g., within a 32-token or 64-token block) tend to activate the same FFN regions.

Instead of computing the routing signal (gate_proj) for EVERY single token independently
(which costs 1/3 of total MLP FLOPs), this module implements Sequence-Aware Clustered Routing:

1. Divide the prefill sequence into clusters (e.g., 64 tokens).
2. Sub-sample the cluster (e.g., take 4 representative tokens).
3. Compute gate_proj ONLY on the representatives.
4. Determine the active blocks for the cluster.
5. Execute the sparse FFN for the ENTIRE cluster using the shared block mask.

This drastically reduces FLOPs and memory traffic for the gate_proj during prefill,
leveraging the "Region / Locality Execution" principle.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

class PrefillSparseMLP:
    """
    Executes sequence-aware clustered sparse MLP for prefill workloads.
    """
    def __init__(
        self,
        block_size: int = 128,
        keep_ratio: float = 0.5,
        cluster_size: int = 64,
        subsample_size: int = 4
    ):
        self.block_size = block_size
        self.keep_ratio = keep_ratio
        self.cluster_size = cluster_size
        self.subsample_size = subsample_size
        
        self.stats = {
            "gate_flops": 0,
            "sparse_flops": 0,
            "dense_flops_equivalent": 0
        }

    def forward(
        self,
        x: torch.Tensor,         # [bsz, seq_len, hidden]
        W_gate: torch.Tensor,    # [d_ff, hidden]
        W_up: torch.Tensor,      # [d_ff, hidden]
        W_down: torch.Tensor,    # [hidden, d_ff]
    ) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        d_ff = W_gate.shape[0]
        total_blocks = d_ff // self.block_size
        
        out = torch.zeros_like(x)
        
        # We process in 2D to simplify
        x_2d = x.reshape(bsz * seq_len, hidden)
        out_2d = out.reshape(bsz * seq_len, hidden)
        total_tokens = bsz * seq_len
        
        # Dense baseline for stats
        self.stats["dense_flops_equivalent"] += total_tokens * (3 * 2 * hidden * d_ff)
        
        # Process cluster by cluster
        for start_idx in range(0, total_tokens, self.cluster_size):
            end_idx = min(start_idx + self.cluster_size, total_tokens)
            cluster_len = end_idx - start_idx
            
            x_cluster = x_2d[start_idx:end_idx]
            
            # ── 1. Sub-sample for routing ──────────────────────────────────────
            # Uniformly sample tokens from the cluster
            step = max(1, cluster_len // self.subsample_size)
            sample_indices = torch.arange(0, cluster_len, step, device=x.device)[:self.subsample_size]
            x_samples = x_cluster[sample_indices]
            
            # Compute gate ONLY on samples
            gate_samples = F.linear(x_samples, W_gate)
            gate_silu = F.silu(gate_samples)
            
            # Accumulate gate FLOPs (drastically reduced)
            self.stats["gate_flops"] += len(sample_indices) * (2 * hidden * d_ff)
            
            # ── 2. Determine shared active blocks ─────────────────────────────
            # Aggregate importance across samples
            gate_blocked = gate_silu.abs().view(len(sample_indices), total_blocks, self.block_size).mean(dim=(0, 2))
            
            k_blocks = max(1, int(total_blocks * self.keep_ratio))
            _, active_block_ids = torch.topk(gate_blocked, k_blocks, sorted=False)
            active_block_ids, _ = torch.sort(active_block_ids)
            
            offsets = active_block_ids * self.block_size
            active_ids = (offsets.unsqueeze(1) + torch.arange(self.block_size, device=x.device).unsqueeze(0)).reshape(-1)
            
            # ── 3. Execute Sparse FFN for the entire cluster ──────────────────
            # (In a fully native kernel, this would bypass index_select, but for 
            # PyTorch demonstration of the logic, we slice).
            W_gate_sparse = W_gate[active_ids, :]
            W_up_sparse = W_up[active_ids, :]
            W_down_sparse = W_down[:, active_ids]
            
            # Now we compute gate_proj on the full cluster, but ONLY for active blocks!
            # (Previously we computed full gate_proj on all tokens)
            gate_full = F.linear(x_cluster, W_gate_sparse)
            up_full = F.linear(x_cluster, W_up_sparse)
            
            mixed = F.silu(gate_full) * up_full
            out_cluster = F.linear(mixed, W_down_sparse)
            
            out_2d[start_idx:end_idx] = out_cluster
            
            # FLOP accounting: 
            # gate_sparse + up_sparse + down_sparse
            k_active = active_ids.shape[0]
            self.stats["sparse_flops"] += cluster_len * (3 * 2 * hidden * k_active)
            
        return out
        
    def get_summary(self) -> dict:
        total_sparse = self.stats["gate_flops"] + self.stats["sparse_flops"]
        dense = self.stats["dense_flops_equivalent"]
        return {
            "dense_flops": dense,
            "actual_flops": total_sparse,
            "reduction_pct": round((1.0 - total_sparse / max(1, dense)) * 100, 2)
        }
