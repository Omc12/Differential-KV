"""
runtime/sparse_mlp.py

Phase 11 — Real Block-Sparse MLP Execution

Engineering reality:
  - Qwen2 MLP: gate_proj (full), up_proj (sparse), down_proj (sparse)
  - gate_proj MUST run fully to compute routing signals
  - up_proj and down_proj execute only for active neuron blocks
  - Block size 128: tensor-core aligned, contiguous memory, no scatter

FLOP math (Qwen2-7B, decode seq=1):
  Dense:  3 × (1 × 3584 × 18944) = 203M FLOPs/layer
  Sparse: 1 × (1 × 3584 × 18944)    [gate_proj: always full]
        + 2 × (1 × 3584 × k_active) [up_proj + down_proj: sparse]
  
  At 50% neuron keep: 1 + 2×0.5 = 2.0 matmuls vs 3.0 = 33% FLOP reduction
  At 30% neuron keep: 1 + 2×0.3 = 1.6 matmuls vs 3.0 = 47% FLOP reduction

Memory bandwidth math:
  Dense load: (18944×3584 + 18944×3584 + 3584×18944) × 2 bytes = 406MB/layer
  Sparse 50%: (18944×3584 + 9472×3584 + 3584×9472) × 2 bytes = 272MB/layer = 33% less

This is the dominant gain mechanism on bandwidth-bound GPUs during decode.

Routing signal: gate activation magnitude (real, not random)
  SiLU(gate) * up governs each neuron's contribution.
  Low-magnitude gate outputs → that block contributes ~0 to down_proj.
  We compute gate_proj fully, then select top-k blocks by block-mean |gate|.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


# Block size: must be power of 2, >= 64 for tensor-core alignment
# 128 is optimal for A100/RTX-class GPUs
BLOCK_SIZE = 128


@dataclass
class SparsityStats:
    """Tracks real sparsity measurements per call."""
    keep_ratio: float = 1.0          # fraction of neurons actually computed
    active_blocks: int = 0
    total_blocks: int = 0
    gate_l1_mean: float = 0.0        # mean |gate| — diagnostic for routing collapse
    skipped_flop_fraction: float = 0.0

    def __post_init__(self):
        if self.total_blocks > 0:
            self.keep_ratio = self.active_blocks / self.total_blocks
            self.skipped_flop_fraction = (1.0 - self.keep_ratio) * (2 / 3)  # only up+down sparse


class BlockSparseMLPExecutor:
    """
    Real block-sparse MLP executor.

    Replaces Qwen2MLP.forward with a path that:
      1. Runs gate_proj fully (routing signal)
      2. Selects top-k blocks by block-mean |gate|
      3. Runs up_proj + down_proj only for active blocks

    This produces GENUINE FLOP and memory bandwidth reduction.

    Parameters
    ----------
    block_size   : neurons per block (must be divisor of intermediate_size)
    keep_ratio   : fraction of blocks to keep (0.0-1.0)
    min_keep     : minimum number of blocks to always keep (prevents collapse)
    """

    def __init__(
        self,
        block_size:  int   = BLOCK_SIZE,
        keep_ratio:  float = 0.5,
        min_keep:    int   = 8,
    ):
        self.block_size = block_size
        self.keep_ratio = keep_ratio
        self.min_keep   = min_keep

        # Rolling diagnostic stats
        self._call_count = 0
        self._total_keep  = 0.0
        self._total_gate_l1 = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Core sparse forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,     # [bsz, seq, hidden]
        gate_proj:  torch.nn.Linear,
        up_proj:    torch.nn.Linear,
        down_proj:  torch.nn.Linear,
        act_fn,
    ) -> Tuple[torch.Tensor, SparsityStats]:
        """
        Real sparse MLP forward.

        Returns
        -------
        output : [bsz, seq, hidden]
        stats  : SparsityStats with real measurements
        """
        bsz, seq, hidden = x.shape
        d_ff = gate_proj.weight.shape[0]  # intermediate_size
        total_blocks = d_ff // self.block_size

        # ── Step 1: Full gate projection (routing signal) ─────────────────────
        # This must run fully — it IS the routing computation.
        gate_vals = F.linear(x, gate_proj.weight, gate_proj.bias)  # [bsz, seq, d_ff]
        gate_activated = act_fn(gate_vals)                          # SiLU applied

        # ── Step 2: Block importance scoring ─────────────────────────────────
        # Mean absolute gate activation per block — real importance signal.
        # Reshape: [bsz, seq, total_blocks, block_size]
        gate_blocked = gate_activated.view(bsz, seq, total_blocks, self.block_size)
        block_importance = gate_blocked.abs().mean(dim=(0, 1, 3))  # [total_blocks]

        # ── Step 3: Top-k block selection ────────────────────────────────────
        k_blocks = max(self.min_keep, int(total_blocks * self.keep_ratio))
        k_blocks = min(k_blocks, total_blocks)
        _, top_block_ids = torch.topk(block_importance, k_blocks, sorted=False)
        top_block_ids, _ = torch.sort(top_block_ids)  # sort for contiguous-ish access

        # Convert block ids → neuron index ranges
        # active_neuron_ids: [k_blocks * block_size] — contiguous within each block
        offsets = top_block_ids * self.block_size           # [k_blocks]
        neuron_ids = (
            offsets.unsqueeze(1) +
            torch.arange(self.block_size, device=x.device).unsqueeze(0)
        ).reshape(-1)                                        # [k_blocks * block_size]

        k_neurons = k_blocks * self.block_size

        # ── Step 4: Sparse up_proj ────────────────────────────────────────────
        # Gather only active rows of W_up: [k_neurons, hidden]
        # This is real FLOP reduction: [bsz, seq, hidden] × [hidden, k_neurons]
        # vs dense: [bsz, seq, hidden] × [hidden, d_ff]
        W_up_sparse = up_proj.weight[neuron_ids, :]         # [k_neurons, hidden]
        up_vals_sparse = F.linear(x, W_up_sparse)           # [bsz, seq, k_neurons]

        # ── Step 5: Sparse gate × up ─────────────────────────────────────────
        gate_sparse = gate_activated[..., neuron_ids]       # [bsz, seq, k_neurons]
        mixed = gate_sparse * up_vals_sparse                # [bsz, seq, k_neurons]

        # ── Step 6: Sparse down_proj ──────────────────────────────────────────
        # Gather active columns of W_down: [hidden, k_neurons]
        # Real reduction: [bsz, seq, k_neurons] × [k_neurons, hidden]
        # vs dense: [bsz, seq, d_ff] × [d_ff, hidden]
        W_down_sparse = down_proj.weight[:, neuron_ids]     # [hidden, k_neurons]
        # F.linear(x, W) = x @ W.T
        # W_down_sparse = [hidden, k_neurons], so W.T = [k_neurons, hidden]
        # mixed [bsz, seq, k_neurons] @ [k_neurons, hidden] = [bsz, seq, hidden]  OK
        output = F.linear(mixed, W_down_sparse)              # [bsz, seq, hidden]

        # ── Step 7: Diagnostics ───────────────────────────────────────────────
        stats = SparsityStats(
            keep_ratio           = k_neurons / d_ff,
            active_blocks        = k_blocks,
            total_blocks         = total_blocks,
            gate_l1_mean         = block_importance.mean().item(),
            skipped_flop_fraction= (1.0 - k_neurons / d_ff) * (2 / 3),
        )
        self._call_count     += 1
        self._total_keep     += stats.keep_ratio
        self._total_gate_l1  += stats.gate_l1_mean

        return output, stats

    # ──────────────────────────────────────────────────────────────────────────
    # Dense baseline (for quality comparison)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def dense_forward(
        x:         torch.Tensor,
        gate_proj: torch.nn.Linear,
        up_proj:   torch.nn.Linear,
        down_proj: torch.nn.Linear,
        act_fn,
    ) -> torch.Tensor:
        """Original dense Qwen2 MLP forward. Used as quality baseline."""
        return down_proj(act_fn(gate_proj(x)) * up_proj(x))

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        n = max(1, self._call_count)
        return {
            "calls":            self._call_count,
            "avg_keep_ratio":   round(self._total_keep / n, 4),
            "avg_gate_l1":      round(self._total_gate_l1 / n, 4),
            "block_size":       self.block_size,
            "configured_keep":  self.keep_ratio,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Patcher: replaces Qwen2MLP.forward transparently
# ─────────────────────────────────────────────────────────────────────────────

class SparseMLP:
    """
    Drop-in replacement for Qwen2MLP.forward.

    Usage:
        sparse = SparseMLP(model, keep_ratio=0.5)
        sparse.patch()
        # Model now uses sparse MLP execution
        sparse.unpatch()
        # Restored to dense
    """

    def __init__(self, model, keep_ratio: float = 0.5, block_size: int = BLOCK_SIZE):
        self.model      = model
        self.keep_ratio = keep_ratio
        self.block_size = block_size
        self.executor   = BlockSparseMLPExecutor(block_size=block_size, keep_ratio=keep_ratio)
        self._patched   = False
        self._originals = {}    # layer_idx -> original forward fn
        self.layer_stats: dict = {}   # layer_idx -> last SparsityStats

    def patch(self):
        """Monkey-patch all Qwen2MLP layers with sparse forward."""
        if self._patched:
            return
        import types

        executor = self.executor
        sparse_mlp = self   # capture for closure

        for i, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            self._originals[i] = mlp.forward
            idx = i  # capture loop var

            def make_sparse_forward(mlp_module, layer_idx):
                def sparse_forward(x):
                    out, stats = executor.forward(
                        x,
                        mlp_module.gate_proj,
                        mlp_module.up_proj,
                        mlp_module.down_proj,
                        mlp_module.act_fn,
                    )
                    sparse_mlp.layer_stats[layer_idx] = stats
                    return out
                return sparse_forward

            mlp.forward = make_sparse_forward(mlp, idx)

        self._patched = True

    def unpatch(self):
        """Restore original dense MLP forward for all layers."""
        if not self._patched:
            return
        for i, layer in enumerate(self.model.model.layers):
            if i in self._originals:
                layer.mlp.forward = self._originals[i]
        self._originals.clear()
        self._patched = False

    def get_stats(self) -> dict:
        if not self.layer_stats:
            return {}
        keeps   = [s.keep_ratio for s in self.layer_stats.values()]
        skipped = [s.skipped_flop_fraction for s in self.layer_stats.values()]
        return {
            "patched_layers":        len(self.layer_stats),
            "avg_keep_ratio":        round(sum(keeps) / len(keeps), 4),
            "avg_flop_skip_fraction": round(sum(skipped) / len(skipped), 4),
            "executor":              self.executor.summary(),
        }
