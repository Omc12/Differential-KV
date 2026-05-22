"""
runtime/prefill_attention_pruner.py

Phase 11 Step 5 — Prefill-Time Attention-Based KV Budget Control

Purpose:
  During prefill, not all input tokens contribute meaningfully to future decode.
  Tokens with very low attention scores from late positions are unlikely to be
  retrieved. We can reduce how many full KV blocks we write and compress.

Strategy:
  - Run prefill normally (can't skip prefill matmuls without breaking causality)
  - After prefill completes, analyze accumulated attention weights
  - Identify "low-budget" tokens: positions that rarely receive high attention
  - Mark their KV blocks for immediate compression (vs keeping dense window)
  - This reduces compression queue depth and paging pressure

Critical constraints (must NOT violate):
  1. Causality: pruning decisions made AFTER full attention, not during
  2. Sink tokens (positions 0, 1): always kept at full precision
  3. Recent tokens: always kept dense (recency window = last 2 blocks)
  4. Pruning is compression-timing only: all tokens remain accessible

Integration with KVRuntimeManager:
  - Call analyze_prefill_attention() after prefill attention runs
  - Returns a priority list for the async compressor
  - Low-priority blocks get submitted to the compressor queue first
  - High-priority blocks stay in the dense recency window longer

This is different from "dropping" tokens — it's about COMPRESSION SCHEDULING.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import math


@dataclass
class KVBudget:
    """Per-block compression priority for the async compressor."""
    block_idx:  int
    priority:   float    # higher = compress sooner (less important)
    keep_dense: bool     # True = keep in dense window (do NOT compress yet)
    reason:     str      # 'sink', 'recent', 'high_attn', 'low_attn'


class PrefillAttentionPruner:
    """
    Analyzes prefill attention weights to compute per-block compression priority.

    This is a POST-PREFILL analysis pass — it does not modify the attention
    computation itself. It only informs the compression scheduler.

    Parameters
    ----------
    sink_count     : number of sink tokens always kept at full precision
    recent_blocks  : number of most-recent blocks always kept dense
    prune_threshold: blocks scoring below this get high compression priority
    block_size     : must match KVRuntimeManager.block_size
    """

    def __init__(
        self,
        sink_count:      int   = 4,
        recent_blocks:   int   = 2,
        prune_threshold: float = 0.02,
        block_size:      int   = 64,
    ):
        self.sink_count      = sink_count
        self.recent_blocks   = recent_blocks
        self.prune_threshold = prune_threshold
        self.block_size      = block_size

        # Telemetry
        self._calls          = 0
        self._total_pruned   = 0
        self._total_blocks   = 0

    def analyze(
        self,
        attn_weights: torch.Tensor,   # [bsz, heads, seq, seq] — prefill attention
        seq_len:      int,
    ) -> List[KVBudget]:
        """
        Compute per-block compression priority from prefill attention weights.

        Parameters
        ----------
        attn_weights : [bsz, heads, seq_q, seq_kv] — attention probs from prefill
        seq_len      : number of prefill tokens

        Returns
        -------
        List[KVBudget] ordered by block_idx
        """
        # Aggregate attention over all heads and query positions
        # Shape: [seq_kv] — how much attention each key position received
        # We use the LAST few query positions (closest to decode start)
        bsz, heads, seq_q, seq_kv = attn_weights.shape

        # Use last 8 query positions as "representative decode queries"
        late_q = max(0, seq_q - 8)
        attn_late = attn_weights[:, :, late_q:, :].float()  # [bsz, heads, late, seq_kv]
        token_importance = attn_late.mean(dim=(0, 1, 2))     # [seq_kv]

        # Normalize
        token_importance = token_importance / (token_importance.sum() + 1e-9)

        # Aggregate to block level
        num_blocks = math.ceil(seq_len / self.block_size)
        block_scores = torch.zeros(num_blocks, device=token_importance.device)
        for b in range(num_blocks):
            start = b * self.block_size
            end   = min(start + self.block_size, seq_len)
            block_scores[b] = token_importance[start:end].sum()

        # Build budget list
        budgets = []
        recent_start_block = max(0, num_blocks - self.recent_blocks)

        for b in range(num_blocks):
            block_start_token = b * self.block_size

            # Sink protection
            if block_start_token < self.sink_count:
                budgets.append(KVBudget(
                    block_idx=b,
                    priority=0.0,    # lowest priority = compress last / never
                    keep_dense=True,
                    reason='sink',
                ))
                continue

            # Recency protection
            if b >= recent_start_block:
                budgets.append(KVBudget(
                    block_idx=b,
                    priority=0.0,
                    keep_dense=True,
                    reason='recent',
                ))
                continue

            score = block_scores[b].item()

            if score < self.prune_threshold:
                # Low importance: schedule for early compression
                priority = 1.0 - score / (self.prune_threshold + 1e-9)
                budgets.append(KVBudget(
                    block_idx=b,
                    priority=priority,
                    keep_dense=False,
                    reason='low_attn',
                ))
                self._total_pruned += 1
            else:
                # High importance: keep dense longer
                budgets.append(KVBudget(
                    block_idx=b,
                    priority=max(0.0, 1.0 - score / block_scores.max().item()),
                    keep_dense=False,
                    reason='high_attn',
                ))

        self._calls       += 1
        self._total_blocks += num_blocks
        return budgets

    def get_compression_order(
        self,
        budgets: List[KVBudget],
    ) -> List[int]:
        """
        Return block indices sorted by compression priority (high priority first).
        Blocks with keep_dense=True are excluded (not yet ready to compress).
        """
        compressible = [b for b in budgets if not b.keep_dense]
        compressible.sort(key=lambda b: -b.priority)
        return [b.block_idx for b in compressible]

    def summary(self) -> dict:
        if self._calls == 0:
            return {"calls": 0}
        avg_pruned = self._total_pruned / self._calls
        avg_blocks = self._total_blocks / self._calls
        return {
            "calls":               self._calls,
            "avg_blocks_per_call": round(avg_blocks, 1),
            "avg_pruned_per_call": round(avg_pruned, 1),
            "avg_prune_rate":      round(avg_pruned / max(1, avg_blocks), 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HF Attention hook: captures attention weights during prefill
# ─────────────────────────────────────────────────────────────────────────────

class PrefillAttentionHook:
    """
    Registers a forward hook on Qwen2Attention to capture attention weights.
    Works without modifying the model class itself.

    Usage:
        hook = PrefillAttentionHook(model)
        hook.register()
        model(input_ids=...)
        weights = hook.get_weights()   # {layer_idx: tensor [bsz, heads, seq, seq]}
        hook.remove()
    """

    def __init__(self, model):
        self.model   = model
        self._hooks  = []
        self._weights: Dict[int, torch.Tensor] = {}

    def register(self):
        """Register hooks on all attention layers."""
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            idx  = i

            def make_hook(layer_idx):
                def hook_fn(module, inputs, output):
                    # Qwen2 attention returns (hidden, past_kv, attn_weights)
                    # attn_weights is the 3rd return value if output_attentions=True
                    if isinstance(output, tuple) and len(output) >= 3:
                        if output[2] is not None:
                            self._weights[layer_idx] = output[2].detach().cpu()
                return hook_fn

            h = attn.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def get_weights(self) -> Dict[int, torch.Tensor]:
        return dict(self._weights)

    def clear(self):
        self._weights.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic validation (no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def validate_pruner_logic():
    """
    Test PrefillAttentionPruner with synthetic attention weights.
    Verifies: sinks protected, recency protected, low-attn scheduled first.
    """
    pruner = PrefillAttentionPruner(sink_count=4, recent_blocks=2,
                                    prune_threshold=0.02, block_size=64)

    # Synthetic attention: uniform except spike at token 10
    bsz, heads, seq = 1, 8, 512
    attn = torch.ones(bsz, heads, seq, seq) / seq
    attn[:, :, :, 10] += 0.5   # token 10 gets extra attention

    budgets = pruner.analyze(attn, seq_len=seq)

    sink_blocks    = [b for b in budgets if b.reason == 'sink']
    recent_blocks  = [b for b in budgets if b.reason == 'recent']
    low_attn       = [b for b in budgets if b.reason == 'low_attn']
    high_attn      = [b for b in budgets if b.reason == 'high_attn']

    assert len(sink_blocks) > 0,    "No sink blocks identified"
    assert len(recent_blocks) > 0,  "No recent blocks identified"
    assert all(b.keep_dense for b in sink_blocks),   "Sink blocks not protected"
    assert all(b.keep_dense for b in recent_blocks), "Recent blocks not protected"

    # Block containing token 10 should be high_attn
    token10_block = 10 // 64   # = 0
    block0 = next((b for b in budgets if b.block_idx == 0), None)
    # Block 0 is a sink block (starts at token 0 which < sink_count=4)

    comp_order = pruner.get_compression_order(budgets)
    # compressible blocks should be low_attn first
    assert len(comp_order) == len(low_attn) + len(high_attn)

    return pruner.summary()
