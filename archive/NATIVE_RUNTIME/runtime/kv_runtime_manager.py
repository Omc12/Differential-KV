"""
runtime/kv_runtime_manager.py

Phase 7 — Paged Sparse Memory, Reconstruction Cache, Async Compression.

Manages KV cache residency and on-demand reconstruction for Differential KV.

Phase 7 additions over Phase 5:
  - PagedKVStore: GPU → CPU RAM spillover under memory pressure.
  - ReconstructionCache: LRU cache of recently reconstructed dense blocks to
    eliminate repeated U@V GEMMs.
  - AsyncCompressor: SVD compression moved off the decode hot path to a
    background thread, eliminating stalls.
  - get_raw_blocks() (Phase 6): direct access to compressed block list for
    fused sparse attention.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import sys, os

from runtime.triton_dkv import TritonDKV
from compression.lowrank import compress_lowrank, LowRankDelta
from runtime.paged_kv_store import PagedKVStore
from runtime.recon_cache import ReconstructionCache
from runtime.async_compressor import AsyncCompressor

try:
    from compression.adaptive import AdaptiveRankSelector as _AdaptiveRankSelector
    _ADAPTIVE_RANK = True
except ImportError:
    _ADAPTIVE_RANK = False
    _AdaptiveRankSelector = None


# ─────────────────────────────────────────────────────────────────────────────
# KVBlock definition (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KVBlock:
    """Physically stores compressed KV memory for one block of tokens."""
    anchor_idx: int
    anchor_kv:  torch.Tensor          # [1, 2, heads, head_dim]
    U:          Optional[torch.Tensor] = None   # [block_size, rank]
    V:          Optional[torch.Tensor] = None   # [rank, feat_dim]
    scale:      float = 1.0
    token_indices: List[int] = None
    cosine_sim: float = 1.0
    norm_drift: float = 0.0

    # Optional uncompressed tokens (dense window)
    active_k: Optional[torch.Tensor] = None
    active_v: Optional[torch.Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class KVRuntimeManager:
    """
    Manages KV cache residency and on-demand reconstruction.

    Phase 7:
      - Wraps PagedKVStore for GPU/CPU tiered residency.
      - Uses ReconstructionCache to skip repeat U@V GEMMs.
      - Submits compression jobs to AsyncCompressor.
    """

    def __init__(
        self,
        num_layers:          int,
        heads:               int,
        head_dim:            int,
        device:              str   = "cuda",
        gpu_budget_gb:       float = 2.0,
        recon_cache_size:    int   = 64,
        async_compression:   bool  = True,
        adaptive_rank:       bool  = True,
    ):
        self.num_layers  = num_layers
        self.heads       = heads
        self.head_dim    = head_dim
        self.device      = device
        self.feat_dim    = 2 * heads * head_dim

        # session_id -> layer_idx -> List[KVBlock]
        self.session_blocks: Dict[str, Dict[int, List[KVBlock]]] = {}

        self.block_size           = 64
        self.rank                 = 8    # fallback fixed rank
        self.dense_recency_blocks = 2

        # ── Phase 7 subsystems ────────────────────────────────────────────
        self.pager       = PagedKVStore(gpu_budget_gb=gpu_budget_gb, device=device)
        self.recon_cache = ReconstructionCache(max_entries=recon_cache_size)
        self._async      = async_compression
        self._compressor = AsyncCompressor(compress_fn=self._compress_block_sync)
        if self._async:
            self._compressor.start()

        # ── Salvaged P1: Adaptive Rank Selector ───────────────────────────
        # Selects compression rank based on 95% singular-value energy of each
        # KV delta block.  Falls back to fixed rank=8 if not available.
        self._adaptive_rank = adaptive_rank and _ADAPTIVE_RANK
        if self._adaptive_rank:
            self._rank_selector = _AdaptiveRankSelector(
                rank_buckets=[4, 8, 16, 32],
                method="variance",    # fast variance proxy (no SVD overhead)
            )
        else:
            self._rank_selector = None

        # Telemetry
        self.vram_saved_bytes   = 0
        self.total_compressions = 0
        self.total_cosine_sim   = 0.0
        self.total_norm_drift   = 0.0
        self.rank_histogram     = {}    # rank -> count

    # ── Session management ────────────────────────────────────────────────────

    def init_session(self, session_id: str):
        if session_id not in self.session_blocks:
            self.session_blocks[session_id] = {i: [] for i in range(self.num_layers)}

    def clear_session(self, session_id: str):
        if session_id in self.session_blocks:
            del self.session_blocks[session_id]
        self.pager.evict_session(session_id)

    def clear(self):
        self.session_blocks.clear()
        self.recon_cache.clear()

    # ── Sequence length ───────────────────────────────────────────────────────

    def get_seq_len(self, session_id: str, layer_idx: int = 0) -> int:
        if session_id not in self.session_blocks:
            return 0
        total = 0
        for block in self.session_blocks[session_id][layer_idx]:
            total += 1  # anchor
            if block.U is not None:
                total += block.U.shape[0]
            if block.active_k is not None:
                total += block.active_k.shape[2]
        return total

    # ── Phase 6: raw block access ─────────────────────────────────────────────

    def get_raw_blocks(self, session_id: str, layer_idx: int) -> list:
        """
        Return the raw KVBlock list. Used by fused_sparse_attention_decode.
        Touches each block in the pager to prevent spurious eviction.
        """
        if session_id not in self.session_blocks:
            return []
        blocks = self.session_blocks[session_id][layer_idx]
        for idx, block in enumerate(blocks):
            self.pager.touch(session_id, layer_idx, idx)
        return blocks

    # ── Dense KV reconstruction (legacy path for prefill) ────────────────────

    def get_kv(self, session_id: str, layer_idx: int
               ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Reconstruct full dense KV for prefill path. Uses ReconstructionCache
        to avoid repeat GEMM for already-reconstructed blocks.
        """
        if session_id not in self.session_blocks or \
           not self.session_blocks[session_id][layer_idx]:
            return None, None

        blocks = self.session_blocks[session_id][layer_idx]
        k_list, v_list = [], []

        for idx, block in enumerate(blocks):
            self.pager.touch(session_id, layer_idx, idx)

            # Anchor
            k_list.append(block.anchor_kv[:, 0].unsqueeze(2))
            v_list.append(block.anchor_kv[:, 1].unsqueeze(2))

            if block.U is not None and block.V is not None:
                # Check reconstruction cache first
                cached_k, cached_v = self.recon_cache.get(block)
                if cached_k is not None:
                    k_list.append(cached_k)
                    v_list.append(cached_v)
                else:
                    anchor_flat = block.anchor_kv.view(-1).to(torch.float16)
                    recon = TritonDKV.reconstruct_lowrank(
                        block.U, block.V, anchor_flat, scale=block.scale
                    )
                    hds  = block.anchor_kv.shape[2]
                    hdim = block.anchor_kv.shape[3]
                    recon = recon.view(1, -1, 2, hds, hdim)
                    recon_k = recon[:, :, 0].transpose(1, 2)
                    recon_v = recon[:, :, 1].transpose(1, 2)
                    # Cache for reuse
                    self.recon_cache.put(block, recon_k, recon_v)
                    k_list.append(recon_k)
                    v_list.append(recon_v)

            if block.active_k is not None:
                k_list.append(block.active_k)
                v_list.append(block.active_v)

        full_k = torch.cat(k_list, dim=2) if len(k_list) > 1 else k_list[0]
        full_v = torch.cat(v_list, dim=2) if len(v_list) > 1 else v_list[0]
        return full_k, full_v

    # ── KV write (from attention forward) ────────────────────────────────────

    def set_kv(self, session_id: str, layer_idx: int,
               k: torch.Tensor, v: torch.Tensor):
        if session_id not in self.session_blocks:
            self.init_session(session_id)

        blocks = self.session_blocks[session_id][layer_idx]

        # ---- Initial prefill phase ----------------------------------------
        if len(blocks) == 0:
            seq_len = k.shape[2]
            for start_idx in range(0, seq_len, self.block_size):
                end_idx = min(start_idx + self.block_size, seq_len)
                chunk_k = k[:, :, start_idx:end_idx]
                chunk_v = v[:, :, start_idx:end_idx]
                anchor_kv = torch.stack(
                    [chunk_k[:, :, 0], chunk_v[:, :, 0]], dim=1
                )
                block = KVBlock(
                    anchor_idx=start_idx,
                    anchor_kv=anchor_kv,
                    token_indices=list(range(start_idx, end_idx)),
                )
                if chunk_k.shape[2] > 1:
                    active_k = chunk_k[:, :, 1:]
                    active_v = chunk_v[:, :, 1:]
                    block.active_k = active_k
                    block.active_v = active_v
                blocks.append(block)
                self.pager.register_block(session_id, layer_idx, len(blocks) - 1, block)

            # Compress oldest prefill blocks (outside recency window)
            full_blocks = [
                b for b in blocks
                if b.active_k is not None and b.active_k.shape[2] >= self.block_size - 1
            ]
            while len(full_blocks) > self.dense_recency_blocks:
                oldest = full_blocks.pop(0)
                self._submit_compression(oldest, oldest.active_k, oldest.active_v)
            return

        # ---- Continuous decode phase: append 1 token ----------------------
        last_block = blocks[-1]
        curr_k = k[:, :, -1:]
        curr_v = v[:, :, -1:]

        if last_block.U is not None:
            # Last block already compressed — start a new block
            anchor_idx = last_block.anchor_idx + self.block_size
            anchor_kv  = torch.stack([curr_k[:, :, 0], curr_v[:, :, 0]], dim=1)
            new_block  = KVBlock(
                anchor_idx=anchor_idx, anchor_kv=anchor_kv,
                token_indices=[anchor_idx],
            )
            blocks.append(new_block)
            self.pager.register_block(session_id, layer_idx, len(blocks) - 1, new_block)
            return

        if last_block.active_k is None:
            last_block.active_k = curr_k
            last_block.active_v = curr_v
        else:
            last_block.active_k = torch.cat([last_block.active_k, curr_k], dim=2)
            last_block.active_v = torch.cat([last_block.active_v, curr_v], dim=2)

        # Invalidate recon cache for this block since active_k changed
        self.recon_cache.invalidate(last_block)

        # Compress oldest full dense blocks outside the recency window
        full_dense = [
            b for b in blocks
            if b.U is None and b.active_k is not None
            and b.active_k.shape[2] >= self.block_size - 1
        ]
        while len(full_dense) > self.dense_recency_blocks:
            oldest = full_dense.pop(0)
            self._submit_compression(oldest, oldest.active_k, oldest.active_v)

        # Trigger pager to check budget
        self.pager.maybe_evict()

    # ── Compression helpers ───────────────────────────────────────────────────

    def _submit_compression(self, block: KVBlock,
                            k: torch.Tensor, v: torch.Tensor):
        """Route to async or sync compressor."""
        if self._async:
            self._compressor.submit(block, k, v)
        else:
            self._compress_block_sync(block, k, v)

    def _compress_block_sync(self, block: KVBlock,
                             k: torch.Tensor, v: torch.Tensor):
        """Synchronous SVD compression (used by AsyncCompressor worker).
        
        Uses AdaptiveRankSelector (salvaged from RESEARCH_PROTOTYPES) when available
        to pick the minimal rank that preserves 95%+ KV delta variance.
        Falls back to self.rank when adaptive selector is unavailable.
        """
        anchor_flat = block.anchor_kv.view(-1).float()
        seq_len  = k.shape[2]
        heads    = k.shape[1]
        head_dim = k.shape[3]
        feat_dim = 2 * heads * head_dim

        stacked     = torch.stack([k[0].transpose(0, 1), v[0].transpose(0, 1)], dim=1)
        flat_tokens = stacked.reshape(seq_len, feat_dim).float()
        deltas      = flat_tokens - anchor_flat.unsqueeze(0)

        # ── Adaptive rank selection (P1 salvage) ──────────────────────────
        if self._adaptive_rank and self._rank_selector is not None:
            rank = self._rank_selector.select_rank(deltas)
        else:
            rank = self.rank

        lr_delta = compress_lowrank(deltas, rank)

        block.U          = lr_delta.U
        block.V          = lr_delta.V
        block.scale      = lr_delta.scale
        block.cosine_sim = lr_delta.cosine_sim
        block.norm_drift = lr_delta.norm_drift

        block.active_k = None
        block.active_v = None

        self.total_compressions += 1
        self.total_cosine_sim   += lr_delta.cosine_sim
        self.total_norm_drift   += lr_delta.norm_drift
        self.rank_histogram[rank] = self.rank_histogram.get(rank, 0) + 1

        fp16_bytes = seq_len * feat_dim * 2
        lr_bytes   = block.U.numel() * 2 + block.V.numel() * 2
        self.vram_saved_bytes += (fp16_bytes - lr_bytes)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def runtime_summary(self) -> dict:
        pager_s = self.pager.summary()
        recon_s = self.recon_cache.summary()
        comp_s  = self._compressor.summary()
        avg_cos   = (self.total_cosine_sim / max(1, self.total_compressions))
        avg_drift = (self.total_norm_drift  / max(1, self.total_compressions))
        return {
            "sessions":              len(self.session_blocks),
            "total_compressions":    self.total_compressions,
            "avg_cosine_sim":        round(avg_cos, 4),
            "avg_norm_drift":        round(avg_drift, 4),
            "vram_saved_mb":         round(self.vram_saved_bytes / 1e6, 2),
            "adaptive_rank":         self._adaptive_rank,
            "rank_histogram":        dict(sorted(self.rank_histogram.items())),
            "pager":                 pager_s,
            "recon_cache":           recon_s,
            "async_compressor":      comp_s,
        }
