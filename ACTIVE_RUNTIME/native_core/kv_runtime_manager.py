"""
native_core/kv_runtime_manager.py

Minimal Runtime — KV Session Manager.

Manages KV cache residency and on-demand reconstruction for Differential KV.

  - PagedKVStore: GPU -> CPU RAM spillover under memory pressure.
  - ReconstructionCache: LRU cache of recently reconstructed dense blocks.
  - AsyncCompressor: SVD compression moved off the decode hot path.
  - StreamingSparseIngestManager: sparse-first prefill (no dense-first allocation).
  - NativeBlockPool: contiguous GPU memory pool for Triton kernel dispatch.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import sys, os
import threading

from native_core.sparse_decode.triton_diffkv import TritonDiffKV
from native_core.compression.lowrank import compress_lowrank, LowRankDelta
from native_core.paging.paged_kv_store import PagedKVStore
from native_core.recon_cache import ReconstructionCache
from native_core.compression.async_compressor import AsyncCompressor


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
    pool_idx: Optional[int] = None
    dirty:    bool = True
    _lock:    threading.Lock = field(default_factory=threading.Lock, repr=False)


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
        streaming_ingest:    bool  = True,
        micro_block_size:    int   = 16,
        rank:                int   = 8,
        kv_heads:            int   = None,
    ):
        self.num_layers  = num_layers
        self.heads       = heads
        self.kv_heads    = kv_heads if kv_heads is not None else heads
        self.head_dim    = head_dim
        self.device      = device
        self.feat_dim    = 2 * self.kv_heads * head_dim

        # session_id -> layer_idx -> List[KVBlock]
        self.session_blocks: Dict[str, Dict[int, List[KVBlock]]] = {}

        self.block_size           = 64
        self.rank                 = rank  # fixed, set at construction
        self.dense_recency_blocks = 1
        self.streaming_ingest     = streaming_ingest
        self.micro_block_size     = micro_block_size

        # ── Phase 28 Native Block Pool ──────────────────────────────────────────
        from runtime.native_block_pool import NativeBlockPool
        
        # Dynamically calculate max_blocks based on gpu_budget_gb
        pool_rank = self.rank
        pool_block_size = self.micro_block_size if self.streaming_ingest else self.block_size
        
        bytes_per_block = (
            (pool_block_size * pool_rank * 2) +                                # U
            (pool_rank * self.kv_heads * self.head_dim * 2) * 2 +              # V_K, V_V
            (self.kv_heads * self.head_dim * 2) * 2 +                          # anchors_K, anchors_V
            6                                                                  # scales (2) + seq_lens (4)
        )
        
        # Dedicate 75% of the gpu_budget to the contiguous NativeBlockPool to ensure plenty of blocks
        pool_budget_bytes = int(gpu_budget_gb * (1024 ** 3) * 0.75)
        dynamic_max_blocks = max(20000, pool_budget_bytes // bytes_per_block)
        
        self.native_pool = NativeBlockPool(
            max_blocks=dynamic_max_blocks,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            rank=pool_rank,
            max_seq_len=pool_block_size,
            device=self.device,
            dtype=torch.float16
        )

        # ── Phase 7 subsystems ────────────────────────────────────────────
        self.pager       = PagedKVStore(gpu_budget_gb=gpu_budget_gb, device=device)
        self.recon_cache = ReconstructionCache(max_entries=recon_cache_size)

        # Instantiate ReconstructionPool for high-throughput decode path
        from native_core.recon_cache import ReconstructionPool
        self.recon_pool = ReconstructionPool(
            max_cached_blocks=2048,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            micro_block_size=self.micro_block_size,
            device=self.device
        )
        self.decode_workspace = {}

        self._async      = async_compression
        self._compressor = AsyncCompressor(compress_fn=self._compress_block_sync)
        if self._async:
            self._compressor.start()

        # ── Phase 24.5: Streaming Sparse Ingest Manager ───────────────────
        if self.streaming_ingest:
            from native_core.streaming_sparse_ingest import StreamingSparseIngestManager
            self._streaming_mgr = StreamingSparseIngestManager(
                compressor=self._compressor,
                compress_fn=self._compress_block_sync,
                micro_block_size=self.micro_block_size,
                dense_anchor_only=True,
                native_pool=self.native_pool,
            )
        else:
            self._streaming_mgr = None

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
        if self._streaming_mgr is not None and session_id not in self._streaming_mgr.session_blocks:
            self._streaming_mgr.init_session(session_id, self.num_layers)

    def clear_session(self, session_id: str):
        # Free blocks from NativeBlockPool before deleting references
        if hasattr(self, 'native_pool') and self.native_pool is not None:
            if session_id in self.session_blocks:
                for layer_idx, blocks in self.session_blocks[session_id].items():
                    for block in blocks:
                        if getattr(block, 'pool_idx', None) is not None:
                            self.native_pool.free_block(block.pool_idx)
                            block.pool_idx = None
            if self._streaming_mgr is not None and session_id in self._streaming_mgr.session_blocks:
                for layer_idx, blocks in self._streaming_mgr.session_blocks[session_id].items():
                    for block in blocks:
                        if getattr(block, 'pool_idx', None) is not None:
                            self.native_pool.free_block(block.pool_idx)
                            block.pool_idx = None

        # Invalidate slots in ReconstructionPool for blocks of this session
        if hasattr(self, 'recon_pool') and self.recon_pool is not None:
            pool_idxs_to_invalidate = []
            if session_id in self.session_blocks:
                for layer_idx, blocks in self.session_blocks[session_id].items():
                    for block in blocks:
                        if getattr(block, 'pool_idx', None) is not None:
                            pool_idxs_to_invalidate.append(block.pool_idx)
            if self._streaming_mgr is not None and session_id in self._streaming_mgr.session_blocks:
                for layer_idx, blocks in self._streaming_mgr.session_blocks[session_id].items():
                    for block in blocks:
                        if getattr(block, 'pool_idx', None) is not None:
                            pool_idxs_to_invalidate.append(block.pool_idx)
            if pool_idxs_to_invalidate:
                self.recon_pool.invalidate_pool_indices(pool_idxs_to_invalidate)

        # Delete workspaces for this session
        if hasattr(self, 'decode_workspace'):
            keys_to_del = [k for k in self.decode_workspace.keys() if k[0] == session_id]
            for k in keys_to_del:
                del self.decode_workspace[k]

        if session_id in self.session_blocks:
            del self.session_blocks[session_id]
        if self._streaming_mgr is not None:
            self._streaming_mgr.clear_session(session_id)
        self.pager.evict_session(session_id)

    def clear(self):
        # 1. Cleanly clear all registered sessions to release their pool blocks
        sessions = set(self.session_blocks.keys())
        if self._streaming_mgr is not None and hasattr(self._streaming_mgr, 'session_blocks'):
            sessions.update(self._streaming_mgr.session_blocks.keys())
            
        for session_id in sessions:
            self.clear_session(session_id)

        # 2. Reset subsystems and clear references
        self.session_blocks.clear()
        self.recon_cache.clear()
        if hasattr(self, 'recon_pool') and self.recon_pool is not None:
            self.recon_pool.clear()
        if hasattr(self, 'decode_workspace'):
            self.decode_workspace.clear()
        
        if hasattr(self, 'native_pool') and self.native_pool is not None:
            self.native_pool.reset()
            
        if hasattr(self, 'pager') and self.pager is not None:
            self.pager.clear()

    # ── Phase 24.5: Streaming Sparse Ingest ───────────────────────────────────

    def ingest_streaming(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,  # [1, heads, T, head_dim] — full prefill or single decode token
        v: torch.Tensor,
    ) -> None:
        """
        Phase 24.5 entry point: streaming sparse ingest.

        Routes tokens through StreamingSparseIngestManager instead of set_kv().
        - Compresses during ingest, not after.
        - Dense footprint bounded to 1 micro-block (default 16 tokens) at any time.
        - For decode (T=1), delegates to append_decode_token().
        """
        if self._streaming_mgr is None:
            # Fallback to legacy dense path if streaming disabled
            self.set_kv(session_id, layer_idx, k, v)
            return

        if session_id not in self._streaming_mgr.session_blocks:
            self._streaming_mgr.init_session(session_id, self.num_layers)

        if k.shape[2] == 1:
            self._streaming_mgr.append_decode_token(session_id, layer_idx, k, v)
        else:
            self._streaming_mgr.ingest_chunk(session_id, layer_idx, k, v)

    def get_streaming_blocks(self, session_id: str, layer_idx: int) -> list:
        """
        Return blocks from streaming manager (Phase 24.5 path).
        Falls back to legacy session_blocks if streaming not active.
        """
        if self._streaming_mgr is not None and session_id in self._streaming_mgr.session_blocks:
            return self._streaming_mgr.get_blocks(session_id, layer_idx)
        return self.get_raw_blocks(session_id, layer_idx)

    def get_streaming_summary(self, session_id: str = None) -> dict:
        """Return streaming ingest telemetry for Phase 24.5 reporting."""
        if self._streaming_mgr is None:
            return {"streaming_ingest": False}
        s = self._streaming_mgr.summary(session_id)
        s["streaming_ingest"] = True
        s["micro_block_size"] = self.micro_block_size
        return s

    # ── High-throughput decode KV assembly ────────────────────────────────────

    def assemble_decode_kv(
        self,
        session_id: str,
        layer_idx: int,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Vectorized KV assembly for the decode hot path.

        Design goals vs. the old block-by-block loop in diffkv_attention.py:
          1. Single workspace tensor reused across decode steps (no per-step malloc).
          2. Anchor tokens copied via vectorized index_put_ (not a Python for-loop).
          3. Compressed blocks satisfied from ReconstructionPool where possible
             (zero GEMM cost on hit); batch GEMM for all misses in one bmm call.
          4. Dense active_k/v blocks copied in one per-block copy (unchanged).

        Returns
        -------
        k_tensor : [1, kv_heads, total_seq_len, head_dim]  or None if no blocks
        v_tensor : [1, kv_heads, total_seq_len, head_dim]  or None
        """
        blocks = self.get_streaming_blocks(session_id, layer_idx)
        if not blocks:
            return None, None

        # Helper class for thread-safe block snapshots
        class BlockSnapshot:
            def __init__(self, block):
                self.b = block
                lock = getattr(block, "_lock", None)
                if lock is not None:
                    with lock:
                        self.anchor_kv = block.anchor_kv
                        self.U = block.U
                        self.V = block.V
                        self.scale = getattr(block, "scale", 1.0)
                        self.active_k = block.active_k
                        self.active_v = block.active_v
                        self.pool_idx = getattr(block, "pool_idx", None)
                        self.dirty = getattr(block, "dirty", True)
                else:
                    self.anchor_kv = block.anchor_kv
                    self.U = block.U
                    self.V = block.V
                    self.scale = getattr(block, "scale", 1.0)
                    self.active_k = block.active_k
                    self.active_v = block.active_v
                    self.pool_idx = getattr(block, "pool_idx", None)
                    self.dirty = getattr(block, "dirty", True)

        # ── 1. Snapshot blocks under lock to prevent background compression races ──
        snapshots = []
        for b in blocks:
            snapshots.append(BlockSnapshot(b))

        # ── 2. Compute total sequence length from snapshots ─────────────────────
        total_seq_len = 0
        for b_snap in snapshots:
            total_seq_len += 1  # anchor
            if b_snap.U is not None:
                total_seq_len += b_snap.U.shape[0]
            elif b_snap.active_k is not None:
                total_seq_len += b_snap.active_k.shape[2]

        # ── 3. Allocate or resize persistent workspace ────────────────────────
        ws_key = (session_id, layer_idx)
        ws = self.decode_workspace.get(ws_key)
        ws_new = False
        if ws is None or ws[0].shape[2] < total_seq_len:
            # Allocate with head room (+25%) to reduce future reallocations
            alloc_len = max(total_seq_len + total_seq_len // 4, 64)
            k_ws = torch.zeros(
                (1, self.kv_heads, alloc_len, self.head_dim),
                dtype=dtype, device=self.device
            )
            v_ws = torch.zeros(
                (1, self.kv_heads, alloc_len, self.head_dim),
                dtype=dtype, device=self.device
            )
            self.decode_workspace[ws_key] = (k_ws, v_ws)
            ws = (k_ws, v_ws)
            ws_new = True

        k_ws, v_ws = ws

        # If workspace was newly allocated or resized, all blocks must be rewritten
        if ws_new:
            for b_snap in snapshots:
                b_snap.dirty = True

        # ── 4. Separate dirty blocks into categories ───────────────────────────────
        anchor_positions = []     # position index in k_ws
        anchor_k_list   = []     # [kv_heads, head_dim] tensors
        anchor_v_list   = []

        compressed_hits   = []   # (b_snap, pool_slot, start_pos)
        compressed_misses = []   # (b_snap, start_pos)

        dense_copies = []        # (b_snap, start_pos, length)

        cursor = 0
        miss_pool_idxs = []
        dirty_blocks = []

        for b_snap in snapshots:
            b_anchor_pos = cursor
            cursor += 1
            b_content_pos = cursor

            if b_snap.U is not None and b_snap.V is not None:
                block_len = b_snap.U.shape[0]
            elif b_snap.active_k is not None:
                block_len = b_snap.active_k.shape[2]
            else:
                block_len = 0

            cursor += block_len

            if b_snap.dirty:
                dirty_blocks.append(b_snap)

                anchor_positions.append(b_anchor_pos)
                anchor_k_list.append(b_snap.anchor_kv[0, 0])
                anchor_v_list.append(b_snap.anchor_kv[0, 1])

                if b_snap.U is not None and b_snap.V is not None:
                    pool_idx = b_snap.pool_idx
                    if pool_idx is not None:
                        if pool_idx < self.recon_pool.pool_to_slot.shape[0]:
                            slot = int(self.recon_pool.pool_to_slot[pool_idx])
                        else:
                            slot = -1
                        if slot >= 0:
                            compressed_hits.append((b_snap, slot, b_content_pos))
                        else:
                            compressed_misses.append((b_snap, b_content_pos))
                            miss_pool_idxs.append(pool_idx)
                    else:
                        compressed_misses.append((b_snap, b_content_pos))
                        miss_pool_idxs.append(-1)
                elif b_snap.active_k is not None:
                    dense_copies.append((b_snap, b_content_pos, block_len))

        # ── 5. Vectorized anchor copy for dirty blocks ─────────────────────────
        if anchor_positions:
            pos_t = torch.tensor(anchor_positions, device=self.device, dtype=torch.long)
            ak_t  = torch.stack(anchor_k_list, dim=0)   # [N, kv_heads, head_dim]
            av_t  = torch.stack(anchor_v_list, dim=0)
            k_ws[0, :, pos_t, :] = ak_t.transpose(0, 1)   # [kv_heads, N, head_dim]
            v_ws[0, :, pos_t, :] = av_t.transpose(0, 1)

        # ── 6. Compressed hits — direct slice from ReconstructionPool ─────────
        for b_snap, slot, start_pos in compressed_hits:
            block_len = b_snap.U.shape[0]
            k_ws[0, :, start_pos:start_pos + block_len, :] = \
                self.recon_pool.K[slot, :, :block_len, :]
            v_ws[0, :, start_pos:start_pos + block_len, :] = \
                self.recon_pool.V[slot, :, :block_len, :]

        # ── 7. Compressed misses — batched GEMM then write into pool + ws ─────
        if compressed_misses:
            B = len(compressed_misses)
            miss_blocks = [t[0] for t in compressed_misses]
            miss_starts = [t[1] for t in compressed_misses]

            stacked_U      = torch.stack([b_snap.U.float()           for b_snap in miss_blocks], dim=0)
            stacked_V      = torch.stack([b_snap.V.float()           for b_snap in miss_blocks], dim=0)
            stacked_scale  = torch.tensor(
                [b_snap.scale for b_snap in miss_blocks], device=self.device, dtype=torch.float32
            ).view(B, 1, 1)
            stacked_anchor = torch.stack(
                [b_snap.anchor_kv.reshape(-1).float() for b_snap in miss_blocks], dim=0
            ).unsqueeze(1)

            recon_flat = torch.bmm(stacked_U, stacked_V) * stacked_scale + stacked_anchor

            if not torch.isfinite(recon_flat).all():
                recon_flat = torch.nan_to_num(recon_flat, nan=0.0, posinf=0.0, neginf=0.0)

            recon = recon_flat.view(B, -1, 2, self.kv_heads, self.head_dim).to(dtype)
            recon_k = recon[:, :, 0].permute(0, 2, 1, 3)
            recon_v = recon[:, :, 1].permute(0, 2, 1, 3)

            valid_pool_idxs = [p for p in miss_pool_idxs if p >= 0]
            if valid_pool_idxs:
                alloc_slots = self.recon_pool.allocate_slots(valid_pool_idxs)
            else:
                alloc_slots = []

            slot_iter = iter(alloc_slots)
            for i, (b_snap, start_pos) in enumerate(zip(miss_blocks, miss_starts)):
                block_len = b_snap.U.shape[0]
                k_ws[0, :, start_pos:start_pos + block_len, :] = recon_k[i, :, :block_len, :]
                v_ws[0, :, start_pos:start_pos + block_len, :] = recon_v[i, :, :block_len, :]

                pool_idx = miss_pool_idxs[i]
                if pool_idx >= 0:
                    slot = next(slot_iter, -1)
                    if slot >= 0:
                        self.recon_pool.K[slot, :, :block_len, :] = recon_k[i, :, :block_len, :].to(torch.float16)
                        self.recon_pool.V[slot, :, :block_len, :] = recon_v[i, :, :block_len, :].to(torch.float16)

                self.recon_cache.put(
                    b_snap.b,
                    recon_k[i:i+1, :, :block_len, :].contiguous().clone(),
                    recon_v[i:i+1, :, :block_len, :].contiguous().clone(),
                )

        # ── 8. Dense active blocks — slice copy ───────────────────────────────
        for b_snap, start_pos, block_len in dense_copies:
            k_ws[0, :, start_pos:start_pos + block_len, :] = b_snap.active_k[0]
            v_ws[0, :, start_pos:start_pos + block_len, :] = b_snap.active_v[0]

        # ── 9. Reset dirty flags on processed blocks under lock, if they haven't changed since snapshot ──
        for b_snap in dirty_blocks:
            lock = getattr(b_snap.b, "_lock", None)
            if lock is not None:
                with lock:
                    if (b_snap.b.U is b_snap.U) and (b_snap.b.active_k is b_snap.active_k):
                        b_snap.b.dirty = False
            else:
                if (b_snap.b.U is b_snap.U) and (b_snap.b.active_k is b_snap.active_k):
                    b_snap.b.dirty = False

        # ── 10. Return a view of the valid portion (no copy) ───────────────────
        return k_ws[:, :, :total_seq_len, :], v_ws[:, :, :total_seq_len, :]

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
        if self.get_seq_len(session_id, layer_idx) > 16384:
            raise RuntimeError(f"[DIFFKV] FATAL MEMORY GUARD: get_kv() called for sequence > 16k tokens (session {session_id}). Full-sequence dense reconstruction bypassed to prevent 20GB OOM spike.")

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
                    anchor_flat = block.anchor_kv.reshape(-1).to(torch.float16)
                    recon = TritonDiffKV.reconstruct_lowrank(
                        block.U, block.V, anchor_flat, scale=block.scale
                    )
                    hds  = block.anchor_kv.shape[2]
                    hdim = block.anchor_kv.shape[3]
                    recon = recon.view(1, -1, 2, hds, hdim)
                    # Use .contiguous().clone() so the cached tensors are fully independent:
                    # .transpose() creates non-contiguous views, and without cloning they'd
                    # alias reconstruct_lowrank's output which may be reused by a future call.
                    recon_k = recon[:, :, 0].transpose(1, 2).contiguous().clone()
                    recon_v = recon[:, :, 1].transpose(1, 2).contiguous().clone()
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
        Fixed rank -- simple, stable, predictable.
        """
        input_device = k.device
        if input_device.type == "cpu":
            anchor_kv_local = getattr(block, "anchor_kv_cpu", None)
            if anchor_kv_local is None:
                anchor_kv_local = block.anchor_kv.cpu()
        else:
            anchor_kv_local = block.anchor_kv
        anchor_flat = anchor_kv_local.reshape(-1).float().to(input_device)
        seq_len  = k.shape[2]
        heads    = k.shape[1]
        head_dim = k.shape[3]
        feat_dim = 2 * heads * head_dim

        stacked     = torch.stack([k[0].transpose(0, 1), v[0].transpose(0, 1)], dim=1)
        flat_tokens = stacked.reshape(seq_len, feat_dim).float()
        deltas      = flat_tokens - anchor_flat.unsqueeze(0)

        rank = self.rank
        lr_delta = compress_lowrank(deltas, rank)

        gpu_device = block.anchor_kv.device
        u_gpu = lr_delta.U.to(gpu_device)
        v_gpu = lr_delta.V.to(gpu_device)

        lock = getattr(block, "_lock", None)
        if lock is None:
            lock = threading.Lock()
            try:
                block._lock = lock
            except AttributeError:
                pass

        if lock is not None:
            with lock:
                block.U          = u_gpu
                block.V          = v_gpu
                block.scale      = lr_delta.scale
                block.cosine_sim = lr_delta.cosine_sim
                block.norm_drift = lr_delta.norm_drift

                block.active_k = None
                block.active_v = None
                block.dirty    = True

                if hasattr(block, 'state'):
                    block.state = "COMPRESSED"
        else:
            block.U          = u_gpu
            block.V          = v_gpu
            block.scale      = lr_delta.scale
            block.cosine_sim = lr_delta.cosine_sim
            block.norm_drift = lr_delta.norm_drift

            block.active_k = None
            block.active_v = None
            block.dirty    = True

            if hasattr(block, 'state'):
                block.state = "COMPRESSED"

        # Phase 28 Native Block Pool Integration
        if hasattr(self, 'native_pool') and self.native_pool is not None:
            try:
                if getattr(block, 'pool_idx', None) is None:
                    block.pool_idx = self.native_pool.allocate_block()
                self.native_pool.write_block(
                    pool_idx=block.pool_idx,
                    U=block.U,
                    V=block.V,
                    anchor_K=block.anchor_kv[0, 0],
                    anchor_V=block.anchor_kv[0, 1],
                    scale=block.scale,
                    seq_len=block.U.shape[0]
                )
            except Exception as e:
                # Log warning but do not crash generation, as we can still decode using the standard PyTorch path!
                print(f"[DiffKV] WARNING: Failed to write block to NativeBlockPool: {e}")

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
            "fixed_rank":            self.rank,
            "rank_histogram":        dict(sorted(self.rank_histogram.items())),
            "pager":                 pager_s,
            "recon_cache":           recon_s,
            "async_compressor":      comp_s,
        }
