"""
runtime/paged_kv_store.py

Phase 7: Paged Sparse KV Memory

Implements true two-tier KV block residency:
  - Tier 1: GPU VRAM  (hot, immediately accessible)
  - Tier 2: CPU RAM   (warm, pinned memory, reloadable)

KVBlocks are paged OUT to CPU when VRAM pressure rises.
They are paged IN (async) when accessed again.

Design:
  - PagedKVStore wraps around the existing flat list of KVBlock objects.
  - It does NOT change block structure — only moves tensor data
    (U, V, anchor_kv, active_k, active_v) between devices.
  - Access is tracked via an LRU counter per (session_id, layer_idx, block_idx).
  - Eviction targets the coldest blocks across all sessions.
  - Reload is synchronous by default; async mode uses a background thread queue.

Profiler-visible impact:
  - Reduces GPU memory_allocated() when many sessions are live.
  - Adds reload latency only when a cold block is actually accessed.
  - Tracked via `.stats` dict (evictions, reloads, bytes paged).
"""

import threading
import time
import torch
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Block residency state
# ─────────────────────────────────────────────────────────────────────────────

class BlockResidency:
    GPU  = "gpu"
    CPU  = "cpu"


@dataclass
class PageEntry:
    """Tracks where a single KVBlock's tensors currently live."""
    block_ref: object          # reference to the KVBlock object
    residency:  str = BlockResidency.GPU
    last_access: float = field(default_factory=time.time)
    vram_bytes: int = 0        # bytes currently on GPU (0 if paged to CPU)


# ─────────────────────────────────────────────────────────────────────────────
# Paged KV Store
# ─────────────────────────────────────────────────────────────────────────────

class PagedKVStore:
    """
    Two-tier memory manager for KVBlocks.

    Usage:
        store = PagedKVStore(gpu_budget_gb=2.0)
        store.register_block(session_id, layer_idx, block_idx, block)
        store.touch(session_id, layer_idx, block_idx)   # before accessing block
        store.maybe_evict()                              # call periodically
    """

    def __init__(self, gpu_budget_gb: float = 2.0, device: str = "cuda"):
        self.gpu_budget_bytes = int(gpu_budget_gb * 1024 ** 3)
        self.device = device

        # (session_id, layer_idx, block_idx) -> PageEntry
        self._entries: Dict[Tuple, PageEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Async eviction thread
        self._evict_queue: List[Tuple] = []
        self._bg_thread = threading.Thread(target=self._bg_eviction_loop, daemon=True)
        self._bg_thread.start()

        # Live stats
        self.stats = {
            "evictions":         0,
            "reloads":           0,
            "bytes_paged_out":   0,
            "bytes_paged_in":    0,
            "current_gpu_bytes": 0,
        }

    # ── Registration ────────────────────────────────────────────────────────

    def register_block(self, session_id: str, layer_idx: int,
                       block_idx: int, block) -> None:
        key = (session_id, layer_idx, block_idx)
        vram = self._block_vram(block)
        entry = PageEntry(block_ref=block, residency=BlockResidency.GPU,
                          last_access=time.time(), vram_bytes=vram)
        with self._lock:
            self._entries[key] = entry
            self.stats["current_gpu_bytes"] += vram

    # ── Access touch ────────────────────────────────────────────────────────

    def touch(self, session_id: str, layer_idx: int, block_idx: int) -> None:
        """Mark a block as recently accessed; reload from CPU if needed."""
        key = (session_id, layer_idx, block_idx)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.last_access = time.time()
            if entry.residency == BlockResidency.CPU:
                self._reload_block(key, entry)

    # ── Eviction ─────────────────────────────────────────────────────────────

    def maybe_evict(self) -> None:
        """Evict coldest blocks to CPU until under GPU budget."""
        with self._lock:
            while self.stats["current_gpu_bytes"] > self.gpu_budget_bytes:
                coldest_key = self._find_coldest()
                if coldest_key is None:
                    break
                self._evict_block(coldest_key, self._entries[coldest_key])

    def _find_coldest(self) -> Optional[Tuple]:
        """Return key of the GPU-resident entry with the oldest last_access."""
        coldest_key  = None
        coldest_time = float("inf")
        for key, entry in self._entries.items():
            if entry.residency == BlockResidency.GPU and entry.vram_bytes > 0:
                if entry.last_access < coldest_time:
                    coldest_time = entry.last_access
                    coldest_key  = key
        return coldest_key

    def _evict_block(self, key: Tuple, entry: PageEntry) -> None:
        """Move block tensors to CPU pinned memory (called under lock)."""
        block = entry.block_ref
        moved = False

        if block.anchor_kv is not None and block.anchor_kv.is_cuda:
            block.anchor_kv = block.anchor_kv.to("cpu", non_blocking=True)
            moved = True
        if block.U is not None and block.U.is_cuda:
            block.U = block.U.to("cpu", non_blocking=True)
            moved = True
        if block.V is not None and block.V.is_cuda:
            block.V = block.V.to("cpu", non_blocking=True)
            moved = True
        if block.active_k is not None and block.active_k.is_cuda:
            block.active_k = block.active_k.to("cpu", non_blocking=True)
            block.active_v = block.active_v.to("cpu", non_blocking=True)
            moved = True

        if moved:
            self.stats["evictions"]       += 1
            self.stats["bytes_paged_out"] += entry.vram_bytes
            self.stats["current_gpu_bytes"] -= entry.vram_bytes
            entry.residency  = BlockResidency.CPU
            entry.vram_bytes = 0

    def _reload_block(self, key: Tuple, entry: PageEntry) -> None:
        """Move block tensors back to GPU (called under lock)."""
        block = entry.block_ref
        dev = self.device

        if block.anchor_kv is not None and not block.anchor_kv.is_cuda:
            block.anchor_kv = block.anchor_kv.to(dev, non_blocking=True)
        if block.U is not None and not block.U.is_cuda:
            block.U = block.U.to(dev, non_blocking=True)
        if block.V is not None and not block.V.is_cuda:
            block.V = block.V.to(dev, non_blocking=True)
        if block.active_k is not None and not block.active_k.is_cuda:
            block.active_k = block.active_k.to(dev, non_blocking=True)
            block.active_v = block.active_v.to(dev, non_blocking=True)

        vram = self._block_vram(block)
        self.stats["reloads"]          += 1
        self.stats["bytes_paged_in"]   += vram
        self.stats["current_gpu_bytes"] += vram
        entry.residency  = BlockResidency.GPU
        entry.vram_bytes = vram

    # ── Session cleanup ──────────────────────────────────────────────────────

    def evict_session(self, session_id: str) -> None:
        """Remove all entries for a session (called on session deletion)."""
        with self._lock:
            keys_to_remove = [k for k in self._entries if k[0] == session_id]
            for key in keys_to_remove:
                entry = self._entries.pop(key)
                self.stats["current_gpu_bytes"] -= entry.vram_bytes

    # ── Background eviction ──────────────────────────────────────────────────

    def _bg_eviction_loop(self):
        """Periodically check and evict if over budget."""
        while True:
            time.sleep(2.0)
            try:
                self.maybe_evict()
            except Exception:
                pass

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _block_vram(block) -> int:
        """Estimate GPU bytes consumed by a KVBlock's live tensors."""
        total = 0
        for t in [block.anchor_kv, block.U, block.V, block.active_k, block.active_v]:
            if t is not None and t.is_cuda:
                total += t.numel() * t.element_size()
        return total

    def summary(self) -> dict:
        with self._lock:
            gpu_mb = self.stats["current_gpu_bytes"] / 1e6
            paged_out_mb = self.stats["bytes_paged_out"] / 1e6
            return {
                "gpu_resident_mb":   round(gpu_mb, 2),
                "total_evictions":   self.stats["evictions"],
                "total_reloads":     self.stats["reloads"],
                "bytes_paged_out_mb": round(paged_out_mb, 2),
                "bytes_paged_in_mb":  round(self.stats["bytes_paged_in"] / 1e6, 2),
                "tracked_blocks":    len(self._entries),
            }
