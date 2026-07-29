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

import os
import threading
import time
import torch
from collections import OrderedDict


def dkv_deterministic() -> bool:
    """DKV_DETERMINISTIC=1 — no background thread may mutate runtime state.

    Every background mutator here fires on a WALL-CLOCK timer, so which blocks
    are resident (and therefore which the router can pick) when decode starts
    depends on thread scheduling. Two runs of the same build on the same prompt
    then take different routing decisions and emit different tokens, at
    temperature 0. That makes both the generated text and per-layer
    cosine-vs-dense useless as A/B metrics.

    MLX -- the reference implementation, which IS reproducible -- runs a single
    background thread and no timed eviction/prefetch at all. This flag brings the
    CUDA/Metal path to that same configuration for measurement. Leave it OFF in
    production, where the overlap is the point.
    """
    return os.environ.get("DKV_DETERMINISTIC", "0") == "1"
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
    prefetched: bool = False   # True if loaded via async prefetch


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
        self._running = True
        self._deterministic = dkv_deterministic()
        # Timed eviction is OPT-IN (DKV_PAGER_BG_EVICT=1), default OFF.
        #
        # It is REDUNDANT: `maybe_evict()` is already called synchronously after
        # every ingest (kv_runtime_manager, "Trigger pager to check budget"), and
        # this loop calls that same method -- so it adds no capability, only a
        # 2.0 s wall-clock trigger. That timer decided WHICH blocks were resident
        # when decode started, which made the runtime non-reproducible at
        # temperature 0 once the allocation grew large enough for eviction to
        # engage at all. Measured: two identical runs diverged, one into a
        # degenerate repetition loop; disabling this pinned them.
        #
        # MLX has no timed eviction of any kind, so synchronous-only is also the
        # parity behaviour.
        #
        # Trade-off: a session that stops ingesting no longer has memory
        # reclaimed while idle. Reclaim resumes on the next ingest, and the
        # budget check that matters happens at allocation time. Set
        # DKV_PAGER_BG_EVICT=1 to restore the timer.
        self._bg_thread = None
        if not self._deterministic and os.environ.get("DKV_PAGER_BG_EVICT", "0") == "1":
            self._bg_thread = threading.Thread(target=self._bg_eviction_loop, daemon=True)
            self._bg_thread.start()

        # Async prefetch thread
        self._prefetch_queue: List[Tuple] = []
        self._prefetch_lock = threading.Lock()
        self._prefetch_cv = threading.Condition(self._prefetch_lock)
        # Background prefetch is OPT-IN (DKV_PAGER_BG_PREFETCH=1), default OFF —
        # same reasoning as the eviction timer above. `_bg_prefetch_loop` calls
        # `_reload_block`, i.e. it changes a block's RESIDENCY on its own thread,
        # so which blocks are on the GPU when decode starts depends on thread
        # timing. A miss is not a correctness failure: `_reload_block` is also
        # reachable synchronously on access, so prefetch only ever hides latency.
        # MLX has no tier-prefetch thread at all.
        self._bg_prefetch_thread = None
        if not self._deterministic and os.environ.get("DKV_PAGER_BG_PREFETCH", "0") == "1":
            self._bg_prefetch_thread = threading.Thread(target=self._bg_prefetch_loop, daemon=True)
            self._bg_prefetch_thread.start()

        # Live stats
        self.stats = {
            "evictions":         0,
            "reloads":           0,
            "bytes_paged_out":   0,
            "bytes_paged_in":    0,
            "current_gpu_bytes": 0,
            "prefetch_issued":   0,
            "prefetch_hits":     0,
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
                entry.prefetched = False
                self._reload_block(key, entry)
            elif getattr(entry, "prefetched", False):
                self.stats["prefetch_hits"] = self.stats.get("prefetch_hits", 0) + 1
                entry.prefetched = False

    # ── Eviction ─────────────────────────────────────────────────────────────

    def maybe_evict(self) -> None:
        """Evict coldest blocks to CPU until under GPU budget."""
        with self._lock:
            while self.stats["current_gpu_bytes"] > self.gpu_budget_bytes:
                coldest_key = self._find_coldest()
                if coldest_key is None:
                    break
                self._evict_block(coldest_key, self._entries[coldest_key])

    def prefetch(self, session_id: str, layer_idx: int, block_idx: int) -> None:
        """Issue an asynchronous prefetch request for a block if it is on CPU."""
        import os
        if os.environ.get("DKV_PREDICTIVE_PAGING", "0") != "1":
            return

        key = (session_id, layer_idx, block_idx)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.residency == BlockResidency.GPU:
                return

        with self._prefetch_lock:
            if key not in self._prefetch_queue:
                self._prefetch_queue.append(key)
                self.stats["prefetch_issued"] = self.stats.get("prefetch_issued", 0) + 1
                self._prefetch_cv.notify()

    def _bg_prefetch_loop(self):
        """Background thread loop to process prefetch requests."""
        while self._running:
            with self._prefetch_lock:
                while not self._prefetch_queue and self._running:
                    self._prefetch_cv.wait(timeout=1.0)
                if not self._running:
                    break
                if not self._prefetch_queue:
                    continue
                key = self._prefetch_queue.pop(0)

            with self._lock:
                entry = self._entries.get(key)
                if entry is not None and entry.residency == BlockResidency.CPU:
                    self._reload_block(key, entry)
                    entry.prefetched = True

    def _find_coldest(self) -> Optional[Tuple]:
        """Return key of the GPU-resident entry with the oldest last_access, adjusted for reinforcement."""
        coldest_key  = None
        coldest_time = float("inf")
        manager = getattr(self, "manager", None)
        
        for key, entry in self._entries.items():
            if entry.residency == BlockResidency.GPU and entry.vram_bytes > 0:
                session_id, layer_idx, block_idx = key
                strength = 1.0
                if manager is not None:
                    srl_state = manager._session_srl.get(session_id)
                    if srl_state is not None and getattr(srl_state, "slot_activation_strength", None) is not None:
                        pool_idx = getattr(entry.block_ref, "pool_idx", None)
                        if pool_idx is not None:
                            strength = srl_state.slot_activation_strength.get(pool_idx, 1.0)
                
                # Boost time by strength * 300 seconds (5 minutes of virtual activity per unit strength)
                boost = (strength - 1.0) * 300.0
                composite_time = entry.last_access + boost
                
                if composite_time < coldest_time:
                    coldest_time = composite_time
                    coldest_key  = key
        return coldest_key

    def _evict_block(self, key: Tuple, entry: PageEntry) -> None:
        """Move block tensors to CPU pinned memory (called under lock)."""
        block = entry.block_ref
        moved = False

        if block.anchor_kv is not None and block.anchor_kv.is_cuda:
            block.anchor_kv = block.anchor_kv.to("cpu", non_blocking=True)
            moved = True
        u_val = getattr(block, "_U", None)
        if u_val is not None and u_val.is_cuda:
            block._U = u_val.to("cpu", non_blocking=True)
            u_scale = getattr(block, "_U_scale", None)
            if u_scale is not None:
                block._U_scale = u_scale.to("cpu", non_blocking=True)
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
        u_val = getattr(block, "_U", None)
        if u_val is not None and not u_val.is_cuda:
            block._U = u_val.to(dev, non_blocking=True)
            u_scale = getattr(block, "_U_scale", None)
            if u_scale is not None:
                block._U_scale = u_scale.to(dev, non_blocking=True)
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
        while self._running:
            time.sleep(2.0)
            if not self._running:
                break
            try:
                self.maybe_evict()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop background threads."""
        self._running = False
        with self._prefetch_lock:
            self._prefetch_cv.notify_all()

    def clear(self) -> None:
        """Clear all block entries and reset residency/stats."""
        with self._lock:
            self._entries.clear()
            self.stats = {
                "evictions":         0,
                "reloads":           0,
                "bytes_paged_out":   0,
                "bytes_paged_in":    0,
                "current_gpu_bytes": 0,
                "prefetch_issued":   0,
                "prefetch_hits":     0,
            }
        with self._prefetch_lock:
            self._prefetch_queue.clear()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _block_vram(block) -> int:
        total = 0
        for t in [block.anchor_kv, getattr(block, "_U", None), getattr(block, "_U_scale", None), block.V, block.active_k, block.active_v]:
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
                "prefetch_issued":   self.stats.get("prefetch_issued", 0),
                "prefetch_hits":     self.stats.get("prefetch_hits", 0),
                "tracked_blocks":    len(self._entries),
            }
