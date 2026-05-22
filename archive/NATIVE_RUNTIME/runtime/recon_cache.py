"""
runtime/recon_cache.py

Phase 7 Step 3: Reconstruction Hot-Block Cache

KVBlocks are reconstructed frequently during fused sparse attention.
If a block appears in many successive decode steps (e.g. the first few history
blocks during a long conversation), we pay the GEMM cost repeatedly.

This LRU cache stores recently reconstructed dense (K, V) tensors keyed by
the block's identity (id(block), which is stable for the block's lifetime).
Cache hit = zero reconstruction cost.
Cache miss = reconstruct once, store result.

Design constraints:
  - Fixed max number of cached entries (default 32 blocks).
  - Each cache entry is a pair of fp16 tensors [1, heads, seq, head_dim].
  - Eviction is pure LRU (OrderedDict move-to-end trick).
  - Thread-safe via a single lock.
  - GPU-resident only (don't cache on CPU — the benefit would vanish).

Profiler-visible impact:
  - Reduces repeat `U @ V` GEMMs per decode step.
  - Most beneficial for long conversations where history blocks are stable.
"""

import threading
from collections import OrderedDict
from typing import Optional, Tuple
import torch


class ReconstructionCache:
    """
    LRU cache mapping block identity -> (reconstructed_k, reconstructed_v).

    Usage:
        cache = ReconstructionCache(max_entries=32)

        k, v = cache.get(block)          # None, None on miss
        if k is None:
            k, v = expensive_reconstruct(block)
            cache.put(block, k, v)
    """

    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._store: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._lock  = threading.Lock()

        # Stats
        self.hits   = 0
        self.misses = 0

    # ── Public interface ─────────────────────────────────────────────────────

    def get(self, block) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        key = id(block)
        with self._lock:
            if key in self._store:
                # Move to end (most-recently-used)
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None, None

    def put(self, block, k: torch.Tensor, v: torch.Tensor) -> None:
        key = id(block)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (k, v)
            # Evict oldest if over budget
            if len(self._store) > self.max_entries:
                evicted_key, (ek, ev) = self._store.popitem(last=False)
                # Explicitly delete tensors to free GPU memory immediately
                del ek, ev

    def invalidate(self, block) -> None:
        """Call when a block's U/V changes (e.g. re-compressed)."""
        key = id(block)
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits   = 0
            self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def summary(self) -> dict:
        with self._lock:
            return {
                "cached_blocks": len(self._store),
                "hits":          self.hits,
                "misses":        self.misses,
                "hit_rate":      round(self.hit_rate, 3),
            }
