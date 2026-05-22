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
import numpy as np


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
        cache_id = getattr(block, "_cache_id", None)
        if cache_id is None:
            # If the block has no cache_id, it is brand new and cannot be in the cache.
            self.misses += 1
            return None, None

        with self._lock:
            if cache_id in self._store:
                # Move to end (most-recently-used)
                self._store.move_to_end(cache_id)
                self.hits += 1
                return self._store[cache_id]
            self.misses += 1
            return None, None

    def put(self, block, k: torch.Tensor, v: torch.Tensor) -> None:
        cache_id = getattr(block, "_cache_id", None)
        if cache_id is None:
            import uuid
            cache_id = str(uuid.uuid4())
            try:
                block._cache_id = cache_id
            except AttributeError:
                # Fallback to python object id in case the block object is frozen or read-only
                cache_id = id(block)

        with self._lock:
            if cache_id in self._store:
                self._store.move_to_end(cache_id)
            self._store[cache_id] = (k, v)
            # Evict oldest if over budget
            if len(self._store) > self.max_entries:
                evicted_key, (ek, ev) = self._store.popitem(last=False)
                # Explicitly delete tensors to free GPU memory immediately
                del ek, ev

    def invalidate(self, block) -> None:
        """Call when a block's U/V changes (e.g. re-compressed)."""
        cache_id = getattr(block, "_cache_id", None)
        if cache_id is not None:
            with self._lock:
                self._store.pop(cache_id, None)

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


class ReconstructionPool:
    """
    GPU-resident cache pool for reconstructed low-rank blocks.
    Maps pool_idx -> cache_slot_idx using 100% GPU-resident PyTorch tensors.
    """
    def __init__(self, max_cached_blocks: int, num_kv_heads: int, head_dim: int, micro_block_size: int, device: str):
        self.max_cached_blocks = max_cached_blocks
        self.device = device
        self.micro_block_size = micro_block_size
        
        # Preallocated GPU buffers
        self.K = torch.zeros((max_cached_blocks, num_kv_heads, micro_block_size, head_dim), dtype=torch.float16, device=device)
        self.V = torch.zeros((max_cached_blocks, num_kv_heads, micro_block_size, head_dim), dtype=torch.float16, device=device)
        
        # Map: pool_idx -> cache_slot_idx (GPU tensor)
        self.pool_to_slot = torch.full((200000,), -1, dtype=torch.int32, device=device)
        # Map: pool_idx -> cache_slot_idx (CPU shadow NumPy array to avoid PCIe sync)
        self.pool_to_slot_cpu = np.full((200000,), -1, dtype=np.int32)
        # Map: cache_slot_idx -> pool_idx (GPU tensor)
        self.slot_to_pool = torch.full((max_cached_blocks,), -1, dtype=torch.int32, device=device)
        
        # LRU state: tracking the "timestamp" of when each slot was last used
        self.lru_scores = torch.zeros((max_cached_blocks,), dtype=torch.float32, device=device)
        self.step_counter = 0
        self._lock = threading.Lock()

    def allocate_slots(self, pool_indices: list) -> list:
        """
        Allocates cache slots for the given pool_indices, evicting LRU slots if necessary.
        Returns the allocated slot indices.
        """
        if not pool_indices:
            return []

        # Convert input list to a long tensor on correct device
        pool_indices_t = torch.tensor(pool_indices, device=self.device, dtype=torch.long)

        # Automatically grow pool_to_slot if pool_idx exceeds the shape
        max_idx_t = pool_indices_t.max()
        if max_idx_t >= self.pool_to_slot.shape[0]:
            new_size = int(max_idx_t.item() * 2)
            new_tensor = torch.full((new_size,), -1, dtype=torch.int32, device=self.device)
            new_tensor[:self.pool_to_slot.shape[0]] = self.pool_to_slot
            self.pool_to_slot = new_tensor

            new_cpu = np.full((new_size,), -1, dtype=np.int32)
            new_cpu[:self.pool_to_slot_cpu.shape[0]] = self.pool_to_slot_cpu
            self.pool_to_slot_cpu = new_cpu

        with self._lock:
            self.step_counter += 1
            
            # Look up current slots
            slots = self.pool_to_slot[pool_indices_t]
            miss_mask = (slots < 0)
            
            if not miss_mask.any():
                # All Cache Hits! Update LRU scores and return slots
                self.lru_scores[slots.long()] = self.step_counter
                return slots.tolist()

            # Handle Cache Misses
            miss_pool_idxs = pool_indices_t[miss_mask]
            num_misses = miss_pool_idxs.shape[0]

            # Priority scores for slots: prefer free slots (-1.0), then occupied slots with oldest timestamps
            scores = torch.where(
                self.slot_to_pool == -1,
                torch.tensor(-1.0, device=self.device),
                self.lru_scores
            )

            # Find the num_misses slots with the smallest scores
            _, allocated_slots = torch.topk(scores, k=num_misses, largest=False, sorted=False)

            # Evict old pool indices
            old_pool_idxs = self.slot_to_pool[allocated_slots].long()
            valid_old_mask = (old_pool_idxs >= 0)
            if valid_old_mask.any():
                self.pool_to_slot[old_pool_idxs[valid_old_mask]] = -1
                self.pool_to_slot_cpu[old_pool_idxs[valid_old_mask].cpu().numpy()] = -1

            # Write new pool indices
            self.pool_to_slot[miss_pool_idxs] = allocated_slots.to(torch.int32)
            self.slot_to_pool[allocated_slots] = miss_pool_idxs.to(torch.int32)
            self.pool_to_slot_cpu[miss_pool_idxs.cpu().numpy()] = allocated_slots.cpu().numpy()

            # Update final slots list and update LRU scores
            final_slots = self.pool_to_slot[pool_indices_t]
            self.lru_scores[final_slots.long()] = self.step_counter

            return final_slots.tolist()

    def invalidate_pool_indices(self, pool_indices: list) -> None:
        """
        Invalidates slot mappings for given pool indices (e.g. when session ends or blocks are freed).
        """
        if not pool_indices:
            return
        with self._lock:
            pool_indices_t = torch.tensor(pool_indices, device=self.device, dtype=torch.long)
            pool_indices_t = pool_indices_t[pool_indices_t < self.pool_to_slot.shape[0]]
            if pool_indices_t.numel() == 0:
                return

            slots = self.pool_to_slot[pool_indices_t]
            valid_slots_mask = (slots >= 0)
            if not valid_slots_mask.any():
                return

            valid_slots = slots[valid_slots_mask].long()
            valid_pool_idxs = pool_indices_t[valid_slots_mask]

            # Clear mappings
            self.slot_to_pool[valid_slots] = -1
            self.pool_to_slot[valid_pool_idxs] = -1
            self.pool_to_slot_cpu[valid_pool_idxs.cpu().numpy()] = -1
            self.lru_scores[valid_slots] = 0.0

    def update_lru(self, hit_slots: list) -> None:
        """
        Updates the LRU timestamps for hit slots (called from decode hot path to avoid eviction).
        """
        if not hit_slots:
            return
        with self._lock:
            self.step_counter += 1
            hit_slots_t = torch.tensor(hit_slots, device=self.device, dtype=torch.long)
            self.lru_scores[hit_slots_t] = self.step_counter

    def clear(self):
        with self._lock:
            self.pool_to_slot.fill_(-1)
            self.pool_to_slot_cpu.fill(-1)
            self.slot_to_pool.fill_(-1)
            self.lru_scores.zero_()
            self.step_counter = 0
            self.K.zero_()
            self.V.zero_()


