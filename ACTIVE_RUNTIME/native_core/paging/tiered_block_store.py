"""
TieredBlockStore: Proactive, heat-scored CPU offloading layer for PagedKVStore.

This module provides a tiered storage management system for KV cache blocks.
It wraps an existing PagedKVStore and NativeBlockPool, using a heat-scoring
system to proactively evict cold GPU blocks to CPU RAM, and restoring them
when needed. For CUDA devices, Host-To-Device (H2D) transfers are performed
asynchronously using dedicated CUDA streams and events.
"""

import os
import threading
import time
from typing import Dict, List, Optional, Any

import torch


class TieredBlockStore:
    def __init__(
        self,
        pool,
        pager,
        device: str,
        evict_threshold: float = 0.80,
        evict_batch: int = 32
    ):
        self.pool = pool
        self.pager = pager
        self.device = str(device)
        self.evict_threshold = evict_threshold
        self.evict_batch = evict_batch

        self._heat: Dict[int, float] = {}
        self._tier: Dict[int, str] = {}
        self._cpu_store: Dict[int, Dict[str, torch.Tensor]] = {}
        self._warm_futures: Dict[int, Optional[Any]] = {}

        self._evictions = 0
        self._restores = 0

        self.DECAY = 0.95

        self.is_cuda = self.device.startswith("cuda")
        self._h2d_stream = None
        if self.is_cuda and torch.cuda.is_available():
            self._h2d_stream = torch.cuda.Stream(device=self.device)
            
    def update_heat(self, slot_id: int, routing_score: float = 1.0):
        current_heat = self._heat.get(slot_id, 0.0)
        new_heat = 0.7 * current_heat * self.DECAY + 0.3 * routing_score
        self._heat[slot_id] = max(0.0, min(1.0, new_heat))
        
        # If the slot was on CPU, mark it as routing-active
        if self._tier.get(slot_id) == 'CPU':
            pass

    def decay_all(self, step: int = 1):
        decay_factor = self.DECAY ** step
        for slot_id in self._heat:
            self._heat[slot_id] *= decay_factor

    def maybe_evict(self, occupied_slots: List[int]) -> int:
        if getattr(self.pool, 'max_blocks', 0) <= 0:
            return 0
            
        occupancy = len(occupied_slots) / float(self.pool.max_blocks)
        if occupancy <= self.evict_threshold:
            return 0

        # Filter out WARMING slots or those not on GPU
        evictable_slots = [
            s for s in occupied_slots
            if self._tier.get(s, 'GPU') == 'GPU'
        ]

        # Sort by heat ascending
        evictable_slots.sort(key=lambda s: self._heat.get(s, 0.0))

        evict_count = 0
        for slot_id in evictable_slots[:self.evict_batch]:
            if self.evict_slot(slot_id):
                evict_count += 1

        return evict_count

    def evict_slot(self, slot_id: int) -> bool:
        if self._tier.get(slot_id, 'GPU') != 'GPU':
            return False

        try:
            # Read pool slot tensors
            u = self.pool.U[slot_id]
            u_scale = self.pool.U_scale[slot_id]
            v_kv = self.pool.V_KV[slot_id]
            anchors_kv = self.pool.anchors_KV[slot_id]
            seq_len = self.pool.seq_lens[slot_id].item()

            if self.is_cuda and torch.cuda.is_available():
                # Use pinned memory for faster CPU<->GPU transfers
                cpu_u = torch.empty_like(u, device='cpu', pin_memory=True).copy_(u)
                cpu_u_scale = torch.empty_like(u_scale, device='cpu', pin_memory=True).copy_(u_scale)
                cpu_v_kv = torch.empty_like(v_kv, device='cpu', pin_memory=True).copy_(v_kv)
                cpu_anchors_kv = torch.empty_like(anchors_kv, device='cpu', pin_memory=True).copy_(anchors_kv)
            else:
                cpu_u = u.contiguous().cpu()
                cpu_u_scale = u_scale.contiguous().cpu()
                cpu_v_kv = v_kv.contiguous().cpu()
                cpu_anchors_kv = anchors_kv.contiguous().cpu()

            self._cpu_store[slot_id] = {
                'U': cpu_u,
                'U_scale': cpu_u_scale,
                'V_KV': cpu_v_kv,
                'anchors_KV': cpu_anchors_kv,
                'seq_len': seq_len
            }
            
            # Zero GPU slot to mark empty
            self.pool.seq_lens[slot_id] = 0
            self._tier[slot_id] = 'CPU'
            self._evictions += 1
            return True
        except Exception as e:
            return False

    def ensure_warm(self, slot_ids: List[int]) -> List[int]:
        warming = []
        for slot_id in slot_ids:
            tier = self.get_tier(slot_id)
            if tier == 'CPU':
                self.restore_slot(slot_id, blocking=False)
                warming.append(slot_id)
            elif tier == 'WARMING':
                warming.append(slot_id)
        return warming

    def restore_slot(self, slot_id: int, blocking: bool = False) -> bool:
        if slot_id not in self._cpu_store:
            return False

        store_data = self._cpu_store[slot_id]
        self._tier[slot_id] = 'WARMING'

        if self.is_cuda and torch.cuda.is_available() and self._h2d_stream is not None:
            with torch.cuda.stream(self._h2d_stream):
                self.pool.U[slot_id].copy_(store_data['U'], non_blocking=not blocking)
                self.pool.U_scale[slot_id].copy_(store_data['U_scale'], non_blocking=not blocking)
                self.pool.V_KV[slot_id].copy_(store_data['V_KV'], non_blocking=not blocking)
                self.pool.anchors_KV[slot_id].copy_(store_data['anchors_KV'], non_blocking=not blocking)
                self.pool.seq_lens[slot_id] = store_data['seq_len']
                
                event = torch.cuda.Event()
                event.record(self._h2d_stream)
                self._warm_futures[slot_id] = event
                
            if blocking:
                event.synchronize()
                self._tier[slot_id] = 'GPU'
                del self._cpu_store[slot_id]
                self._warm_futures.pop(slot_id, None)
        else:
            # CPU/MPS synchronous copy
            self.pool.U[slot_id].copy_(store_data['U'].to(self.device))
            self.pool.U_scale[slot_id].copy_(store_data['U_scale'].to(self.device))
            self.pool.V_KV[slot_id].copy_(store_data['V_KV'].to(self.device))
            self.pool.anchors_KV[slot_id].copy_(store_data['anchors_KV'].to(self.device))
            self.pool.seq_lens[slot_id] = store_data['seq_len']
            
            self._tier[slot_id] = 'GPU'
            del self._cpu_store[slot_id]
            self._warm_futures.pop(slot_id, None)

        self._restores += 1
        return True

    def sync_warming_slots(self, slot_ids: List[int]):
        for slot_id in slot_ids:
            if self.get_tier(slot_id) == 'WARMING':
                event = self._warm_futures.get(slot_id)
                if event is not None and self.is_cuda and torch.cuda.is_available():
                    event.synchronize()
                    torch.cuda.current_stream().wait_event(event)
                
                self._tier[slot_id] = 'GPU'
                if slot_id in self._cpu_store:
                    del self._cpu_store[slot_id]
                self._warm_futures.pop(slot_id, None)

    def get_tier(self, slot_id: int) -> str:
        return self._tier.get(slot_id, 'GPU')
        
    def warm_slot_count(self) -> int:
        return sum(1 for tier in self._tier.values() if tier == 'GPU')
        
    def cold_slot_count(self) -> int:
        return sum(1 for tier in self._tier.values() if tier == 'CPU')
        
    def stats(self) -> dict:
        warming_count = sum(1 for tier in self._tier.values() if tier == 'WARMING')
        return {
            'evictions': self._evictions,
            'restores': self._restores,
            'warm_count': self.warm_slot_count(),
            'cold_count': self.cold_slot_count(),
            'warming_count': warming_count
        }
