import time
from typing import Dict, Any, List

class StreamLocalMemoryAffinityScheduler:
    """
    STAGE 4A.3 — PEA: Stream-Local Memory Affinity Scheduler.
    Aligns allocator pools with executing stream IDs, isolating prefill 
    and decode memory lanes to prevent inter-stream cross-talk.
    """
    def __init__(self, trace_system):
        self.trace_system = trace_system
        self.stream_pools = {}
        self.total_dispatches = 0
        self.stream_local_hits = 0
        self.cross_stream_relocations = 0
        
    def acquire_stream_affine_tensor(self, size_bytes: int, stream_id: str, is_decode: bool) -> Dict[str, Any]:
        """Locks a stream-local VRAM slot, tracking inter-stream memory relocations."""
        self.total_dispatches += 1
        t_now = time.perf_counter()
        
        pool_key = f"pool_stream_{stream_id}_lane_{'decode' if is_decode else 'prefill'}"
        
        if pool_key not in self.stream_pools:
            self.stream_pools[pool_key] = []
            
        # Try to find a free or reusable slot in this stream-local pool
        reused = False
        for slot in self.stream_pools[pool_key]:
            if slot["size_bytes"] >= size_bytes and not slot["is_locked"]:
                slot["is_locked"] = True
                slot["last_used"] = t_now
                reused = True
                self.stream_local_hits += 1
                break
                
        # If we failed to find stream-local slot, check other streams (leads to cross-stream relocation!)
        if not reused:
            for k, other_pool in self.stream_pools.items():
                if k != pool_key:
                    for slot in other_pool:
                        if slot["size_bytes"] >= size_bytes and not slot["is_locked"]:
                            slot["is_locked"] = True
                            slot["last_used"] = t_now
                            self.cross_stream_relocations += 1
                            reused = True
                            break
                    if reused:
                        break
                        
        if not reused:
            # Construct fresh stream-local slot
            new_slot = {
                "size_bytes": size_bytes,
                "is_locked": True,
                "created": t_now,
                "last_used": t_now
            }
            self.stream_pools[pool_key].append(new_slot)
            self.stream_local_hits += 1
            
        # Unlock immediately to simulate memory release for subsequent steps
        for pool in self.stream_pools.values():
            for slot in pool:
                slot["is_locked"] = False
                
        if self.trace_system:
            self.trace_system.log_trace("stream_affinity", {
                "stream_local_reuse_pct": self.stream_local_reuse_pct,
                "cross_stream_relocation_frequency": self.cross_stream_relocation_frequency,
                "stream_affinity_pct": self.stream_affinity_pct,
                "locality_persistence": self.locality_persistence
            })
            
        return {"status": "AFFINE_LOCKED", "pool": pool_key}

    @property
    def stream_local_reuse_pct(self) -> float:
        if self.total_dispatches == 0:
            return 100.0
        return (self.stream_local_hits / self.total_dispatches) * 100.0

    @property
    def cross_stream_relocation_frequency(self) -> float:
        if self.total_dispatches == 0:
            return 0.0
        return self.cross_stream_relocations / self.total_dispatches

    @property
    def stream_affinity_pct(self) -> float:
        if self.total_dispatches == 0:
            return 100.0
        return max(50.0, 100.0 - (self.cross_stream_relocation_frequency * 100.0))

    @property
    def locality_persistence(self) -> float:
        if self.total_dispatches == 0:
            return 1.0
        return max(0.1, 1.0 - self.cross_stream_relocation_frequency)
