"""
runtime/tiered_ffn.py

Phase 12 — Hierarchical Transformer Weight Residency

Implements real conditional weight materialization for Sparse MLPs.
Instead of keeping all FFN weights (up_proj, down_proj) in VRAM, this manager
stores them in pinned CPU RAM and dynamically pages them into a VRAM cache
when routed by the gate_proj.

This provides true VRAM reductions for model weights, at the cost of PCIe transfer latency.

Architecture:
- Cold Tier: Pinned CPU RAM [Total Blocks, Block Size, Hidden]
- Hot Tier: VRAM Cache [Budget Blocks, Block Size, Hidden]
- Eviction Policy: LRU

For Qwen2-7B:
Total FFN up/down blocks = 148 per layer.
We can budget e.g. 64 blocks in VRAM.
If a token routes to an un-cached block, we stall and transfer it over PCIe.
"""

import torch
from typing import List, Tuple, Dict
import time

class TieredFFNWeights:
    """
    Manages hierarchical residency for FFN weight matrices (up_proj and down_proj).
    """
    def __init__(
        self,
        W_up: torch.Tensor,       # [d_ff, hidden]
        W_down: torch.Tensor,     # [hidden, d_ff]
        block_size: int = 128,
        vram_budget_blocks: int = 64,
        device: str = "cuda"
    ):
        self.block_size = block_size
        self.d_ff = W_up.shape[0]
        self.hidden = W_up.shape[1]
        self.total_blocks = self.d_ff // block_size
        self.budget = vram_budget_blocks
        self.device = device

        assert W_down.shape == (self.hidden, self.d_ff), "W_down must be transposed"

        # ── COLD TIER: Pinned CPU RAM ──
        # Reshape to [total_blocks, block_size, hidden] for contiguous block transfers
        self.W_up_cpu = W_up.view(self.total_blocks, self.block_size, self.hidden).cpu().pin_memory()
        
        # W_down is [hidden, d_ff], which means columns are contiguous but we need block-wise access.
        # It's better to store W_down transposed [d_ff, hidden] so blocks are contiguous, 
        # then transpose on the fly in the kernel, or keep it [total_blocks, block_size, hidden].
        # For simplicity of transfer, we store as [d_ff, hidden] in CPU, and let Triton handle the stride,
        # or we just transpose it. Let's transpose it so it's [d_ff, hidden] like W_up.
        W_down_t = W_down.t().contiguous() # [d_ff, hidden]
        self.W_down_cpu = W_down_t.view(self.total_blocks, self.block_size, self.hidden).cpu().pin_memory()

        # ── HOT TIER: VRAM Cache ──
        self.cache_up   = torch.zeros((self.budget, self.block_size, self.hidden), device=device, dtype=W_up.dtype)
        self.cache_down = torch.zeros((self.budget, self.block_size, self.hidden), device=device, dtype=W_up.dtype)

        # Mapping: Block ID -> Cache Index
        self.block_to_cache: Dict[int, int] = {}
        # Mapping: Cache Index -> Block ID
        self.cache_to_block: Dict[int, int] = {}
        
        # LRU state
        self.lru_counter = 0
        self.cache_last_used = torch.zeros(self.budget, dtype=torch.long)

        # Diagnostics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "transfer_time_ms": 0.0,
            "queries": 0
        }
        
        # Initialize cache with first 'budget' blocks (default prior)
        self._init_cache()

    def _init_cache(self):
        """Pre-fill cache with the first N blocks."""
        init_blocks = min(self.budget, self.total_blocks)
        for i in range(init_blocks):
            self.cache_up[i].copy_(self.W_up_cpu[i], non_blocking=True)
            self.cache_down[i].copy_(self.W_down_cpu[i], non_blocking=True)
            self.block_to_cache[i] = i
            self.cache_to_block[i] = i
            self.cache_last_used[i] = self.lru_counter
            self.lru_counter += 1
        torch.cuda.synchronize()

    def fetch_blocks(self, active_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a list of active block IDs, ensure they are in VRAM.
        Returns the VRAM indices (cache indices) corresponding to the active IDs,
        and the hot tier tensors (so Triton can index them).
        
        Parameters:
            active_ids: [k] int32 tensor of requested block IDs.
        
        Returns:
            cache_indices: [k] int32 tensor indexing into the hot cache.
            cache_up: [budget, block_size, hidden] VRAM tensor.
            cache_down: [budget, block_size, hidden] VRAM tensor.
        """
        self.stats["queries"] += 1
        req_ids = active_ids.tolist()
        cache_indices = []
        
        misses = []
        
        # Phase 1: Check hits / misses
        for bid in req_ids:
            if bid in self.block_to_cache:
                cidx = self.block_to_cache[bid]
                cache_indices.append(cidx)
                self.cache_last_used[cidx] = self.lru_counter
                self.lru_counter += 1
                self.stats["hits"] += 1
            else:
                misses.append(bid)
                self.stats["misses"] += 1

        # Phase 2: Handle misses via LRU eviction and PCIe transfer
        if misses:
            t0 = time.perf_counter()
            
            # Find LRU slots
            num_misses = len(misses)
            # Find the 'num_misses' oldest cache entries
            _, lru_cidxs = torch.topk(self.cache_last_used, num_misses, largest=False)
            lru_cidxs = lru_cidxs.tolist()
            
            # Transfer
            for i, bid in enumerate(misses):
                cidx = lru_cidxs[i]
                
                # Evict old
                if cidx in self.cache_to_block:
                    old_bid = self.cache_to_block[cidx]
                    del self.block_to_cache[old_bid]
                    self.stats["evictions"] += 1
                
                # Load new
                self.cache_up[cidx].copy_(self.W_up_cpu[bid], non_blocking=True)
                self.cache_down[cidx].copy_(self.W_down_cpu[bid], non_blocking=True)
                
                # Update maps
                self.block_to_cache[bid] = cidx
                self.cache_to_block[cidx] = bid
                self.cache_last_used[cidx] = self.lru_counter
                self.lru_counter += 1
                
                cache_indices.append(cidx)
                
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.stats["transfer_time_ms"] += (t1 - t0) * 1000

        # Sort the output indices slightly differently if we wanted to maintain order,
        # but the Triton kernel just maps active_idx -> cache_idx. 
        # We need to return them in the same order as requested.
        # Re-build strictly ordered:
        final_indices = [self.block_to_cache[bid] for bid in req_ids]
        
        idx_tensor = torch.tensor(final_indices, dtype=torch.int32, device=self.device)
        return idx_tensor, self.cache_up, self.cache_down

    def get_summary(self) -> dict:
        hits = self.stats["hits"]
        misses = self.stats["misses"]
        total = max(1, hits + misses)
        return {
            "vram_budget_blocks": self.budget,
            "total_blocks": self.total_blocks,
            "vram_savings_mb": (self.total_blocks - self.budget) * self.block_size * self.hidden * 4 / 1024 / 1024, # 4 bytes for up+down fp16
            "hit_rate": round(hits / total, 4),
            "evictions": self.stats["evictions"],
            "avg_transfer_ms_per_query": round(self.stats["transfer_time_ms"] / max(1, self.stats["queries"]), 3),
        }
