"""
native_core/srl/semantic_index.py

Approximate Nearest Neighbor search over block descriptor vectors.

At decode time, given a query descriptor q [DESC_DIM], this finds the
top-K blocks whose descriptors are most similar to q via a single matmul:
    scores = desc_matrix @ q   →   [N] cosine similarities

For N=781 blocks, DESC_DIM=64: this is ~50K multiplications — takes
<0.05ms on any modern GPU or CPU.

Build cost (once per prefill):
  - Gather all descriptors: O(N)
  - L2 normalize: O(N × DESC_DIM)
  - Total: ~0.02s for 781 blocks
"""

from __future__ import annotations
import sys
import os
# Add the build directory containing dkv_core.so to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.abspath(os.path.join(_script_dir, "../dkv_core"))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SemanticIndex:
    """
    Per-session ANN index over block descriptor vectors.

    desc_matrix[i] is the L2-normalized descriptor for pool slot slot_ids[i].
    All cosine similarities are computed as dot products (both sides normalized).
    """
    desc_matrix: torch.Tensor   # [N, DESC_DIM] float16 — L2 normalized
    slot_ids:    torch.Tensor   # [N] int32 — pool slot IDs (maps row → pool slot)

    # Vectorized reverse map (built lazily, zero Python-dict GC pressure):
    # _sorted_slots:      slot IDs sorted ascending        [N] int32
    # _sorted_to_orig:    permutation mapping sorted→orig  [N] int64
    _sorted_slots:    Optional[torch.Tensor] = field(default=None, repr=False)
    _sorted_to_orig:  Optional[torch.Tensor] = field(default=None, repr=False)

    def _build_sorted_index(self) -> None:
        """Build sorted-slot lookup tensors (CPU, called once per session)."""
        slot_cpu = self.slot_ids.cpu().to(torch.int32)
        perm = torch.argsort(slot_cpu)          # ascending sort permutation
        self._sorted_slots   = slot_cpu[perm]   # sorted slot values
        self._sorted_to_orig = perm             # maps sorted-idx → original row

    def slot_to_idx(self, slot_id: int) -> int:
        """Return the row index for a given pool slot ID (-1 if not found)."""
        if self._sorted_slots is None:
            self._build_sorted_index()
        pos = torch.bucketize(torch.tensor([slot_id], dtype=torch.int32),
                              self._sorted_slots, right=False)
        pos_i = int(pos[0])
        N = self._sorted_slots.shape[0]
        if pos_i >= N or int(self._sorted_slots[pos_i]) != slot_id:
            return -1
        return int(self._sorted_to_orig[pos_i])

    def slot_to_row_vec(self, slot_ids_t: torch.Tensor) -> torch.Tensor:
        """
        Vectorized reverse lookup: [M] int32 slot IDs → [M] int64 row indices.
        Slots not present in the index are mapped to -1.
        Runs entirely in C++ via torch.bucketize — zero Python loops.
        """
        if self._sorted_slots is None:
            self._build_sorted_index()
        q = slot_ids_t.cpu().to(torch.int32)
        pos = torch.bucketize(q, self._sorted_slots, right=False)   # [M]
        N = self._sorted_slots.shape[0]
        valid = (pos < N) & (self._sorted_slots[pos.clamp(max=N - 1)] == q)
        row_indices = torch.where(valid, self._sorted_to_orig[pos.clamp(max=N - 1)], torch.tensor(-1, dtype=torch.int64))
        return row_indices  # [M] int64, -1 for misses

    def search(self, q_desc: torch.Tensor, k: int) -> torch.Tensor:
        """
        Find top-k pool slot IDs by cosine similarity to q_desc.

        Args:
            q_desc: [DESC_DIM] float32, L2 normalized
            k:      number of results to return

        Returns:
            [min(k, N)] int32 tensor of pool slot IDs (not row indices)
        """
        N = self.desc_matrix.shape[0]
        if N == 0:
            return torch.tensor([], dtype=torch.int32, device=q_desc.device)

        k = min(k, N)
        try:
            import dkv_core as _dkv_core
            if getattr(_dkv_core, "HAS_SRL_ROUTER", False):
                top_k_indices = _dkv_core.semantic_search_topk(q_desc, self.desc_matrix, k)
                return self.slot_ids.to(top_k_indices.device)[top_k_indices]
        except ImportError:
            pass

        # Cast query to float16 to match desc_matrix dtype — faster matmul
        q16 = q_desc.half().to(self.desc_matrix.device)
        if q16.device.type == "mps":
            scores = self.desc_matrix.float() @ q16.float()
        else:
            scores = self.desc_matrix @ q16                  # [N]

        top_k_indices = torch.topk(scores, k=k, largest=True, sorted=True).indices
        return self.slot_ids.to(top_k_indices.device)[top_k_indices]              # pool slot IDs


def build_semantic_index(
    pool,                   # NativeBlockPool — has pool.desc [max_blocks, DESC_DIM]
    slot_ids: list,         # ordered list of active pool slot IDs for this session
) -> SemanticIndex:
    """
    Build a SemanticIndex from the descriptor vectors already written to the pool.

    Called once at the end of prefill compression, after all blocks are finalized.

    Args:
        pool:     NativeBlockPool instance (pool.desc must be populated)
        slot_ids: list[int] — pool slot IDs in chronological order

    Returns:
        SemanticIndex ready for decode-time ANN search
    """
    if not slot_ids:
        # Empty session — return empty index
        dummy = torch.zeros((0, 64), dtype=torch.float16)
        dummy_ids = torch.zeros((0,), dtype=torch.int32)
        return SemanticIndex(desc_matrix=dummy, slot_ids=dummy_ids)

    slot_tensor = torch.tensor(slot_ids, dtype=torch.long, device=pool.desc.device)
    desc_matrix = pool.desc[slot_tensor].clone()        # [N, DESC_DIM] float16

    # Re-normalize (descriptors were L2-normalized at write time, but ensure
    # precision after any pool growth/reset operations)
    norms = desc_matrix.float().norm(dim=1, keepdim=True).clamp(min=1e-8)
    desc_matrix = (desc_matrix.float() / norms).half()

    return SemanticIndex(
        desc_matrix = desc_matrix,
        slot_ids    = slot_tensor.to(torch.int32),
    )
