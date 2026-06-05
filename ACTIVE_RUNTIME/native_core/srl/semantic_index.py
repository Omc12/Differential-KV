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
from dataclasses import dataclass
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

    # Reverse map: pool_slot_id → row index in desc_matrix (built lazily)
    _slot_to_idx: Optional[dict] = None

    def slot_to_idx(self, slot_id: int) -> int:
        """Return the row index for a given pool slot ID."""
        if self._slot_to_idx is None:
            self._slot_to_idx = {
                int(s): i for i, s in enumerate(self.slot_ids.tolist())
            }
        return self._slot_to_idx.get(slot_id, -1)

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
        # Cast query to float16 to match desc_matrix dtype — faster matmul
        q16 = q_desc.half().to(self.desc_matrix.device)
        if q16.device.type == "mps":
            scores = (self.desc_matrix.to("cpu") @ q16.to("cpu")).to("mps")
        else:
            scores = self.desc_matrix @ q16                  # [N]

        top_k_indices = torch.topk(scores, k=k, largest=True, sorted=True).indices
        return self.slot_ids[top_k_indices]              # pool slot IDs


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
