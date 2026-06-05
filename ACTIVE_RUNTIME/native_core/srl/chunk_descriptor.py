"""
native_core/srl/chunk_descriptor.py

Semantic fingerprinting for KV blocks.

Each compressed block is assigned a 64-dimensional descriptor vector that
captures the "average topic direction" of its key space. Two blocks covering
the same topic will have high cosine similarity between their descriptors.

Design:
  - The descriptor is computed from anchor_K and the compressed U/V matrices
    already written to the NativeBlockPool — no raw K/V tensors needed.
  - A fixed random projection matrix W_proj [64, head_dim] is used.
    It is initialized once (stored on NativeBlockPool.W_proj) and never
    updated. This ensures descriptors from different blocks are comparable.
  - All descriptors are L2-normalized so cosine similarity = dot product.
  - Cost: ~3R + 2D multiplications per block during write_block() — negligible.

Block centroid estimation:
  centroid ≈ anchor_K_mean + mean(U_dequant) @ V_K_mean
  
  This approximates the average key direction of all tokens in the block
  without materializing them. Two topic-similar blocks will cluster together.
"""

from __future__ import annotations
import torch


DESC_DIM = 64  # descriptor vector dimension


def compute_descriptor(
    anchor_K:   torch.Tensor,   # [kv_heads, head_dim] float16
    U_int8:     torch.Tensor,   # [seq_len, rank] int8  (already quantized)
    U_scale:    torch.Tensor,   # scalar float16
    V_K:        torch.Tensor,   # [rank, kv_heads, head_dim] float16
    W_proj:     torch.Tensor,   # [DESC_DIM, head_dim] float32
) -> torch.Tensor:              # [DESC_DIM] float16
    """
    Compute a 64-dim semantic fingerprint for one compressed block.

    Called inside NativeBlockPool.write_block() after all pool tensors
    have been written. Reads data back from pool tensors (already on device).
    """
    # ── Step 1: Block centroid in key space ───────────────────────────────
    # anchor contributes its mean over kv heads
    anchor_mean = anchor_K.float().mean(dim=0)          # [head_dim]

    # Dequantize U: [seq_len, rank] float32
    U_f32 = U_int8.float() * U_scale.float()            # [seq_len, rank]

    # Mean of U activations: [rank]
    mean_u = U_f32.mean(dim=0)                          # [rank]

    # V_K mean over kv_heads and projected to head_dim
    # V_K shape: [rank, kv_heads, head_dim]
    vk_mean = V_K.float().mean(dim=1)                   # [rank, head_dim]

    # Delta centroid: mean_u [rank] @ vk_mean [rank, head_dim] → [head_dim]
    delta_centroid = mean_u @ vk_mean                   # [head_dim]

    centroid = anchor_mean + delta_centroid              # [head_dim]

    # ── Step 2: Project to descriptor space ───────────────────────────────
    # W_proj: [DESC_DIM, head_dim]
    desc = W_proj.float() @ centroid                    # [DESC_DIM]

    # ── Step 3: L2 normalize (enables cosine similarity via dot product) ──
    desc = desc / (desc.norm() + 1e-8)

    return desc.half()


def compute_query_descriptor(
    Q:      torch.Tensor,   # [H, D] float16/float32 — current query (all heads)
    W_proj: torch.Tensor,   # [DESC_DIM, D] float32
) -> torch.Tensor:          # [DESC_DIM] float32, L2 normalized
    """
    Compute a query descriptor matching the same space as block descriptors.

    Called once per decode step in route_query().
    """
    q_mean = Q.float().mean(dim=0)          # [D] — average over query heads
    desc   = W_proj.float() @ q_mean        # [DESC_DIM]
    desc   = desc / (desc.norm() + 1e-8)    # L2 normalize
    return desc
