"""
runtime/kv_runtime_manager.py

Manages KV cache residency and on-demand reconstruction for Differential KV.
Focuses on minimizing memory bandwidth and Python overhead.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from runtime.fused_reconstruction import fused_lowrank_reconstruct
from runtime.triton_diffkv import TritonDiffKV
from compression.quantization import dequantize_int8

@dataclass
class KVBlock:
    anchor_idx: int
    anchor_kv: torch.Tensor  # [2, heads, head_dim]
    U: Optional[torch.Tensor] = None # [block_size, rank]
    V: Optional[torch.Tensor] = None # [rank, feat_dim]
    scale: float = 1.0
    sparse_indices: Optional[torch.Tensor] = None
    sparse_values: Optional[torch.Tensor] = None
    token_indices: List[int] = None
    mode: str = "lowrank" # lowrank, int8, periodic
    q_deltas: Optional[torch.Tensor] = None # For int8 mode
    
    @property
    def is_compressed(self) -> bool:
        return self.U is not None or self.q_deltas is not None

class KVRuntimeManager:
    """
    Manages KV cache for a single layer or multiple layers.
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.config = config
        self.device = device
        self.mode = config.get("mode", "fp16") # fp16, int8, lowrank, lowrank_sparse
        self.block_size = config.get("block_size", 64)
        self.rank = config.get("rank", 16)
        
        # storage: layer_idx -> list of KVBlock
        self.cache: Dict[int, List[KVBlock]] = {}
        
    def add_block(self, layer_idx: int, block: KVBlock):
        if layer_idx not in self.cache:
            self.cache[layer_idx] = []
        self.cache[layer_idx].append(block)
        
    def reconstruct_layer(self, layer_idx: int, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        Reconstruct all blocks for a layer into a single contiguous tensor.
        """
        if layer_idx not in self.cache:
            return None
            
        blocks = self.cache[layer_idx]
        total_tokens = sum([len(b.token_indices) for b in blocks if b.token_indices])
        
        # Get dimensions from first anchor
        first_anchor = blocks[0].anchor_kv
        heads, head_dim = first_anchor.shape[-2:]
        
        out = torch.empty((total_tokens, 2, heads, head_dim), device=self.device, dtype=target_dtype)
        
        curr_idx = 0
        for block in blocks:
            # Anchor
            out[curr_idx] = block.anchor_kv.to(target_dtype)
            curr_idx += 1
            
            # Deltas
            if block.mode == "lowrank" or block.mode == "lowrank_sparse":
                if block.U is not None:
                    recon_deltas = TritonDiffKV.reconstruct_lowrank_sparse(
                        block.U,
                        block.V,
                        block.anchor_kv.reshape(-1),
                        block.sparse_indices,
                        block.sparse_values,
                        scale=block.scale
                    )
                    num_deltas = recon_deltas.shape[0]
                    # Reshape [num_deltas, feat_dim] -> [num_deltas, 2, heads, head_dim]
                    recon_deltas = recon_deltas.view(num_deltas, 2, heads, head_dim)
                    out[curr_idx : curr_idx + num_deltas] = recon_deltas.to(target_dtype)
                    curr_idx += num_deltas
            elif block.mode == "int8":
                if block.q_deltas is not None:
                    # [num_deltas, 2, heads, head_dim]
                    deltas = dequantize_int8(block.q_deltas, target_dtype=torch.float32)
                    heads, head_dim = block.anchor_kv.shape[-2:]
                    deltas = deltas.view(-1, 2, heads, head_dim)
                    recon = deltas + block.anchor_kv.float().unsqueeze(0)
                    num_deltas = recon.shape[0]
                    out[curr_idx : curr_idx + num_deltas] = recon.to(target_dtype)
                    curr_idx += num_deltas
            elif block.mode == "periodic":
                # Periodic is basically just raw anchors + raw deltas
                # In this manager, we store them as KVBlocks where U/V are None but token_indices are set
                if block.token_indices:
                    # We assume raw deltas are stored in some way if needed, 
                    # but periodic usually just means we don't compress.
                    pass
                
        return out

    def get_vram_usage(self) -> int:
        """Calculate total VRAM usage in bytes."""
        total_bytes = 0
        for layer_idx, blocks in self.cache.items():
            for block in blocks:
                total_bytes += block.anchor_kv.element_size() * block.anchor_kv.nelement()
                if block.U is not None:
                    total_bytes += block.U.element_size() * block.U.nelement()
                if block.V is not None:
                    total_bytes += block.V.element_size() * block.V.nelement()
                if block.sparse_values is not None:
                    total_bytes += block.sparse_values.element_size() * block.sparse_values.nelement()
                if block.sparse_indices is not None:
                    total_bytes += block.sparse_indices.element_size() * block.sparse_indices.nelement()
        return total_bytes

    def clear(self):
        """Clear all cached blocks."""
        self.cache = {}
