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
from runtime.triton_dkv import TritonDKV
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
    basis_id: Optional[int] = None # For shared basis
    
    @property
    def is_compressed(self) -> bool:
        return self.U is not None or self.q_deltas is not None

class KVRuntimeManager:
    """
    Manages KV cache residency and on-demand reconstruction for Differential KV.
    """
    def __init__(self, num_layers: int, heads: int, head_dim: int, device: str = "cuda"):
        self.num_layers = num_layers
        self.heads = heads
        self.head_dim = head_dim
        self.device = device
        
        # Session state: session_id -> {layer_idx -> List[KVBlock]}
        self.session_caches: Dict[str, Dict[int, List[KVBlock]]] = {}
        self.active_session: Optional[str] = None
        self.basis_cache: Dict[int, Dict[int, torch.Tensor]] = {} # layer_idx -> {basis_id: V}

    def switch_session(self, session_id: str):
        if session_id not in self.session_caches:
            self.session_caches[session_id] = {i: [] for i in range(self.num_layers)}
        self.active_session = session_id

    @property
    def active_cache(self) -> Dict[int, List[KVBlock]]:
        if not self.active_session:
            # Fallback for non-session-aware code
            if "default" not in self.session_caches:
                self.switch_session("default")
            return self.session_caches["default"]
        return self.session_caches[self.active_session]

    def add_block(self, layer_idx: int, block: KVBlock, session_id: Optional[str] = None):
        cache = self.session_caches[session_id] if session_id else self.active_cache
        if layer_idx not in cache:
            cache[layer_idx] = []
        cache[layer_idx].append(block)

    def update_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, session_id: Optional[str] = None):
        # Implementation for real-time updates (simplified)
        pass

    def add_basis(self, layer_idx: int, basis_id: int, V: torch.Tensor):
        if layer_idx not in self.basis_cache:
            self.basis_cache[layer_idx] = {}
        self.basis_cache[layer_idx][basis_id] = V
        
    def reconstruct_layer(self, layer_idx: int, session_id: Optional[str] = None, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        cache = self.session_caches[session_id] if session_id else self.active_cache
        if layer_idx not in cache or not cache[layer_idx]:
            return None
            
        blocks = cache[layer_idx]
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
            if block.mode in ["lowrank", "lowrank_sparse", "shared_basis"]:
                V = block.V
                if block.mode == "shared_basis" and block.basis_id is not None:
                    V = self.basis_cache.get(layer_idx, {}).get(block.basis_id)
                
                if block.U is not None and V is not None:
                    recon_deltas = TritonDKV.reconstruct_lowrank_sparse(
                        block.U, V, block.anchor_kv.reshape(-1),
                        block.sparse_indices, block.sparse_values, scale=block.scale
                    )
                    num_deltas = recon_deltas.shape[0]
                    max_deltas = len(block.token_indices) - 1
                    num_to_copy = min(num_deltas, max_deltas)
                    recon_deltas = recon_deltas.view(num_deltas, 2, heads, head_dim)
                    out[curr_idx : curr_idx + num_to_copy] = recon_deltas[:num_to_copy].to(target_dtype)
                    curr_idx += num_to_copy
            elif block.mode == "int8":
                if block.q_deltas is not None:
                    deltas = dequantize_int8(block.q_deltas, target_dtype=torch.float32)
                    deltas = deltas.view(-1, 2, heads, head_dim)
                    recon = deltas + block.anchor_kv.float().unsqueeze(0)
                    num_deltas = recon.shape[0]
                    max_deltas = len(block.token_indices) - 1
                    num_to_copy = min(num_deltas, max_deltas)
                    out[curr_idx : curr_idx + num_to_copy] = recon[:num_to_copy].to(target_dtype)
                    curr_idx += num_to_copy
                
        return out

    def get_vram_usage(self) -> int:
        total_bytes = 0
        for sid, cache in self.session_caches.items():
            for layer_idx, blocks in cache.items():
                for block in blocks:
                    total_bytes += block.anchor_kv.element_size() * block.anchor_kv.nelement()
                    if block.U is not None: total_bytes += block.U.element_size() * block.U.nelement()
                    if block.V is not None: total_bytes += block.V.element_size() * block.V.nelement()
        return total_bytes

    def clear(self):
        self.session_caches = {}
        self.basis_cache = {}
