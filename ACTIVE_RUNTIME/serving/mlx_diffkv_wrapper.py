import os
import sys
import time
import math
import re
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load as mlx_load
import torch

def _normalize_references(text: str) -> str:
    """Normalise citation-list formatting inconsistencies produced by the model."""
    lines = text.split('\n')
    
    # 1. Search for a reference header line
    header_re = re.compile(r'\b(references?|bibliography|works\s+cited|reference\s+list|sources|citations)\b', re.IGNORECASE)
    header_idx = None
    for i, line in enumerate(lines):
        if len(line) <= 100 and header_re.search(line):
            header_idx = i
    
    # 2. Find matching reference entries
    ref_entry_re = re.compile(r'^(?:[iI]n\s+)?(?:[*\-•]\s*)?\[\d+\]')
    unambiguous_re = re.compile(r'^(?:[*\-•]\s*)?\[\d+\]')
    
    matching_indices = []
    unambiguous_indices = []
    for i, line in enumerate(lines):
        if header_idx is not None and i <= header_idx:
            continue
        stripped = line.strip()
        if ref_entry_re.match(stripped):
            matching_indices.append(i)
            if unambiguous_re.match(stripped):
                unambiguous_indices.append(i)
                
    if header_idx is not None and not matching_indices:
        matching_indices = []
        unambiguous_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ref_entry_re.match(stripped):
                matching_indices.append(i)
                if unambiguous_re.match(stripped):
                    unambiguous_indices.append(i)
        header_idx = None

    if not matching_indices:
        return text
        
    if header_idx is not None:
        ref_start_idx = header_idx + 1
    elif unambiguous_indices:
        ref_start_idx = unambiguous_indices[0]
    else:
        return text
                
    body = '\n'.join(lines[:ref_start_idx])
    ref_block = '\n'.join(lines[ref_start_idx:])
    
    pattern = re.compile(
        r'^\s*'
        r'(?:[iI]n\s+)?'
        r'(?:[*\-•]\s*)?'
        r'(\[\d+\])'
        r'(?:,\s*|\.\s*|\s+)?',
        re.MULTILINE
    )
    normalized_ref_block = pattern.sub(r'\1 ', ref_block)
    
    if body:
        return body + '\n' + normalized_ref_block
    return normalized_ref_block

class MLXCompressedBlock:
    def __init__(self, anchor_idx: int, token_indices: List[int], U: mx.array, V_K: mx.array, V_V: mx.array, anchor_k: mx.array, anchor_v: mx.array, scale: float, seq_len: int):
        self.anchor_idx = anchor_idx
        self.token_indices = token_indices
        self.U = U                  # mx.array [S_comp, R]
        self.V_K = V_K              # mx.array [H_kv, R, D]
        self.V_V = V_V              # mx.array [H_kv, R, D]
        self.anchor_k = anchor_k    # mx.array [H_kv, D]
        self.anchor_v = anchor_v    # mx.array [H_kv, D]
        self.scale = scale          # float
        self.seq_len = seq_len      # int
        
    def clone(self):
        return MLXCompressedBlock(
            self.anchor_idx,
            self.token_indices.copy(),
            self.U,
            self.V_K,
            self.V_V,
            self.anchor_k,
            self.anchor_v,
            self.scale,
            self.seq_len
        )

def compress_mlx_block(deltas: mx.array, rank: int, n_oversamples: int = 5, n_iter: int = 2) -> Tuple[mx.array, mx.array, float, int]:
    """Compress a block of KV delta vectors using randomised truncated SVD.

    Runs entirely in NumPy (CPU) to avoid MLX GPU-op limitations:
      - mx.linalg.qr  is GPU-only in current MLX builds
      - mx.where(cond) single-arg (nonzero) form doesn't accept stream kwarg
    NumPy SVD on a 255x512 block takes ~2 ms -- acceptable since compression
    fires at most once every block_size (256) tokens.
    """
    n, d = deltas.shape
    rank = min(rank, n, d)
    if rank < 1:
        return mx.zeros((n, 1), dtype=deltas.dtype), mx.zeros((1, d), dtype=deltas.dtype), 1.0, 1

    # Materialise the MLX array into NumPy (unified memory, near zero-copy on Apple Silicon)
    mx.eval(deltas)
    x_np = np.array(deltas, copy=False).astype(np.float32)

    scale = float(np.max(np.abs(x_np)))
    if scale < 1e-9:
        return mx.zeros((n, rank), dtype=deltas.dtype), mx.zeros((rank, d), dtype=deltas.dtype), 1.0, rank

    x_np = x_np / scale

    # Randomised range finder
    r_proj = min(rank + n_oversamples, n, d)
    rng = np.random.default_rng()
    Omega = rng.standard_normal((d, r_proj)).astype(np.float32)
    Y = x_np @ Omega
    for _ in range(n_iter):
        Y = x_np @ (x_np.T @ Y)

    try:
        # QR + projected SVD (all NumPy, always CPU)
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ x_np
        U_b, S, Vh = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_b

        # Adaptive rank: keep components that explain 99.9% of energy
        total_energy = float(np.sum(S ** 2))
        k = rank
        if total_energy > 1e-9:
            cum = np.cumsum(S ** 2)
            idx = np.where(cum >= 0.999 * total_energy)[0]
            if len(idx) > 0:
                k = max(4, min(int(idx[0]) + 1, rank))

        U_k  = (U[:, :k] * S[:k]).astype(np.float16)
        Vh_k = Vh[:k, :].astype(np.float16)
        return mx.array(U_k), mx.array(Vh_k), scale, k

    except np.linalg.LinAlgError:
        # SVD did not converge (rare, numerically degenerate block) -- return identity-like
        k = min(rank, n, d)
        U_k  = np.zeros((n, k), dtype=np.float16)
        Vh_k = np.zeros((k, d), dtype=np.float16)
        return mx.array(U_k), mx.array(Vh_k), 1.0, k

@mx.compile
def compute_decode_attention_static(
    q: mx.array,              # [H_q, D]
    comp_U: mx.array,         # [max_blocks, S_comp, rank]
    comp_VK: mx.array,        # [max_blocks, kv_heads, rank, head_dim]
    comp_VV: mx.array,        # [max_blocks, kv_heads, rank, head_dim]
    comp_anc_k: mx.array,     # [max_blocks, kv_heads, head_dim]
    comp_anc_v: mx.array,     # [max_blocks, kv_heads, head_dim]
    comp_scale: mx.array,     # [max_blocks]
    comp_seq_len: mx.array,   # [max_blocks]
    num_blocks: mx.array,     # scalar
    dense_k: mx.array,        # [kv_heads, max_dense_len, head_dim]
    dense_v: mx.array,        # [kv_heads, max_dense_len, head_dim]
    dense_len: mx.array,      # scalar
    scale: float,
    gpk: int,
    kv_heads: int,
    block_size: int,
    rank: int,
    max_blocks: int,
    max_dense_len: int
):
    H_q, D = q.shape
    S_comp = block_size - 1
    
    # 1. Sparse / Compressed Attention
    block_idx = mx.arange(max_blocks)
    block_mask = block_idx < num_blocks
    block_mask_expanded = mx.expand_dims(block_mask, 0) # [1, max_blocks]
    
    AncK_e = comp_anc_k
    AncV_e = comp_anc_v
    VK_e = comp_VK
    VV_e = comp_VV
    
    if gpk > 1:
        AncK_e = mx.repeat(AncK_e, gpk, axis=1)
        AncV_e = mx.repeat(AncV_e, gpk, axis=1)
        VK_e = mx.repeat(VK_e, gpk, axis=1)
        VV_e = mx.repeat(VV_e, gpk, axis=1)
        
    AncK_e_perm = AncK_e.transpose(1, 0, 2)
    AncV_e_perm = AncV_e.transpose(1, 0, 2)
    
    s_anc = mx.sum(mx.expand_dims(q, 1) * AncK_e_perm, axis=-1) * scale
    
    VK_e_perm = VK_e.transpose(1, 0, 2, 3)
    q_expanded = mx.expand_dims(mx.expand_dims(q, 1), 2)
    q_proj_n = mx.sum(q_expanded * VK_e_perm, axis=-1) * scale
    
    # Fix broadcasting for delta_s
    q_proj_n_perm = q_proj_n.transpose(1, 0, 2) # [max_blocks, H_q, rank]
    comp_U_transposed = comp_U.transpose(0, 2, 1) # [max_blocks, rank, S_comp]
    comp_U_transposed_exp = mx.expand_dims(comp_U_transposed, 1) # [max_blocks, 1, rank, S_comp]
    q_proj_n_exp = mx.expand_dims(q_proj_n_perm, 2) # [max_blocks, H_q, 1, rank]
    
    delta_s = mx.matmul(q_proj_n_exp, comp_U_transposed_exp).squeeze(2) # [max_blocks, H_q, S_comp]
    delta_s = delta_s.transpose(1, 0, 2) # [H_q, max_blocks, S_comp]
    
    delta_s = delta_s * comp_scale.reshape(1, -1, 1)
    delta_s = delta_s + mx.expand_dims(s_anc, -1)
    
    s_range = mx.arange(S_comp).reshape(1, 1, -1)
    valid_msk = s_range < comp_seq_len.reshape(1, -1, 1)
    delta_s = mx.where(valid_msk, delta_s, -float('inf'))
    
    scores_blocks = mx.concatenate([mx.expand_dims(s_anc, -1), delta_s], axis=-1)
    scores_sparse = scores_blocks.reshape(H_q, -1)
    
    block_mask_sparse = mx.repeat(block_mask_expanded, block_size, axis=1)
    scores_sparse = mx.where(block_mask_sparse, scores_sparse, -float('inf'))
    
    lse_sparse = mx.logsumexp(scores_sparse, axis=-1)
    w = mx.softmax(scores_sparse, axis=-1)
    
    W_comp = w.reshape(H_q, max_blocks, block_size)
    w_anc = W_comp[:, :, 0]
    w_d = W_comp[:, :, 1:]
    
    w_block_sum = w_anc + mx.sum(w_d, axis=-1)
    O_anc = mx.sum(mx.expand_dims(w_block_sum, -1) * AncK_e_perm, axis=1)
    
    # Fix broadcasting for w_proj
    w_d_perm = w_d.transpose(1, 0, 2) # [max_blocks, H_q, S_comp]
    comp_U_exp = mx.expand_dims(comp_U, 1) # [max_blocks, 1, S_comp, rank]
    w_proj = mx.matmul(mx.expand_dims(w_d_perm, 2), comp_U_exp).squeeze(2) # [max_blocks, H_q, rank]
    w_proj = w_proj * comp_scale.reshape(-1, 1, 1)
    
    # Batch matrix multiplication: VV_e has shape [max_blocks, H_q, rank, head_dim]
    O_delta_block = mx.matmul(mx.expand_dims(w_proj, 2), VV_e).squeeze(2) # [max_blocks, H_q, head_dim]
    O_delta = mx.sum(O_delta_block, axis=0) # [H_q, head_dim]
    
    out_sparse = O_anc + O_delta
    out_sparse = mx.where(mx.isnan(out_sparse), 0.0, out_sparse)
    
    # 2. Dense Attention
    dense_idx = mx.arange(max_dense_len)
    dense_mask = dense_idx < dense_len
    dense_mask_expanded = mx.expand_dims(dense_mask, 0)
    
    if gpk > 1:
        dense_k_rot_perm = mx.repeat(dense_k, gpk, axis=0)
        dense_v_perm = mx.repeat(dense_v, gpk, axis=0)
    else:
        dense_k_rot_perm = dense_k
        dense_v_perm = dense_v
        
    scores_dense = mx.sum(mx.expand_dims(q, 1) * dense_k_rot_perm, axis=-1) * scale
    scores_dense = mx.where(dense_mask_expanded, scores_dense, -float('inf'))
    
    lse_dense = mx.logsumexp(scores_dense, axis=-1)
    weights_dense = mx.softmax(scores_dense, axis=-1)
    out_dense = mx.sum(mx.expand_dims(weights_dense, -1) * dense_v_perm, axis=1)
    out_dense = mx.where(mx.isnan(out_dense), 0.0, out_dense)
    
    # 3. Combined
    lses = mx.stack([lse_sparse, lse_dense], axis=0)
    lse_max = mx.max(lses, axis=0)
    w_sparse = mx.exp(lse_sparse - lse_max)
    w_dense = mx.exp(lse_dense - lse_max)
    denom = w_sparse + w_dense
    
    out_combined = (out_sparse * mx.expand_dims(w_sparse, -1) + out_dense * mx.expand_dims(w_dense, -1)) / mx.expand_dims(denom, -1)
    return out_combined

class DummyMLXPool:
    def __init__(self, manager):
        self.manager = manager

    @property
    def _free_indices(self):
        allocated = 0
        for session in self.manager.sessions.values():
            comp_len = session["num_blocks"][0] * self.manager.block_size
            dense_len = session["dense_lens"][0]
            total_len = comp_len + dense_len
            num_logical_blocks = (total_len + self.manager.block_size - 1) // self.manager.block_size
            allocated += num_logical_blocks
        free_count = max(0, self.manager.max_blocks - allocated)
        return [0] * free_count

    @property
    def current_blocks(self):
        return self.manager.max_blocks

class MLXKVBlockManager:
    @property
    def native_pool(self):
        return DummyMLXPool(self)

    def __init__(self, num_layers: int, heads: int, kv_heads: int, head_dim: int, rank: int, block_size: int, recency_window: int = 512):
        self.num_layers = num_layers
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.rank = rank
        self.block_size = block_size
        
        env_engage = os.environ.get("DIFFKV_ENGAGE_THRESHOLD")
        if env_engage is not None:
            try:
                recency_window = int(env_engage)
            except ValueError:
                pass
        self.recency_window = recency_window
        
        # Trim block/dense limits to save RAM:
        # max_blocks=32 handles up to 32*256=8192 compressed tokens (enough for long docs)
        # max_dense_len is exactly what we need: recency window + one block to absorb
        self.max_blocks = 32
        self.max_dense_len = self.recency_window + self.block_size
        
        self.sessions = {}
        self.active_session_ids = ["default"]
        self.position_ids = None
        self._session_token_ids = {}
        self._session_checkpoints = {}

    def get_srl_state(self, session_id: str):
        return None

    def _create_empty_session(self) -> Dict[str, Any]:
        # Use float16 explicitly to halve the RAM vs float32 defaults
        dtype = mx.float16
        return {
            "dense_keys":   [mx.zeros((1, self.kv_heads, self.max_dense_len, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "dense_values": [mx.zeros((1, self.kv_heads, self.max_dense_len, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "dense_lens":   [0 for _ in range(self.num_layers)],
            
            "num_blocks": [0 for _ in range(self.num_layers)],
            "comp_U":     [mx.zeros((self.max_blocks, self.block_size - 1, self.rank), dtype=dtype) for _ in range(self.num_layers)],
            "comp_VK":    [mx.zeros((self.max_blocks, self.kv_heads, self.rank, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_VV":    [mx.zeros((self.max_blocks, self.kv_heads, self.rank, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_anc_k": [mx.zeros((self.max_blocks, self.kv_heads, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_anc_v": [mx.zeros((self.max_blocks, self.kv_heads, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_scale":    [mx.zeros((self.max_blocks,)) for _ in range(self.num_layers)],
            "comp_seq_len": [mx.zeros((self.max_blocks,), dtype=mx.int32) for _ in range(self.num_layers)],
            
            "token_ids": []
        }

    def init_session(self, session_id: str, prefill_len: int = 0, max_tokens_hint: int = None):
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_empty_session()

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def snapshot_session(self, session_id: str, checkpoint_id: str):
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found to snapshot.")
        src = self.sessions[session_id]
        self._session_checkpoints[checkpoint_id] = {
            "dense_keys": [mx.array(k) for k in src["dense_keys"]],
            "dense_values": [mx.array(v) for v in src["dense_values"]],
            "dense_lens": src["dense_lens"].copy(),
            "num_blocks": src["num_blocks"].copy(),
            "comp_U": [mx.array(u) for u in src["comp_U"]],
            "comp_VK": [mx.array(vk) for vk in src["comp_VK"]],
            "comp_VV": [mx.array(vv) for vv in src["comp_VV"]],
            "comp_anc_k": [mx.array(ak) for ak in src["comp_anc_k"]],
            "comp_anc_v": [mx.array(av) for av in src["comp_anc_v"]],
            "comp_scale": [mx.array(s) for s in src["comp_scale"]],
            "comp_seq_len": [mx.array(sl) for src_sl, sl in zip(src["comp_seq_len"], src["comp_seq_len"])], # copy mx arrays properly
            "token_ids": src["token_ids"].copy() if "token_ids" in src else []
        }

    def restore_session(self, session_id: str, checkpoint_id: str):
        if checkpoint_id not in self._session_checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found.")
        ckpt = self._session_checkpoints[checkpoint_id]
        self.sessions[session_id] = {
            "dense_keys": [mx.array(k) for k in ckpt["dense_keys"]],
            "dense_values": [mx.array(v) for v in ckpt["dense_values"]],
            "dense_lens": ckpt["dense_lens"].copy(),
            "num_blocks": ckpt["num_blocks"].copy(),
            "comp_U": [mx.array(u) for u in ckpt["comp_U"]],
            "comp_VK": [mx.array(vk) for vk in ckpt["comp_VK"]],
            "comp_VV": [mx.array(vv) for vv in ckpt["comp_VV"]],
            "comp_anc_k": [mx.array(ak) for ak in ckpt["comp_anc_k"]],
            "comp_anc_v": [mx.array(av) for av in ckpt["comp_anc_v"]],
            "comp_scale": [mx.array(s) for s in ckpt["comp_scale"]],
            "comp_seq_len": [mx.array(sl) for sl in ckpt["comp_seq_len"]],
            "token_ids": ckpt["token_ids"].copy() if "token_ids" in ckpt else []
        }

    def delete_checkpoint(self, checkpoint_id: str):
        self._session_checkpoints.pop(checkpoint_id, None)

    def get_streaming_summary(self, session_id: str = None) -> dict:
        return {"streaming_ingest": False}

    def get_streaming_blocks(self, session_id: str, layer_idx: int) -> list:
        session = self.sessions.get(session_id)
        if session is None:
            return []
        comp_len = session["num_blocks"][layer_idx] * self.block_size
        dense_len = session["dense_lens"][layer_idx]
        total_len = comp_len + dense_len
        num_logical_blocks = (total_len + self.block_size - 1) // self.block_size
        return [object() for _ in range(num_logical_blocks)]

    def get_raw_blocks(self, session_id: str, layer_idx: int) -> list:
        session = self.sessions.get(session_id)
        if session is None:
            return []
        comp_len = session["num_blocks"][layer_idx] * self.block_size
        dense_len = session["dense_lens"][layer_idx]
        total_len = comp_len + dense_len
        num_logical_blocks = (total_len + self.block_size - 1) // self.block_size
        return [object() for _ in range(num_logical_blocks)]

    def rollback_session(self, session_id: str, target_len: int, clear_srl: bool = False):
        session = self.sessions.get(session_id)
        if session is None:
            return
            
        for layer_idx in range(self.num_layers):
            num_blocks = session["num_blocks"][layer_idx]
            comp_len = num_blocks * self.block_size
            
            if target_len <= comp_len:
                # Discard compressed blocks from the end
                keep_blocks = target_len // self.block_size
                session["num_blocks"][layer_idx] = keep_blocks
                # Remaining tokens in dense window
                dense_len = target_len - (keep_blocks * self.block_size)
                session["dense_lens"][layer_idx] = dense_len
                
                # Zero out discarded dense tokens and blocks
                session["dense_keys"][layer_idx][0, :, dense_len:] = 0.0
                session["dense_values"][layer_idx][0, :, dense_len:] = 0.0
                session["comp_U"][layer_idx][keep_blocks:] = 0.0
                session["comp_VK"][layer_idx][keep_blocks:] = 0.0
                session["comp_VV"][layer_idx][keep_blocks:] = 0.0
                session["comp_anc_k"][layer_idx][keep_blocks:] = 0.0
                session["comp_anc_v"][layer_idx][keep_blocks:] = 0.0
                session["comp_scale"][layer_idx][keep_blocks:] = 0.0
                session["comp_seq_len"][layer_idx][keep_blocks:] = 0
            else:
                # Only slice dense window
                dense_len = target_len - comp_len
                session["dense_lens"][layer_idx] = dense_len
                session["dense_keys"][layer_idx][0, :, dense_len:] = 0.0
                session["dense_values"][layer_idx][0, :, dense_len:] = 0.0
            
            mx.eval(
                session["dense_keys"][layer_idx],
                session["dense_values"][layer_idx],
                session["comp_U"][layer_idx],
                session["comp_VK"][layer_idx],
                session["comp_VV"][layer_idx],
                session["comp_anc_k"][layer_idx],
                session["comp_anc_v"][layer_idx]
            )
                
        if "token_ids" in session and session["token_ids"]:
            session["token_ids"] = session["token_ids"][:target_len]

    def clone_session(self, src_sid: str, dst_sid: str):
        if src_sid not in self.sessions:
            return
        src = self.sessions[src_sid]
        self.sessions[dst_sid] = {
            "dense_keys": [mx.array(k) for k in src["dense_keys"]],
            "dense_values": [mx.array(v) for v in src["dense_values"]],
            "dense_lens": src["dense_lens"].copy(),
            "num_blocks": src["num_blocks"].copy(),
            "comp_U": [mx.array(u) for u in src["comp_U"]],
            "comp_VK": [mx.array(vk) for vk in src["comp_VK"]],
            "comp_VV": [mx.array(vv) for vv in src["comp_VV"]],
            "comp_anc_k": [mx.array(ak) for ak in src["comp_anc_k"]],
            "comp_anc_v": [mx.array(av) for av in src["comp_anc_v"]],
            "comp_scale": [mx.array(s) for s in src["comp_scale"]],
            "comp_seq_len": [mx.array(sl) for sl in src["comp_seq_len"]],
            "token_ids": src["token_ids"].copy() if "token_ids" in src else []
        }

    def get_session_sequence_length(self, session_id: str) -> int:
        session = self.sessions.get(session_id)
        if session is None:
            return 0
        comp_len = session["num_blocks"][0] * self.block_size
        dense_len = session["dense_lens"][0]
        return comp_len + dense_len

    def register_prefill_tokens(self, session_id: str, token_ids: torch.Tensor):
        session = self.sessions.setdefault(session_id, self._create_empty_session())
        session["token_ids"].extend(token_ids.cpu().tolist())

    def finalize_compressed_blocks(self):
        pass

    def finalize_srl_index(self, session_id: str, cached_len: int = 0):
        pass

    def compress_deferred_prefill_blocks(self, session_id: str):
        session = self.sessions.get(session_id)
        if session is None:
            return
        for layer_idx in range(self.num_layers):
            self._compress_eligible_blocks(session_id, layer_idx)

    def capture_prefill_kv(self, session_id: str, layer_idx: int, K: mx.array, V: mx.array):
        """Write incoming prefill KV chunk into the dense buffer.
        
        Incoming K/V are shaped [1, kv_heads, L, head_dim].
        We write token-by-token into the dense window, flushing full blocks
        inline whenever the buffer would overflow.  This prevents the
        [broadcast_shapes] error that occurs when L_new > remaining capacity.
        """
        session = self.sessions.setdefault(session_id, self._create_empty_session())
        # K: [1, kv_heads, L, head_dim] → iterate over L dimension
        L_new = K.shape[2]
        k_squeezed = K.squeeze(0)  # [kv_heads, L, head_dim]
        v_squeezed = V.squeeze(0)  # [kv_heads, L, head_dim]

        for t in range(L_new):
            dense_len = session["dense_lens"][layer_idx]

            # If the dense buffer is full, compress the oldest block out
            if dense_len >= self.max_dense_len:
                self._flush_oldest_block(session, layer_idx)
                dense_len = session["dense_lens"][layer_idx]

            session["dense_keys"][layer_idx][0, :, dense_len:dense_len + 1] = (
                mx.expand_dims(k_squeezed[:, t, :], 1)
            )
            session["dense_values"][layer_idx][0, :, dense_len:dense_len + 1] = (
                mx.expand_dims(v_squeezed[:, t, :], 1)
            )
            session["dense_lens"][layer_idx] += 1

    def compress_prefill_kv(self, session_id: str):
        pass

    def ingest_streaming(self, session_id: str, layer_idx: int, k: mx.array, v: mx.array):
        session = self.sessions.setdefault(session_id, self._create_empty_session())
        dense_len = session["dense_lens"][layer_idx]
        
        session["dense_keys"][layer_idx][0, :, dense_len:dense_len + 1] = k.squeeze(0)
        session["dense_values"][layer_idx][0, :, dense_len:dense_len + 1] = v.squeeze(0)
        session["dense_lens"][layer_idx] += 1
        
        self._compress_eligible_blocks(session_id, layer_idx)
        # mx.eval(session["dense_keys"][layer_idx], session["dense_values"][layer_idx])

    def _compress_block(self, session: Dict, layer_idx: int, start: int):
        """Compress a single block starting at `start` in the dense buffer
        and store its low-rank representation in the compressed arrays."""
        num_blocks = session["num_blocks"][layer_idx]
        if num_blocks >= self.max_blocks:
            # Safety: drop oldest compressed block by shifting (rare)
            session["comp_U"][layer_idx][:-1]     = session["comp_U"][layer_idx][1:]
            session["comp_VK"][layer_idx][:-1]    = session["comp_VK"][layer_idx][1:]
            session["comp_VV"][layer_idx][:-1]    = session["comp_VV"][layer_idx][1:]
            session["comp_anc_k"][layer_idx][:-1] = session["comp_anc_k"][layer_idx][1:]
            session["comp_anc_v"][layer_idx][:-1] = session["comp_anc_v"][layer_idx][1:]
            num_blocks = self.max_blocks - 1

        block_k = session["dense_keys"][layer_idx][0, :, start:start + self.block_size]
        block_v = session["dense_values"][layer_idx][0, :, start:start + self.block_size]

        anchor_k = block_k[:, 0, :]
        anchor_v = block_v[:, 0, :]

        deltas_k = block_k[:, 1:, :] - mx.expand_dims(anchor_k, 1)
        deltas_v = block_v[:, 1:, :] - mx.expand_dims(anchor_v, 1)

        S_comp = self.block_size - 1
        deltas_k_2d = deltas_k.transpose(1, 0, 2).reshape(S_comp, -1)
        deltas_v_2d = deltas_v.transpose(1, 0, 2).reshape(S_comp, -1)
        deltas_2d = mx.concatenate([deltas_k_2d, deltas_v_2d], axis=1)

        token_norms = mx.linalg.norm(deltas_2d, axis=-1, keepdims=True)
        token_norms = mx.maximum(token_norms, 1e-5)
        deltas_normalized = deltas_2d / token_norms

        U_k, Vh_k, scale, k_rank = compress_mlx_block(deltas_normalized, self.rank)
        # token_norms is an MLX array; U_k is now a NumPy-backed mx.array.
        # Materialise token_norms before the multiply to keep the graph small.
        mx.eval(token_norms)
        U_k = U_k * token_norms

        U_padded  = mx.pad(U_k,  [(0, 0), (0, self.rank - k_rank)])
        Vh_padded = mx.pad(Vh_k, [(0, self.rank - k_rank), (0, 0)])

        VK_flat = Vh_padded[:, :self.kv_heads * self.head_dim]
        VV_flat = Vh_padded[:, self.kv_heads * self.head_dim:]

        VK = VK_flat.reshape(self.rank, self.kv_heads, self.head_dim).transpose(1, 0, 2)
        VV = VV_flat.reshape(self.rank, self.kv_heads, self.head_dim).transpose(1, 0, 2)

        session["comp_U"][layer_idx][num_blocks]     = U_padded
        session["comp_VK"][layer_idx][num_blocks]    = VK
        session["comp_VV"][layer_idx][num_blocks]    = VV
        session["comp_anc_k"][layer_idx][num_blocks] = anchor_k
        session["comp_anc_v"][layer_idx][num_blocks] = anchor_v
        session["comp_scale"][layer_idx][num_blocks]   = 1.0
        session["comp_seq_len"][layer_idx][num_blocks] = self.block_size
        session["num_blocks"][layer_idx] = num_blocks + 1
        mx.eval(
            session["comp_U"][layer_idx],
            session["comp_VK"][layer_idx],
            session["comp_VV"][layer_idx],
            session["comp_anc_k"][layer_idx],
            session["comp_anc_v"][layer_idx],
        )

    def _flush_oldest_block(self, session: Dict, layer_idx: int):
        """Compress the oldest block_size tokens from the dense buffer and
        shift the remaining tokens to the front."""
        dense_len = session["dense_lens"][layer_idx]
        self._compress_block(session, layer_idx, start=0)
        # Shift remaining tokens left by block_size
        remaining = dense_len - self.block_size
        if remaining > 0:
            session["dense_keys"][layer_idx][0, :, :remaining]   = session["dense_keys"][layer_idx][0, :, self.block_size:dense_len]
            session["dense_values"][layer_idx][0, :, :remaining] = session["dense_values"][layer_idx][0, :, self.block_size:dense_len]
        session["dense_keys"][layer_idx][0, :, remaining:dense_len]   = 0.0
        session["dense_values"][layer_idx][0, :, remaining:dense_len] = 0.0
        session["dense_lens"][layer_idx] = remaining
        # Materialise all pending ops so the lazy graph does not grow unbounded
        mx.eval(
            session["dense_keys"][layer_idx],
            session["dense_values"][layer_idx],
        )
        mx.clear_cache()

    def _compress_eligible_blocks(self, session_id: str, layer_idx: int):
        """Called after each decode step — flush blocks until dense window fits."""
        session = self.sessions[session_id]
        while session["dense_lens"][layer_idx] >= self.recency_window + self.block_size:
            self._flush_oldest_block(session, layer_idx)

    def execute_decode_attention(self, session_id: str, layer_idx: int, q_rot: mx.array, rope: Any, scale: float, num_key_value_groups: int) -> mx.array:
        session = self.sessions[session_id]
        
        q = q_rot.squeeze(2).squeeze(0)
        gpk = num_key_value_groups
        
        comp_U = session["comp_U"][layer_idx]
        comp_VK = session["comp_VK"][layer_idx]
        comp_VV = session["comp_VV"][layer_idx]
        comp_anc_k = session["comp_anc_k"][layer_idx]
        comp_anc_v = session["comp_anc_v"][layer_idx]
        comp_scale = session["comp_scale"][layer_idx]
        comp_seq_len = session["comp_seq_len"][layer_idx]
        
        num_blocks = mx.array(session["num_blocks"][layer_idx])
        
        dense_k = session["dense_keys"][layer_idx][0]
        dense_v = session["dense_values"][layer_idx][0]
        dense_len = mx.array(session["dense_lens"][layer_idx])
        
        out_combined = compute_decode_attention_static(
            q, comp_U, comp_VK, comp_VV, comp_anc_k, comp_anc_v,
            comp_scale, comp_seq_len, num_blocks,
            dense_k, dense_v, dense_len,
            scale, gpk, self.kv_heads, self.block_size, self.rank,
            self.max_blocks, self.max_dense_len
        )
        
        return mx.expand_dims(mx.expand_dims(out_combined, 0), 2)

def scaled_dot_product_attention_mlx_basic(q: mx.array, k: mx.array, v: mx.array, scale: float, mask: Optional[Any] = None) -> mx.array:
    gpk = q.shape[1] // k.shape[1]
    if gpk > 1:
        k = mx.repeat(k, gpk, axis=1)
        v = mx.repeat(v, gpk, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            L = q.shape[2]
            r = mx.arange(L)[:, None]
            c = mx.arange(L)[None, :]
            mask_arr = mx.where(r >= c, 0.0, -float("inf"))
            scores = scores + mask_arr
        else:
            scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ v

def attention_forward(self, x: mx.array, mask: Optional[Any] = None, cache: Optional[Any] = None) -> mx.array:
    """Patched Qwen2 attention that:
    - During PREFILL: uses the native MLX KV cache (via `cache`) so that
      every chunk attends correctly over all preceding tokens.
      Also captures the K/V into DiffKV dense store for later decode use.
    - During DECODE (L==1): bypasses native cache entirely and uses our
      DiffKV compressed+dense attention.
    """
    if not hasattr(self, "kv_manager"):
        return self.original_call(x, mask, cache)

    B, L, D = x.shape
    manager = self.kv_manager
    layer_idx = self.layer_idx

    session_ids = manager.active_session_ids
    position_ids = manager.position_ids

    queries = self.q_proj(x)
    keys    = self.k_proj(x)
    values  = self.v_proj(x)

    queries = queries.reshape(B, L, self.n_heads,    -1).transpose(0, 2, 1, 3)
    keys    = keys.reshape(   B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values  = values.reshape( B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

    queries_rot_list = []
    keys_rot_list    = []
    for b_idx in range(B):
        offset = mx.array(position_ids[b_idx, 0]) if position_ids is not None else mx.array(0)
        queries_rot_list.append(self.rope(queries[b_idx:b_idx+1], offset=offset))
        keys_rot_list.append(  self.rope(keys[   b_idx:b_idx+1], offset=offset))

    queries_rot = mx.concatenate(queries_rot_list, axis=0)  # [B, H_q, L, D]
    keys_rot    = mx.concatenate(keys_rot_list,    axis=0)  # [B, H_kv, L, D]

    is_decode = (L == 1)

    if is_decode:
        # ── DECODE PATH ──
        # Use the native MLX cache for correct attention over the full context.
        # The DiffKV store still ingests the token for future compressed use,
        # but the actual attention output comes from the native cache which is
        # always numerically correct.
        if cache is not None:
            all_k, all_v = cache.update_and_fetch(keys_rot, values)
        else:
            all_k, all_v = keys_rot, values

        out_b = mx.fast.scaled_dot_product_attention(
            queries_rot,
            all_k,
            all_v,
            scale=self.scale,
            mask=mask
        )

        # Still ingest into DiffKV store (architecture intact)
        for b_idx in range(B):
            sid = session_ids[b_idx]
            if sid != "dummy_session":
                manager.ingest_streaming(
                    sid, layer_idx,
                    keys_rot[b_idx:b_idx+1],
                    values[ b_idx:b_idx+1]
                )

        output = out_b.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    else:
        # ── PREFILL PATH ── use native MLX cache for correct causal attention,
        #                   then also capture K/V into DiffKV store.
        #
        # The original qwen2.py does:
        #   keys, values = cache.update_and_fetch(keys, values)
        #   output = scaled_dot_product_attention(queries, keys, values, ...)
        #
        # update_and_fetch accumulates all past KV into the cache and returns
        # the full [1, kv_heads, total_seq_len, head_dim] tensor, so every
        # chunk attends over ALL previous tokens correctly.
        if cache is not None:
            all_k, all_v = cache.update_and_fetch(keys_rot, values)
        else:
            all_k, all_v = keys_rot, values

        out_b = mx.fast.scaled_dot_product_attention(
            queries_rot,
            all_k,
            all_v,
            scale=self.scale,
            mask=mask
        )

        # 2. Capture ONLY the current chunk's K/V into DiffKV store
        #    (all_k/all_v grow with every chunk; we store incrementally)
        for b_idx in range(B):
            sid = session_ids[b_idx]
            if sid == "dummy_session":
                continue
            manager.capture_prefill_kv(
                sid, layer_idx,
                keys_rot[b_idx:b_idx+1],
                values[ b_idx:b_idx+1]
            )

        output = out_b.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

class MLXQwenModel:
    def __init__(self, mlx_model, manager):
        self.mlx_model = mlx_model
        self.manager = manager
        self._diffkv_session_ids = ["default"]
        # Per-session KVCache lists kept alive across prefill+decode.
        self._prefill_caches: dict = {}
        # Tracks whether the previous call was a prefill, per cache_key.
        # Used to fire mx.clear_cache() exactly once at the prefill→decode boundary.
        self._prev_was_prefill: dict = {}

    def _get_or_create_prefill_cache(self, cache_key: tuple):
        if cache_key not in self._prefill_caches:
            from mlx_lm.models.cache import make_prompt_cache
            self._prefill_caches[cache_key] = make_prompt_cache(self.mlx_model)
        return self._prefill_caches[cache_key]

    def __call__(self, input_ids: torch.Tensor, position_ids: torch.Tensor, use_cache: bool = True):
        inputs_np = input_ids.detach().cpu().numpy()
        inputs_mx = mx.array(inputs_np)

        self.manager.active_session_ids = self._diffkv_session_ids
        self.manager.position_ids = position_ids.detach().cpu().numpy() if position_ids is not None else None

        is_prefill = (input_ids.shape[1] > 1)
        cache_key = tuple(self._diffkv_session_ids)

        if is_prefill:
            # Pass the accumulated prefill cache so each chunk attends over
            # ALL previous tokens — this gives correct causal hidden states.
            prefill_cache = self._get_or_create_prefill_cache(cache_key)
            logits_mx = self.mlx_model(inputs_mx, cache=prefill_cache)
        else:
            # ── Prefill → Decode transition ──────────────────────────────────
            # MLX's allocator holds onto the peak GQA-expanded K/V tensors from
            # the final prefill chunk (e.g. [1, 12, 8192, 128] × 28 layers × 2
            # ≈ 1.4 GB). These are no longer needed once decode begins.
            # mx.clear_cache() releases them back to the OS immediately.
            if self._prev_was_prefill.get(cache_key, True):
                mx.eval()          # flush any pending lazy ops first
                mx.clear_cache()   # return peak activation memory to OS
                import gc; gc.collect()

            # Decode: keep the same cache alive so decode tokens attend over
            # the full prefill context + all previously decoded tokens.
            decode_cache = self._prefill_caches.get(cache_key)
            logits_mx = self.mlx_model(inputs_mx, cache=decode_cache)

        self._prev_was_prefill[cache_key] = is_prefill

        mx.eval(logits_mx)

        logits_np = np.array(logits_mx.astype(mx.float32))
        logits_py = torch.from_numpy(logits_np).to(device=input_ids.device)

        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
                self.past_key_values = None

        return ModelOutput(logits_py)



class MLXDiffKVWrapper:
    def __init__(
        self, 
        model_id: str,
        config: Dict[str, Any],
        device: str = None,
        quantization_config: Any = None,
        torch_dtype: Any = None,
        lazy: bool = False,
    ):
        self.model_id = model_id
        self.config = config or {}
        self.lazy = lazy
        self.is_mlx = True
        
        self.block_size = self.config.get("block_size", 256)
        self.rank = self.config.get("rank", 16)
        self.micro_block_size = self.config.get("micro_block_size", 256)
        self.device = "mps"
        
        self.tokenizer = None
        self.stop_token_ids = set()
        self.model = None
        self.manager = None
        self.active_session = None
        self._session_token_ids = {}

        if not self.lazy:
            self.ensure_loaded()

    def ensure_loaded(self):
        if self.model is not None:
            return

        model_id = self.model_id
        quant = self.config.get("quantization")
        
        preset = self.config.get("preset", os.environ.get("DIFFKV_PRESET", "mid")).lower()
        if preset == "low" and not quant and not os.environ.get("DIFFKV_QUANTIZATION"):
            quant = "int4"
            print("[DiffKV MLX] Low preset: auto-enabling 4-bit quantization")

        if quant in ("int4", "int8") and not model_id.startswith("mlx-community/"):
            parts = model_id.split("/")
            if len(parts) == 2:
                org, name = parts
                suffix = "4bit" if quant == "int4" else "8bit"
                model_id = f"mlx-community/{name}-{suffix}"
                print(f"[DiffKV MLX] Loading quantized model: {model_id}")

        print(f"[DiffKV MLX] Loading model via mlx_lm: {model_id}...")
        t0 = time.time()
        model, tokenizer = mlx_load(model_id)
        print(f"[DiffKV MLX] Loaded model in {time.time() - t0:.2f}s")
        
        self.tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        
        self.stop_token_ids = set()
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            self.stop_token_ids.update(eos_id)
        elif isinstance(eos_id, int):
            self.stop_token_ids.add(eos_id)
            
        special_words = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]
        for word in special_words:
            tok_id = self.tokenizer.convert_tokens_to_ids(word)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tok_id)

        self.manager = MLXKVBlockManager(
            num_layers=len(model.layers),
            heads=model.model.layers[0].self_attn.n_heads,
            kv_heads=model.model.layers[0].self_attn.n_kv_heads,
            head_dim=model.model.layers[0].self_attn.q_proj.weight.shape[0] // model.model.layers[0].self_attn.n_heads,
            rank=self.rank,
            block_size=self.block_size
        )
        self._session_token_ids = self.manager._session_token_ids
        
        self._patch_attention_layers(model)
        self.model = MLXQwenModel(model, self.manager)

    def _patch_attention_layers(self, model):
        from mlx_lm.models import qwen2
        if not hasattr(qwen2.Attention, "original_call"):
            qwen2.Attention.original_call = qwen2.Attention.__call__
        qwen2.Attention.__call__ = attention_forward
        
        for layer_idx, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = layer_idx
            layer.self_attn.kv_manager = self.manager

    def close(self):
        if self.manager is not None:
            self.manager.sessions.clear()
            self.manager = None
        self.model = None
        self.tokenizer = None
        mx.metal.clear_cache()

    def stop(self):
        self.close()

    def switch_session(self, session_id: str):
        self.active_session = session_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        self.ensure_loaded()
        session_id = self.active_session or "default"
        
        # Squeeze prompt tokenization
        prompt_ids = self.tokenizer.encode(prompt)
        
        # Check cache reuse
        cached_len = 0
        if session_id in self._session_token_ids:
            stored_ids = self._session_token_ids[session_id]
            if len(stored_ids) > 0 and len(stored_ids) < len(prompt_ids):
                if prompt_ids[:len(stored_ids)] == stored_ids:
                    cached_len = len(stored_ids)
                    print(f"[DiffKV MLX Wrapper] Reusing {cached_len} cached tokens!")
                    
        if cached_len == 0:
            self.manager.clear_session(session_id)
            self._session_token_ids[session_id] = []
            new_prompt_ids = prompt_ids
        else:
            new_prompt_ids = prompt_ids[cached_len:]

        self.manager.init_session(session_id, prefill_len=cached_len + len(new_prompt_ids))
        self.manager.register_prefill_tokens(session_id, torch.tensor(new_prompt_ids, dtype=torch.long))
        self.model._diffkv_session_ids = [session_id]

        # ── Chunked Prefill ──
        PREFILL_CHUNK = 512
        output = None
        for chunk_start in range(0, len(new_prompt_ids), PREFILL_CHUNK):
            chunk = new_prompt_ids[chunk_start:chunk_start + PREFILL_CHUNK]
            clen = len(chunk)
            abs_start = cached_len + chunk_start
            
            chunk_tensor = torch.tensor([chunk], dtype=torch.long)
            pos_tensor = torch.tensor([list(range(abs_start, abs_start + clen))], dtype=torch.long)
            
            output = self.model(chunk_tensor, pos_tensor)
            self.manager.compress_deferred_prefill_blocks(session_id)
            
        # Complete sequence prefill done
        generated = prompt_ids.copy()
        
        # ── Decoding loop ──
        cur_pos = cached_len + len(new_prompt_ids)
        logits = output.logits[0, -1].cpu().numpy()
        
        # Helper sampling
        def sample_logits(logits, temp, top_p):
            if temp <= 0.01:
                return int(np.argmax(logits))
            scaled = logits / temp
            # Softmax
            exp_logits = np.exp(scaled - np.max(scaled))
            probs = exp_logits / np.sum(exp_logits)
            if top_p < 1.0:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cum_probs = np.cumsum(sorted_probs)
                cutoff = np.where(cum_probs > top_p)[0]
                if len(cutoff) > 0:
                    probs[sorted_indices[cutoff[0]+1:]] = 0.0
                    probs = probs / np.sum(probs)
            return int(np.random.choice(len(probs), p=probs))

        for _ in range(max_new_tokens):
            # ── Repetition-loop detection (mirrors batch_engine.py Fix 2) ──────
            # Detect tight token-level loops every 10 new tokens.
            # On detection, widen the penalty window and boost the strength.
            # After 40 tokens without recovery, force-stop generation.
            _new_tokens = generated[len(prompt_ids):]  # tokens produced in this call
            _n_new = len(_new_tokens)
            _loop_detected = getattr(self, "_mlx_loop_detected", False)
            _loop_idx = getattr(self, "_mlx_loop_idx", None)

            if not _loop_detected and _n_new >= 30 and _n_new % 10 == 0:
                _window = _new_tokens[-80:]
                _ng = 5
                if len(_window) >= _ng + 1:
                    _ngrams = [tuple(_window[i:i + _ng]) for i in range(len(_window) - _ng + 1)]
                    _top = Counter(_ngrams).most_common(1)[0][1]
                    if _top / len(_ngrams) >= 0.35:
                        _loop_detected = True
                        self._mlx_loop_detected = True
                        self._mlx_loop_idx = _n_new
                        print(
                            f"[DiffKV MLX] WARNING: repetition loop detected at token "
                            f"{_n_new}. Escalating penalty window to 256 tokens and strength to 1.3x."
                        )

            if _loop_detected:
                if _loop_idx is None:
                    self._mlx_loop_idx = _n_new
                elif _n_new - _loop_idx >= 40:
                    print(
                        "[DiffKV MLX] WARNING: repetition loop persisted for 40 tokens "
                        "after detection \u2014 forcing EOS."
                    )
                    break

            # Repetition penalty (widened window when a loop is active)
            _pen_window = 256 if _loop_detected else 64
            _pen_val = max(repetition_penalty, 1.3) if _loop_detected else repetition_penalty
            if _pen_val != 1.0:
                for tok_id in set(generated[-_pen_window:]):
                    if logits[tok_id] > 0:
                        logits[tok_id] /= _pen_val
                    else:
                        logits[tok_id] *= _pen_val
                        
            next_id = sample_logits(logits, temperature, top_p)
            generated.append(next_id)
            self.manager.register_prefill_tokens(session_id, torch.tensor([next_id], dtype=torch.long))
            
            if next_id in self.stop_token_ids:
                break
                
            input_ids = torch.tensor([[next_id]], dtype=torch.long)
            pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long)
            
            output = self.model(input_ids, pos_tensor)
            logits = output.logits[0, -1].cpu().numpy()
            
            cur_pos += 1

        # Clear loop detection state for this session after generation completes
        self._mlx_loop_detected = False
        self._mlx_loop_idx = None
        self._session_token_ids[session_id] = generated
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return _normalize_references(decoded)
