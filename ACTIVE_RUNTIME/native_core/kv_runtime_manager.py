"""
native_core/kv_runtime_manager.py

Minimal Runtime — KV Session Manager.

Manages KV cache residency and on-demand reconstruction for Differential KV.

  - PagedKVStore: GPU -> CPU RAM spillover under memory pressure.
  - ReconstructionCache: LRU cache of recently reconstructed dense blocks.
  - AsyncCompressor: SVD compression moved off the decode hot path.
  - StreamingSparseIngestManager: sparse-first prefill (no dense-first allocation).
  - NativeBlockPool: contiguous GPU memory pool for Triton kernel dispatch.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import sys, os
import threading

from native_core.sparse_decode.triton_fused_decode import TritonDiffKV
from native_core.compression.lowrank import compress_lowrank, LowRankDelta
from native_core.paging.paged_kv_store import PagedKVStore
from native_core.compression.async_compressor import AsyncCompressor

try:
    from native_core.mac_utils import (
        get_best_device as _get_best_device,
        empty_cache as _empty_cache,
        synchronize as _synchronize,
        has_cuda as _has_cuda,
    )
except ImportError:
    def _get_best_device(): return "cuda" if torch.cuda.is_available() else "cpu"
    def _empty_cache(device=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    def _synchronize(device=None):
        if torch.cuda.is_available(): torch.cuda.synchronize()
    def _has_cuda(): return torch.cuda.is_available()


def get_layer_rank(
    layer_idx: int,
    num_layers: int,
    base_rank: int,
    early_boost: bool = False,
    max_rank_early: int = 0,
) -> int:
    """
    Per-layer adaptive rank schedule tuned for Qwen 2.5 1.5B (28 layers).
    Early layers have broader activation distributions and benefit from higher rank;
    final layers are more concentrated and can use lower rank.

    Schedule is PROPORTIONAL to base_rank so the user's configured rank acts
    as the standard ceiling (no silent VRAM inflation beyond --rank for normal use).

    Normal schedule (early_boost=False, default):
      Layers 0-15%:  base_rank
      Layers 15-50%: base_rank  (no change)
      Layers 50-79%: max(4, round(0.75 * base_rank))
      Layers 79%+:   max(4, round(0.50 * base_rank))

    Boosted schedule (early_boost=True):
      Layers 0-15%:  min(2 * base_rank, max_rank_early or 2 * base_rank)
                     Allows early syntactic layers to retain more KV fidelity.
                     Enable via DIFFKV_EARLY_LAYER_RANK_BOOST=1 or config dict.
      Layers 15%+:   Same as normal schedule.

    Parameters
    ----------
    layer_idx     : index of the current transformer layer (0-indexed)
    num_layers    : total number of layers in the model
    base_rank     : user-configured SVD rank (acts as standard ceiling)
    early_boost   : if True, boost rank for layers 0-15% (default False)
    max_rank_early: hard cap for boosted early-layer rank; 0 = auto (2 * base_rank)
    """
    ratio = layer_idx / max(num_layers, 1)
    if ratio < 0.15:       # layers 0-4 for 28-layer model
        if early_boost:
            # Boost early layers up to 2× base_rank, optionally capped by max_rank_early
            cap = max_rank_early if max_rank_early > 0 else (2 * base_rank)
            return min(2 * base_rank, cap)
        return base_rank
    elif ratio < 0.50:     # layers 4-14
        return base_rank
    elif ratio < 0.79:     # layers 14-22 — slightly reduced
        return max(4, round(0.75 * base_rank))
    else:                  # layers 22-28 — concentrated final layers
        return max(4, round(0.50 * base_rank))



# ─────────────────────────────────────────────────────────────────────────────
# KVBlock definition (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class KVBlock:
    """Physically stores compressed KV memory for one block of tokens."""
    anchor_idx: int
    anchor_kv:  torch.Tensor          # [1, 2, heads, head_dim]
    U:          Optional[torch.Tensor] = None   # [block_size, rank]
    V:          Optional[torch.Tensor] = None   # [rank, feat_dim]
    scale:      float = 1.0
    token_indices: List[int] = None
    cosine_sim: float = 1.0
    norm_drift: float = 0.0
    dynamic_rank: int = -1

    # Optional uncompressed tokens (dense window)
    active_k: Optional[torch.Tensor] = None
    active_v: Optional[torch.Tensor] = None
    pool_idx: Optional[int] = None
    dirty:    bool = True
    _cache_id: Optional[str] = None
    _lock:    threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


# Helper class for thread-safe block snapshots
class BlockSnapshot:
    __slots__ = ('b', 'anchor_kv', 'U', 'V', 'scale', 'active_k', 'active_v', 'pool_idx', 'dirty', 'dynamic_rank', 'active_k_cpu', 'active_v_cpu')

    def __init__(self, block):
        self.b = block
        lock = getattr(block, "_lock", None)
        if lock is not None:
            with lock:
                self.anchor_kv = block.anchor_kv
                self.U = block.U
                self.V = block.V
                self.scale = getattr(block, "scale", 1.0)
                self.active_k = block.active_k
                self.active_v = block.active_v
                self.pool_idx = getattr(block, "pool_idx", None)
                self.dirty = getattr(block, "dirty", True)
                self.dynamic_rank = getattr(block, "dynamic_rank", -1)
                self.active_k_cpu = getattr(block, "active_k_cpu", None)
                self.active_v_cpu = getattr(block, "active_v_cpu", None)
        else:
            self.anchor_kv = block.anchor_kv
            self.U = block.U
            self.V = block.V
            self.scale = getattr(block, "scale", 1.0)
            self.active_k = block.active_k
            self.active_v = block.active_v
            self.pool_idx = getattr(block, "pool_idx", None)
            self.dirty = getattr(block, "dirty", True)
            self.dynamic_rank = getattr(block, "dynamic_rank", -1)
            self.active_k_cpu = getattr(block, "active_k_cpu", None)
            self.active_v_cpu = getattr(block, "active_v_cpu", None)


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class KVRuntimeManager:
    """
    Manages KV cache residency and on-demand reconstruction.

    Phase 7:
      - Wraps PagedKVStore for GPU/CPU tiered residency.
      - Uses ReconstructionCache to skip repeat U@V GEMMs.
      - Submits compression jobs to AsyncCompressor.
    """

    def __init__(
        self,
        num_layers:          int,
        heads:               int,
        head_dim:            int,
        device:              str   = None,  # None → auto-detect (CUDA / MPS / CPU)
        gpu_budget_gb:       float = 2.0,
        recon_cache_size:    int   = 64,
        async_compression:   bool  = True,
        streaming_ingest:    bool  = True,
        micro_block_size:    int   = 256,   # S=256, R=32 → 5.2× compression ratio
        rank:                int   = 8,
        kv_heads:            int   = None,
        serving_mode:        str   = "balanced",
        tokenizer                  = None,   # HuggingFace tokenizer (for SRL stop words)
        config:              dict  = None,
    ):
        from native_core.config import DiffKVConfig
        self.config      = DiffKVConfig(config)
        self.num_layers  = num_layers
        self.heads       = heads
        self.kv_heads    = kv_heads if kv_heads is not None else heads
        self.head_dim    = head_dim
        self.device      = device if device is not None else _get_best_device()
        self.feat_dim    = 2 * self.kv_heads * head_dim

        # ── SRL: tokenizer + precomputed stop token IDs ────────────────────
        self.tokenizer = tokenizer
        self._stop_token_ids: set = set()
        if tokenizer is not None:
            _STOP_WORDS = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "have", "has", "had", "do", "does", "did", "will", "would",
                "could", "should", "may", "might", "shall", "can", "need",
                "to", "of", "in", "on", "at", "by", "for", "with", "as",
                "and", "or", "but", "if", "then", "that", "this", "it",
                "he", "she", "they", "we", "you", "i", "not", "no",
                ",", ".", ":", ";", "?", "!", "(", ")", "'", '"', "-", "\n",
                "system", "user", "assistant", "im_start", "im_end",
            }
            for word in _STOP_WORDS:
                try:
                    ids = tokenizer.encode(word, add_special_tokens=False)
                    self._stop_token_ids.update(ids)
                except Exception:
                    pass
            if hasattr(tokenizer, "all_special_ids"):
                self._stop_token_ids.update(tokenizer.all_special_ids)
        else:
            # Fallback: low-ID BPE tokens are overwhelmingly punctuation/particles
            self._stop_token_ids = set(range(200))

        # Per-session token ID registry (CPU tensors, for inverted index build)
        self._session_token_ids: dict = {}

        # Attention score cache for decode steps
        from native_core.srl.attention_cache import AttentionScoreCache
        self.attention_score_cache = AttentionScoreCache()

        # Per-session SRL state (populated by finalize_srl_index)
        self._session_srl: dict = {}

        # Per-session SRL custom configuration settings
        self.session_configs: dict = {}

        # session_id -> layer_idx -> List[KVBlock]
        self.session_blocks: Dict[str, Dict[int, List[KVBlock]]] = {}

        self.block_size           = 64
        self.rank                 = rank  # fixed, set at construction
        self.dense_recency_blocks = 1
        self.streaming_ingest     = streaming_ingest
        self.micro_block_size     = micro_block_size
        self.serving_mode         = serving_mode

        # Dynamically set budgets and pool/cache counts based on serving mode
        if serving_mode == "lightweight":
            recon_cache_size = 16
            recon_pool_blocks = 32
            gpu_budget_gb = 0.5
        elif serving_mode == "performance":
            recon_cache_size = 256
            recon_pool_blocks = 512
            gpu_budget_gb = 2.0
        elif serving_mode == "long-context":
            recon_cache_size = 512
            recon_pool_blocks = 1024
            gpu_budget_gb = 2.0
        elif serving_mode == "fused-sparse":
            recon_cache_size = 256
            recon_pool_blocks = 512
            gpu_budget_gb = 2.0
        else: # balanced
            recon_cache_size = 64
            recon_pool_blocks = 128 if streaming_ingest else 512
            gpu_budget_gb = 1.5

        from runtime.native_block_pool import NativeBlockPool

        # pool_rank MUST equal self.rank exactly — it determines the physical shape of
        # U [int8: pool_block_size × pool_rank] and V_K/V_V [fp16: pool_rank × kv_heads × head_dim]
        # in the NativeBlockPool. Using max(64, rank) was silently 2–4× over-allocating
        # when rank=8 or rank=16. The sparse kernel handles variable k-rank blocks via
        # min(U.shape[1], pool_rank) guards — no minimum needed here.
        pool_rank = self.rank
        # Pool max_seq_len = micro_block_size (default varies by context length).
        pool_block_size = self.micro_block_size if self.streaming_ingest else self.block_size
        # Ensure pool_block_size can hold the maximum adaptive prefill block size (MBS + 1 anchor)
        pool_block_size = max(pool_block_size, 257)
        
        bytes_per_block = (
            (pool_block_size * pool_rank * 1) +                                # U (int8)
            (pool_rank * self.kv_heads * self.head_dim * 2) * 2 +              # V_K, V_V (fp16)
            (self.kv_heads * self.head_dim * 2) * 2 +                          # anchors_K, anchors_V (fp16)
            6 + 2                                                              # scales (2) + seq_lens (4) + U_scale (2)
        )
        
        # Use the REAL mean block size (adaptive schedule mean ≈ 32 for short contexts).
        # Old formula used micro_block_size=256 as avg even when 99% of sessions use S=32.
        # This over-counted blocks_per_layer by 8× for typical chat, making the pool 8× larger.
        # Conservative estimate: average between 32 (short chat) and micro_block_size (long ctx).
        avg_block_sz = max(32, min(self.micro_block_size, 64))
        max_tokens_map = {"long-context": 32768, "performance": 16384, "balanced": 8192, "lightweight": 4096}
        expected_tokens = max_tokens_map.get(serving_mode, 8192)
        n_blocks_per_layer = max(1, expected_tokens // avg_block_sz)
        
        # Sum of compressed blocks across all layers, scaled by 6 to support multi-session serving
        total_expected_blocks = n_blocks_per_layer * self.num_layers * 6
        pool_budget_bytes = int(total_expected_blocks * bytes_per_block * 1.5)  # 50% safety headroom
        
        # Clamp pool between 128MB and 4.0GB (generous limit for multi-session serving)
        pool_budget_bytes = max(128 * 1024 ** 2, min(4 * 1024 ** 3, pool_budget_bytes))
        
        min_blocks = 2048 if self.serving_mode == "lightweight" else (4096 if self.serving_mode == "balanced" else 8000)
        dynamic_max_blocks = max(min_blocks, min(65536, pool_budget_bytes // bytes_per_block))
        
        self.native_pool = NativeBlockPool(
            max_blocks=dynamic_max_blocks,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            rank=pool_rank,
            max_seq_len=pool_block_size,
            device=self.device,
            dtype=torch.float16,
            initial_blocks=256   # Each slot is now micro_block_size (256) rows
        )
        self.native_pool.config = self.config

        # ── SRL: Initialize random projection matrix W_proj ──────────────
        # Fixed at construction time — never updated.
        # All block descriptors and query descriptors use the same W_proj,
        # making them directly comparable across the lifetime of the pool.
        _desc_dim = 64
        _W = torch.randn(_desc_dim, self.head_dim, dtype=torch.float32)
        _W = _W / (_W.norm(dim=1, keepdim=True) + 1e-8)   # normalize rows
        self.native_pool.W_proj = _W.to(self.device)


        # ── Phase 7 subsystems ────────────────────────────────────────────
        self.pager       = PagedKVStore(gpu_budget_gb=gpu_budget_gb, device=device)
        self.decode_workspace = {}

        # On Apple Silicon/MPS, we use thread-safe CPU-only background SVD compression
        # to guarantee thread-safety and prevent Metal command encoder / buffer validation crashes.
        if self.device == "mps" or (isinstance(self.device, torch.device) and self.device.type == "mps") or "mps" in str(self.device):
            print("[DiffKV] Auto-detected Apple Silicon / MPS device. Enabling thread-safe CPU background SVD compression.")

        self._async      = self.config.async_svd
        self._compressor = AsyncCompressor(compress_fn=self._compress_block_sync)
        if self._async:
            self._compressor.start()

        # ── Phase 24.5: Streaming Sparse Ingest Manager ───────────────────
        if self.streaming_ingest:
            from native_core.streaming_sparse_ingest import StreamingSparseIngestManager
            self._streaming_mgr = StreamingSparseIngestManager(
                compressor=self._compressor,
                compress_fn=self._compress_block_sync,
                micro_block_size=self.micro_block_size,
                dense_anchor_only=True,
                native_pool=self.native_pool,
            )
            self._streaming_mgr.manager = self
        else:
            self._streaming_mgr = None

        # Telemetry
        self.vram_saved_bytes   = 0
        self.total_compressions = 0
        self.total_cosine_sim   = 0.0
        self.total_norm_drift   = 0.0
        self.rank_histogram     = {}    # rank -> count

        # Phase 29 Fix #4: micro_block_size cache (session_id -> int)
        # Value is stable after prefill; invalidated on clear_session / init_session.
        self._mbs_cache: dict = {}

        # Phase 32 Fix: GPU block_indices cache to eliminate PCIe & GPU allocator churn
        self._indices_gpu_cache: dict = {}



        # Fast-path counter: number of blocks in CPU_COMPRESSED state waiting for
        # main-thread GPU upload. finalize_compressed_blocks() returns in O(1)
        # when this is zero (the normal decode steady-state after prefill completes).
        self._pending_cpu_blocks: int = 0
        self._pending_lock = threading.Lock()

    # ── Session management ────────────────────────────────────────────────────

    def init_session(self, session_id: str, prefill_len: int = 0, max_tokens_hint: int = None):
        if session_id not in self.session_blocks:
            self.session_blocks[session_id] = {i: [] for i in range(self.num_layers)}
            
        # Grow NativeBlockPool if needed for this session
        if max_tokens_hint is not None and getattr(self, "native_pool", None) is not None:
            pool = self.native_pool
            growth_factor = 1.5
            block_size = self.micro_block_size if self.streaming_ingest else self.block_size
            block_size = max(block_size, 257)
            needed_blocks = int((max_tokens_hint / block_size) * self.num_layers * growth_factor)
            if needed_blocks > pool.current_blocks:
                print(f"[DiffKV] Preemptively growing block pool from {pool.current_blocks} to {needed_blocks} for session {session_id}")
                pool._grow_pool(new_blocks=needed_blocks)

        if self._streaming_mgr is not None and session_id not in self._streaming_mgr.session_blocks:
            self._streaming_mgr.init_session(session_id, self.num_layers, prefill_len=prefill_len)
        # Invalidate mbs cache so a reused session_id gets a fresh value
        self._mbs_cache.pop(session_id, None)
        # Invalidate GPU block indices cache and decode workspace
        if hasattr(self, 'decode_workspace'):
            self.decode_workspace.pop(session_id, None)

    def rollback_session(self, session_id: str, target_len: int, clear_srl: bool = False) -> None:
        """
        Rollback/truncate a session's KV cache to a target sequence length.
        Used by speculative decoding to discard rejected candidate tokens.
        """
        if self._streaming_mgr is not None:
            self._streaming_mgr.rollback_session(session_id, target_len)

        # Rollback registered token IDs
        if session_id in self._session_token_ids:
            self._session_token_ids[session_id] = self._session_token_ids[session_id][:target_len]

        # Invalidate GPU block indices cache and sliced RoPE caches for this session.
        # Note: concatenated_K_rot and V_V_perm were removed in a prior refactor —
        # those keys no longer exist in decode_workspace.
        if hasattr(self, 'decode_workspace') and session_id in self.decode_workspace:
            session_dict = self.decode_workspace[session_id]
            session_dict.pop("gathered_kv", None)
            session_dict.pop("indices_gpu", None)
            session_dict.pop("decode_cos_sliced", None)
            session_dict.pop("decode_sin_sliced", None)

        if clear_srl:
            self._session_srl.pop(session_id, None)

    def set_session_config(self, session_id: str, config: dict):
        if not hasattr(self, "session_configs"):
            self.session_configs = {}
        if session_id not in self.session_configs:
            self.session_configs[session_id] = {}
        self.session_configs[session_id].update(config)


    def register_prefill_tokens(self, session_id: str, token_ids: torch.Tensor) -> None:
        """
        Store the full prompt token ID sequence for a session.

        Called by batch_engine.py _step() before the prefill forward pass,
        with input_ids.squeeze(0).cpu() — a [seq_len] CPU tensor.

        These token IDs are used by finalize_srl_index() to build the
        lexical inverted index. They are concatenated across prefill chunks
        so multi-chunk prefills are handled correctly.
        """
        existing = self._session_token_ids.get(session_id)
        if existing is None:
            self._session_token_ids[session_id] = token_ids.cpu()
        else:
            # Append new chunk
            self._session_token_ids[session_id] = torch.cat(
                [existing, token_ids.cpu()], dim=0
            )

    def finalize_srl_index(self, session_id: str, cached_len: int = 0) -> None:
        """
        Build all SRL routing structures for a session after prefill completes.

        Called from compress_prefill_kv() once all blocks are finalized
        (pending_cpu_blocks == 0). Also callable directly after the
        compression barrier in batch_engine.py for explicit control.

        Builds:
          - SemanticIndex  (ANN search over 64-dim descriptors)
          - ChunkGraph     (block-to-block similarity graph)
          - InvertedTokenIndex  (token_id → block list)
          - SessionSRLState  (attaches all indexes + sink blocks)
        """
        import os as _os

        pool = getattr(self, "native_pool", None)
        if pool is None or pool.W_proj is None:
            return  # SRL not available (W_proj not initialized)

        # Gather all COMPRESSED pool slot IDs for this session (from layer 0)
        blocks_layer0 = self.get_streaming_blocks(session_id, 0)
        slot_ids = [
            b.pool_idx for b in blocks_layer0
            if getattr(b, "pool_idx", None) is not None
            and getattr(b, "state", "") == "COMPRESSED"
        ]

        if not slot_ids:
            return  # No compressed blocks yet — skip

        try:
            from native_core.srl.semantic_index import build_semantic_index
            from native_core.srl.chunk_graph import build_chunk_graph
            from native_core.srl.inverted_index import build_inverted_index
            from native_core.srl.session_srl_state import SessionSRLState

            # ── 1. Semantic index ───────────────────────────────────────
            sem_index = build_semantic_index(pool, slot_ids)

            # ── 2. Chunk graph ──────────────────────────────────────────
            chunk_graph = build_chunk_graph(
                sem_index.desc_matrix,
                sem_index.slot_ids,
                K_semantic=6,
                K_temporal=2,
            )

            # ── 3. Inverted token index ─────────────────────────────────
            token_ids_cpu = self._session_token_ids.get(session_id)
            mbs = self.get_session_micro_block_size(session_id)
            # block_size for indexing = anchor (1) + active tokens (mbs)
            index_block_size = mbs + 1

            if token_ids_cpu is not None and len(slot_ids) > 0:
                inv_index = build_inverted_index(
                    token_ids      = token_ids_cpu,
                    slot_ids       = slot_ids,
                    block_size     = index_block_size,
                    stop_token_ids = self._stop_token_ids,
                    top_n_per_block = 20,
                )
            else:
                from native_core.srl.inverted_index import InvertedTokenIndex
                inv_index = InvertedTokenIndex(index={}, important_vocab=set())

            # ── 4. Sink blocks (block 0 + special token blocks) ──────────
            sink_blocks: list = []
            # Always include first slot (attention sinks / system prompt start)
            if slot_ids:
                sink_blocks.append(slot_ids[0])

            # Include blocks containing special tokens if tokenizer is available
            if self.tokenizer is not None:
                SPECIAL_WORDS = [
                    "<|system|>", "<|user|>", "<|assistant|>",
                    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
                ]
                special_ids: set = set()
                for w in SPECIAL_WORDS:
                    try:
                        tok_id = self.tokenizer.convert_tokens_to_ids(w)
                        if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                            special_ids.add(tok_id)
                    except Exception:
                        pass
                from native_core.srl.inverted_index import lookup as _inv_lookup
                from native_core.srl.inverted_index import InvertedTokenIndex as _ITI
                _tmp = _ITI(
                    index=inv_index.index,
                    important_vocab=inv_index.important_vocab | special_ids,
                )
                for sid_special in _inv_lookup(_tmp, list(special_ids)):
                    if sid_special not in sink_blocks:
                        sink_blocks.append(sid_special)

            # Get session config if any to dynamically set values on SessionSRLState
            session_config = getattr(self, "session_configs", {}).setdefault(session_id, {})
            default_k_min = int(_os.environ.get("DIFFKV_SRL_K_MIN", "20"))
            default_k_max = int(_os.environ.get("DIFFKV_SRL_K_MAX", "200"))
            default_threshold = self.config.srl_threshold

            k_min = session_config.get("srl_k_min", default_k_min)
            k_max = session_config.get("srl_k_max", default_k_max)
            routing_threshold = session_config.get("srl_threshold", default_threshold)

            # Enable SRL by default for all models
            if "srl_enabled" not in session_config:
                session_config["srl_enabled"] = True

            # Preserve dynamic state flags (like nothing_found) across rebuilds
            existing_srl = self._session_srl.get(session_id)
            nothing_found = getattr(existing_srl, "nothing_found", False)

            # Extract latest query tokens
            current_query_tokens = getattr(existing_srl, "current_query_tokens", [])
            if token_ids_cpu is not None:
                current_query_tokens = token_ids_cpu[cached_len:].tolist()

            # ── 5. Assemble SessionSRLState ──────────────────────────────
            srl_state = SessionSRLState(
                semantic_index    = sem_index,
                chunk_graph       = chunk_graph,
                inverted_index    = inv_index,
                ordered_slot_ids  = slot_ids,
                sink_blocks       = list(dict.fromkeys(sink_blocks)),
                k_min             = k_min,
                k_max             = k_max,
                routing_threshold = routing_threshold,
            )
            srl_state.nothing_found = nothing_found
            srl_state.current_query_tokens = current_query_tokens
            if hasattr(self, "_last_prefill_q") and session_id in self._last_prefill_q:
                srl_state.last_prefill_q = self._last_prefill_q[session_id]
            self._session_srl[session_id] = srl_state

            n = len(slot_ids)
            desc_kb = n * 64 * 2 / 1024
            graph_kb = n * 8 * 4 / 1024
            if _os.environ.get("DIFFKV_TELEMETRY", "0") == "1" or \
               _os.environ.get("DIFFKV_SRL_VERBOSE", "0") == "1":
                print(
                    f"[SRL] Index built: session={session_id} "
                    f"blocks={n} desc={desc_kb:.1f}KB graph={graph_kb:.1f}KB "
                    f"vocab={len(inv_index.important_vocab)} sink={len(sink_blocks)}"
                )

        except Exception as e:
            import traceback
            print(f"[SRL] WARNING: finalize_srl_index failed for session {session_id}: {e}")
            if _os.environ.get("DIFFKV_SRL_VERBOSE", "0") == "1":
                traceback.print_exc()

    def get_srl_state(self, session_id: str):
        """Return the SessionSRLState for a session, or None if not yet built."""
        return self._session_srl.get(session_id)

    # ── Prefill KV capture & batch compression ────────────────────────────────

    def capture_prefill_kv(
        self,
        session_id: str,
        layer_idx: int,
        K: torch.Tensor,   # [1, kv_heads, chunk_len, head_dim]
        V: torch.Tensor,
    ) -> None:
        """
        Immediately streams K/V through the ingest pipeline.

        Old behaviour: accumulated chunks via torch.cat into a growing GPU tensor
        across all 28 layers × 4 chunks = 112 growing cat operations per 2048-token
        prompt, then processed everything at once in compress_prefill_kv(). This
        created an O(N²) GPU allocation spike and held all chunks in VRAM simultaneously.

        New behaviour: each chunk is streamed directly into ingest_streaming() as
        soon as the forward pass for that layer returns. This:
          1. Eliminates all torch.cat accumulation (zero extra allocations).
          2. Allows compression to start immediately on the first chunk.
          3. Keeps VRAM bounded to 1 chunk at a time (not the whole prompt).
        """
        # Stream directly — ingest_streaming handles block alignment, SVD, pool writes.
        self.ingest_streaming(session_id, layer_idx, K, V)

    def compress_prefill_kv(self, session_id: str) -> None:
        """
        No-op barrier stub — compression now fires immediately in capture_prefill_kv().
        Kept for API compatibility with hf_diffkv_wrapper.py which calls this after
        each prefill chunk to trigger SVD overlap with the next chunk forward pass.

        Also triggers SRL index build once all blocks are finalized, if token IDs
        have been registered for this session via register_prefill_tokens().
        """

        import gc as _gc, os as _os
        _gc.collect()
        _empty_cache(self.device)

        # ── SRL: build or update the semantic routing index ──────────────
        # Deferred until ALL blocks are finalized (pending_cpu_blocks == 0).
        # This is a no-op for intermediate chunks; fires on the last chunk
        # when the compression barrier is already drained.
        if session_id in self._session_token_ids:
            pending = getattr(self, "_pending_cpu_blocks", 1)
            if pending <= 0:
                self.finalize_srl_index(session_id)

        if _os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
            if _has_cuda():
                alloc = torch.cuda.memory_allocated() / 1024**3
                print(f"[DiffKV] Post-prefill flush. VRAM: {alloc:.2f} GB")
            else:
                print("[DiffKV] Post-prefill flush (MPS/CPU).")

    def compress_deferred_prefill_blocks(self, session_id: str) -> None:
        """
        Trigger SVD compression for all deferred prefill blocks of a session.
        Called once after the entire prefill is finished.
        """
        if self._streaming_mgr is not None:
            self._streaming_mgr.compress_deferred_blocks(session_id)



    def finalize_compressed_blocks(self):
        """
        Uploads CPU-compressed blocks (SVD computed on background thread) to GPU
        and writes them to the native block pool. Runs on the main thread to ensure
        MPS/Metal thread safety.

        Fast-path: returns immediately (O(1)) when _pending_cpu_blocks == 0.
        This is the common case during decode after all prefill blocks are finalized.
        """
        if self._streaming_mgr is None:
            return

        # O(1) early exit — no blocks waiting. Avoids full session×layer×block scan
        # on every decode token once all prefill blocks are finalized.
        with self._pending_lock:
            if self._pending_cpu_blocks <= 0:
                return

        # Scan all resident sessions and layers
        for session_id, layers in list(self._streaming_mgr.session_blocks.items()):
            for layer_idx, blocks in list(layers.items()):
                for block in blocks:
                    if getattr(block, "state", None) == "CPU_COMPRESSED":
                        try:
                            # Perform the GPU/Metal upload on the main thread
                            gpu_device = block.anchor_kv.device
                            u_cpu = getattr(block, "U_cpu", None)
                            v_cpu = getattr(block, "V_cpu", None)

                            if u_cpu is None or v_cpu is None:
                                # Still waiting for compressor to populate — skip for now
                                continue

                            block.U = u_cpu.to(gpu_device)
                            block.V = v_cpu.to(gpu_device)

                            # Clean up temporary CPU tensors
                            block.U_cpu = None
                            block.V_cpu = None

                            # Write to native pool
                            if hasattr(self, 'native_pool') and self.native_pool is not None:
                                if getattr(block, 'pool_idx', None) is None:
                                    block.pool_idx = self.native_pool.allocate_block()
                                self.native_pool.write_block(
                                    pool_idx=block.pool_idx,
                                    U=block.U,
                                    V=block.V,
                                    anchor_K=self._get_rotated_anchor_k(session_id, block.anchor_kv[0, 0], block.anchor_idx),
                                    anchor_V=block.anchor_kv[0, 1],
                                    scale=block.scale,
                                    seq_len=block.U.shape[0]
                                )

                            # Mark finalized and decrement pending counter
                            block.state = "COMPRESSED"
                            with self._pending_lock:
                                self._pending_cpu_blocks = max(0, self._pending_cpu_blocks - 1)
                            self._streaming_mgr.update_metadata_state(session_id, layer_idx, block)
                        except Exception as e:
                            import traceback
                            print(f"[DiffKV WARNING] Failed to finalize CPU-compressed block: {e}")
                            traceback.print_exc()
                            # Decrement on failure too to avoid infinite loops
                            with self._pending_lock:
                                self._pending_cpu_blocks = max(0, self._pending_cpu_blocks - 1)

    def clear_session(self, session_id: str):
        # Invalidate GPU block indices cache and workspaces
        if hasattr(self, 'decode_workspace'):
            self.decode_workspace.pop(session_id, None)

        # Free blocks from NativeBlockPool before deleting references
        if hasattr(self, 'native_pool') and self.native_pool is not None:
            if session_id in self.session_blocks:
                for layer_idx, blocks in self.session_blocks[session_id].items():
                    for block in blocks:
                        if getattr(block, 'pool_idx', None) is not None:
                            self.native_pool.free_block(block.pool_idx)
                            block.pool_idx = None
            if self._streaming_mgr is not None and session_id in self._streaming_mgr.session_blocks:
                for layer_idx, blocks in self._streaming_mgr.session_blocks[session_id].items():
                    for block in blocks:
                        if getattr(block, 'pool_idx', None) is not None:
                            self.native_pool.free_block(block.pool_idx)
                            block.pool_idx = None

        # Workspaces and caches are already cleared by pop(session_id) above
        pass

        if session_id in self.session_blocks:
            del self.session_blocks[session_id]
        if self._streaming_mgr is not None:
            # Count CPU_COMPRESSED blocks being evicted so the counter stays accurate
            evicted_pending = 0
            if session_id in self._streaming_mgr.session_blocks:
                for layer_idx, blocks in self._streaming_mgr.session_blocks[session_id].items():
                    for b in blocks:
                        if getattr(b, "state", None) == "CPU_COMPRESSED":
                            evicted_pending += 1
            if evicted_pending > 0:
                with self._pending_lock:
                    self._pending_cpu_blocks = max(0, self._pending_cpu_blocks - evicted_pending)
            self._streaming_mgr.clear_session(session_id)
        self.pager.evict_session(session_id)
        # Invalidate mbs cache for this session
        self._mbs_cache.pop(session_id, None)
        # Clean up SRL state and stored token IDs
        self._session_srl.pop(session_id, None)
        self._session_token_ids.pop(session_id, None)
        if hasattr(self, "attention_score_cache"):
            self.attention_score_cache.clear_session(session_id)

    def snapshot_session(self, session_id: str, checkpoint_id: str):
        """
        Takes a zero-copy metadata snapshot of the session's current KV state.
        Saves it under checkpoint_id.
        """
        if not hasattr(self, "_session_checkpoints"):
            self._session_checkpoints = {}

        if session_id not in self.session_blocks and (self._streaming_mgr is None or session_id not in self._streaming_mgr.session_blocks):
            raise ValueError(f"Session {session_id} not found to snapshot.")

        snap_blocks = {}
        if self._streaming_mgr is not None and session_id in self._streaming_mgr.session_blocks:
            src_blocks = self._streaming_mgr.session_blocks[session_id]
        else:
            src_blocks = self.session_blocks[session_id]

        for layer_idx, blocks in src_blocks.items():
            snap_blocks[layer_idx] = []
            for b in blocks:
                import copy
                b_snap = copy.copy(b)
                b_snap._lock = threading.Lock()
                b_snap.token_indices = list(b.token_indices)
                
                # Clone active GPU buffers and re-establish views to ensure complete isolation
                if getattr(b, "_active_buf_k", None) is not None:
                    b_snap._active_buf_k = b._active_buf_k.clone()
                    fill = getattr(b, "_active_fill", 0)
                    if fill > 0:
                        b_snap.active_k = b_snap._active_buf_k[:, :, :fill, :]
                    else:
                        b_snap.active_k = None
                elif getattr(b, "active_k", None) is not None:
                    b_snap.active_k = b.active_k.clone()

                if getattr(b, "_active_buf_v", None) is not None:
                    b_snap._active_buf_v = b._active_buf_v.clone()
                    fill = getattr(b, "_active_fill", 0)
                    if fill > 0:
                        b_snap.active_v = b_snap._active_buf_v[:, :, :fill, :]
                    else:
                        b_snap.active_v = None
                elif getattr(b, "active_v", None) is not None:
                    b_snap.active_v = b.active_v.clone()

                # Clone CPU-pinned uncompressed caches
                if getattr(b, "active_k_cpu", None) is not None:
                    b_snap.active_k_cpu = b.active_k_cpu.clone()
                if getattr(b, "active_v_cpu", None) is not None:
                    b_snap.active_v_cpu = b.active_v_cpu.clone()

                if getattr(b, "anchor_kv", None) is not None:
                    b_snap.anchor_kv = b.anchor_kv.clone()
                if getattr(b, "anchor_kv_cpu", None) is not None:
                    b_snap.anchor_kv_cpu = b.anchor_kv_cpu.clone()

                # Increment pool reference counts for compressed blocks
                if getattr(b, "pool_idx", None) is not None and getattr(self, "native_pool", None) is not None:
                    self.native_pool.increment_ref(b.pool_idx)

                snap_blocks[layer_idx].append(b_snap)

        mbs = self.get_session_micro_block_size(session_id)
        
        # Clone token IDs and deepcopy configs if present
        token_ids_snap = None
        if session_id in self._session_token_ids:
            token_ids_snap = self._session_token_ids[session_id].clone()
            
        configs_snap = None
        if hasattr(self, "session_configs") and session_id in self.session_configs:
            configs_snap = copy.deepcopy(self.session_configs[session_id])
        
        self._session_checkpoints[checkpoint_id] = {
            "blocks": snap_blocks,
            "micro_block_size": mbs,
            "token_ids": token_ids_snap,
            "configs": configs_snap
        }
        print(f"[DiffKV] Session snapshot captured: {session_id} -> {checkpoint_id}")

    def get_session_sequence_length(self, session_id: str) -> int:
        """
        Universal query to get the exact sequence length currently cached in blocks.
        Model-agnostic and session-residency safe.

        Handles all block states:
          - ACCUMULATING: anchor + active_k tokens
          - SUBMITTED:    anchor + tokens in token_indices (active_k may be None during SVD)
          - COMPRESSED:   anchor + U.shape[0] rows
          - PAGED:        anchor + active_k_cpu tokens
        """
        blocks = self.get_streaming_blocks(session_id, 0)
        if not blocks:
            return 0
        last_block = blocks[-1]

        # Use token_indices list length when available — it is always kept up-to-date
        # regardless of block state (SUBMITTED blocks have active_k=None during SVD).
        token_indices = getattr(last_block, "token_indices", None)
        if token_indices is not None and len(token_indices) > 0:
            return int(token_indices[-1]) + 1

        # Fallback: compute from U or active_k
        last_U_len = last_block.U.shape[0] if last_block.U is not None else 0
        last_active_len = 0
        if last_block.active_k is not None:
            last_active_len = last_block.active_k.shape[2]
        elif getattr(last_block, "active_k_cpu", None) is not None:
            last_active_len = last_block.active_k_cpu.shape[2]
        last_token_count = last_U_len if last_U_len > 0 else last_active_len
        return last_block.anchor_idx + 1 + last_token_count


    def log_block_states(self, session_id: str) -> None:
        """
        Emit a per-layer block state summary for *session_id* when
        DIFFKV_TELEMETRY=1.  Counts blocks in each lifecycle state:
          ACCUMULATING — dense, not yet eligible for compression
          SUBMITTED    — queued for async SVD, still holding active_k/v in VRAM
          COMPRESSED   — U/V set, active_k/v freed, minimal VRAM footprint
          PAGED        — evicted to CPU RAM
        This is the primary diagnostic for VRAM anomalies.
        """
        import os
        if os.environ.get("DIFFKV_TELEMETRY", "0") != "1":
            return

        streaming_mgr = getattr(self, "_streaming_mgr", None)
        if streaming_mgr is not None:
            src = streaming_mgr.session_blocks.get(session_id, {})
        else:
            src = self.session_blocks.get(session_id, {})

        if not src:
            print(f"[DiffKV BlockStates] session={session_id}: no blocks found")
            return

        # Aggregate across all layers
        state_counts: Dict[str, int] = {"ACCUMULATING": 0, "SUBMITTED": 0, "COMPRESSED": 0, "PAGED": 0, "UNKNOWN": 0}
        total_active_vram_mb = 0.0
        total_uv_vram_mb = 0.0
        total_blocks = 0

        for layer_idx, blocks in src.items():
            for block in blocks:
                total_blocks += 1
                state = getattr(block, "state", "UNKNOWN")
                state_counts[state] = state_counts.get(state, 0) + 1

                # active_k/v VRAM (dense accumulation or SUBMITTED residual)
                if getattr(block, "active_k", None) is not None:
                    total_active_vram_mb += block.active_k.numel() * block.active_k.element_size() / 1e6
                if getattr(block, "active_v", None) is not None:
                    total_active_vram_mb += block.active_v.numel() * block.active_v.element_size() / 1e6

                # U/V VRAM (compressed representation)
                if getattr(block, "U", None) is not None:
                    total_uv_vram_mb += block.U.numel() * block.U.element_size() / 1e6
                if getattr(block, "V", None) is not None:
                    total_uv_vram_mb += block.V.numel() * block.V.element_size() / 1e6

        print(
            f"[DiffKV BlockStates] session={session_id} "
            f"total_blocks={total_blocks} "
            f"ACCUMULATING={state_counts['ACCUMULATING']} "
            f"SUBMITTED={state_counts['SUBMITTED']} "
            f"COMPRESSED={state_counts['COMPRESSED']} "
            f"PAGED={state_counts['PAGED']} "
            f"| dense_active_VRAM={total_active_vram_mb:.1f} MB "
            f"| UV_compressed_VRAM={total_uv_vram_mb:.1f} MB"
        )

    def restore_session(self, session_id: str, checkpoint_id: str):
        """
        Restores a session to a previously saved checkpoint_id state.
        """
        if not hasattr(self, "_session_checkpoints") or checkpoint_id not in self._session_checkpoints:
            raise KeyError(f"Checkpoint {checkpoint_id} not found.")

        # Clean existing session blocks safely (refcounts will decrement correctly)
        self.clear_session(session_id)
        self.init_session(session_id)

        checkpoint = self._session_checkpoints[checkpoint_id]
        snap_blocks = checkpoint["blocks"]
        mbs = checkpoint["micro_block_size"]

        if self._streaming_mgr is not None:
            self._streaming_mgr.session_micro_block_sizes[session_id] = mbs

        dest_blocks = {}
        for layer_idx, blocks in snap_blocks.items():
            dest_blocks[layer_idx] = []
            for b in blocks:
                import copy
                b_restore = copy.copy(b)
                b_restore._lock = threading.Lock()
                b_restore.token_indices = list(b.token_indices)
                
                # Clone active GPU buffers and re-establish views to ensure complete isolation
                if getattr(b, "_active_buf_k", None) is not None:
                    b_restore._active_buf_k = b._active_buf_k.clone()
                    fill = getattr(b, "_active_fill", 0)
                    if fill > 0:
                        b_restore.active_k = b_restore._active_buf_k[:, :, :fill, :]
                    else:
                        b_restore.active_k = None
                elif getattr(b, "active_k", None) is not None:
                    b_restore.active_k = b.active_k.clone()

                if getattr(b, "_active_buf_v", None) is not None:
                    b_restore._active_buf_v = b._active_buf_v.clone()
                    fill = getattr(b, "_active_fill", 0)
                    if fill > 0:
                        b_restore.active_v = b_restore._active_buf_v[:, :, :fill, :]
                    else:
                        b_restore.active_v = None
                elif getattr(b, "active_v", None) is not None:
                    b_restore.active_v = b.active_v.clone()

                # Clone CPU-pinned uncompressed caches
                if getattr(b, "active_k_cpu", None) is not None:
                    b_restore.active_k_cpu = b.active_k_cpu.clone()
                if getattr(b, "active_v_cpu", None) is not None:
                    b_restore.active_v_cpu = b.active_v_cpu.clone()

                if getattr(b, "anchor_kv", None) is not None:
                    b_restore.anchor_kv = b.anchor_kv.clone()
                if getattr(b, "anchor_kv_cpu", None) is not None:
                    b_restore.anchor_kv_cpu = b.anchor_kv_cpu.clone()

                # Increment pool reference counts for compressed blocks
                if getattr(b, "pool_idx", None) is not None and getattr(self, "native_pool", None) is not None:
                    self.native_pool.increment_ref(b.pool_idx)

                dest_blocks[layer_idx].append(b_restore)
                if self._streaming_mgr is not None:
                    self._streaming_mgr.update_metadata_block(session_id, layer_idx, len(dest_blocks[layer_idx]) - 1, b_restore)

        if self._streaming_mgr is not None:
            self._streaming_mgr.session_blocks[session_id] = dest_blocks
        else:
            self.session_blocks[session_id] = dest_blocks

        # Restore token IDs and configs
        import copy
        if checkpoint.get("token_ids") is not None:
            self._session_token_ids[session_id] = checkpoint["token_ids"].clone()
        else:
            self._session_token_ids.pop(session_id, None)

        if checkpoint.get("configs") is not None:
            if not hasattr(self, "session_configs"):
                self.session_configs = {}
            self.session_configs[session_id] = copy.deepcopy(checkpoint["configs"])

        # Reconstruct / rebuild the SRL Sparse Routing indices
        self.finalize_srl_index(session_id)

        # Invalidate cached categorized structures
        self.pager.evict_session(session_id)
        print(f"[DiffKV] Session restored from checkpoint: {checkpoint_id} -> {session_id}")

    def clone_session(self, src_session_id: str, dest_session_id: str):
        """
        Clones src_session_id state to a new dest_session_id (zero-copy branching).
        """
        temp_ckpt = f"_temp_clone_{src_session_id}_{dest_session_id}"
        self.snapshot_session(src_session_id, temp_ckpt)
        self.restore_session(dest_session_id, temp_ckpt)
        self.delete_checkpoint(temp_ckpt)

    def delete_checkpoint(self, checkpoint_id: str):
        """
        Deletes a saved checkpoint and releases its block pool references.
        """
        if not hasattr(self, "_session_checkpoints") or checkpoint_id not in self._session_checkpoints:
            return
        checkpoint = self._session_checkpoints[checkpoint_id]
        if getattr(self, "native_pool", None) is not None:
            for layer_idx, blocks in checkpoint["blocks"].items():
                for b in blocks:
                    if getattr(b, "pool_idx", None) is not None:
                        self.native_pool.free_block(b.pool_idx)
        del self._session_checkpoints[checkpoint_id]

    def clear(self):
        # 1. Cleanly clear all registered sessions to release their pool blocks
        sessions = set(self.session_blocks.keys())
        if self._streaming_mgr is not None and hasattr(self._streaming_mgr, 'session_blocks'):
            sessions.update(self._streaming_mgr.session_blocks.keys())
            
        for session_id in sessions:
            self.clear_session(session_id)

        # Free checkpoints
        if hasattr(self, "_session_checkpoints"):
            if getattr(self, "native_pool", None) is not None:
                for ckpt in self._session_checkpoints.values():
                    for layer_idx, blocks in ckpt["blocks"].items():
                        for b in blocks:
                            if getattr(b, "pool_idx", None) is not None:
                                self.native_pool.free_block(b.pool_idx)
            self._session_checkpoints.clear()

        # 2. Reset subsystems and clear references
        self.session_blocks.clear()
        if hasattr(self, 'decode_workspace'):
            self.decode_workspace.clear()
        if hasattr(self, '_decode_block_cache'):
            self._decode_block_cache.clear()
        
        if hasattr(self, 'native_pool') and self.native_pool is not None:
            self.native_pool.reset()
            
        if hasattr(self, 'pager') and self.pager is not None:
            self.pager.clear()

    # ── Phase 24.5: Streaming Sparse Ingest ───────────────────────────────────

    def ingest_streaming(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,  # [1, heads, T, head_dim] — full prefill or single decode token
        v: torch.Tensor,
    ) -> None:
        """
        Phase 24.5 entry point: streaming sparse ingest.

        Routes tokens through StreamingSparseIngestManager instead of set_kv().
        - Compresses during ingest, not after.
        - Dense footprint bounded to 1 micro-block (default 16 tokens) at any time.
        - For decode (T=1), delegates to append_decode_token().
        """
        if self._streaming_mgr is None:
            # Fallback to legacy dense path if streaming disabled
            self.set_kv(session_id, layer_idx, k, v)
            return

        if session_id not in self._streaming_mgr.session_blocks:
            self.init_session(session_id, prefill_len=k.shape[2])
        elif k.shape[2] > 1 and session_id in self._streaming_mgr.session_micro_block_sizes:
            # Update adaptive size on the first chunked prefill if it was set by a
            # decode-only init (no prefill_len known at session creation time).
            # Use same tiered schedule as StreamingSparseIngestManager.init_session().
            existing = self._streaming_mgr.session_micro_block_sizes[session_id]
            if existing == min(32, self._streaming_mgr.micro_block_size):
                seq_len = k.shape[2]
                if seq_len < 256:
                    raw_target = 16
                elif seq_len < 1024:
                    raw_target = 32
                elif seq_len < 4096:
                    raw_target = 64
                elif seq_len < 8192:
                    raw_target = 128
                else:
                    raw_target = 256
                target = min(raw_target, self._streaming_mgr.micro_block_size)
                target = max(16, ((target + 15) // 16) * 16)
                self._streaming_mgr.session_micro_block_sizes[session_id] = target



        if k.shape[2] == 1:
            self._streaming_mgr.append_decode_token(session_id, layer_idx, k, v)
        else:
            self._streaming_mgr.ingest_chunk(session_id, layer_idx, k, v)

    def get_streaming_blocks(self, session_id: str, layer_idx: int) -> list:
        """
        Return blocks from streaming manager (Phase 24.5 path).
        Falls back to legacy session_blocks if streaming not active.
        """
        if self._streaming_mgr is not None and session_id in self._streaming_mgr.session_blocks:
            return self._streaming_mgr.get_blocks(session_id, layer_idx)
        return self.get_raw_blocks(session_id, layer_idx)

    def get_streaming_summary(self, session_id: str = None) -> dict:
        """Return streaming ingest telemetry for Phase 24.5 reporting."""
        if self._streaming_mgr is None:
            return {"streaming_ingest": False}
        s = self._streaming_mgr.summary(session_id)
        s["streaming_ingest"] = True
        s["micro_block_size"] = self.micro_block_size
        return s

    def get_session_micro_block_size(self, session_id: str) -> int:
        """
        Phase 29 Fix #4: Cached lookup. The micro_block_size is fixed at prefill time
        and never changes during decode. Previously this scanned ALL blocks in ALL layers
        every single decode step — O(N·L) per token. Now O(1) after first call.
        """
        cached = self._mbs_cache.get(session_id)
        if cached is not None:
            return cached

        # First access: compute the real value
        max_size = 0
        if self._streaming_mgr is not None and session_id in self._streaming_mgr.session_blocks:
            # Read directly from the session_micro_block_sizes dict (O(1))
            mbs = self._streaming_mgr.session_micro_block_sizes.get(session_id, 0)
            if mbs > max_size:
                max_size = mbs

        if max_size == 0 and session_id in self.session_blocks:
            for layer_idx, blocks in self.session_blocks[session_id].items():
                for block in blocks:
                    block_mbs = getattr(block, "micro_block_size", 0)
                    if block_mbs > max_size:
                        max_size = block_mbs

        if max_size == 0:
            max_size = self.micro_block_size

        self._mbs_cache[session_id] = max_size
        return max_size

    def get_cached_decode_blocks(
        self,
        session_id: str,
        layer_idx: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], List[Any], Optional[torch.Tensor], Optional[int], Optional[int]]:
        """
        Vectorized O(1) metadata retrieval from contiguous packed CPU tensors.
        Phase 29: metadata is now CPU-resident (zero CUDA syncs on write).
        Only the final small block_indices array is transferred to GPU.
        """
        if self._streaming_mgr is None:
            return None, [], None, None, None

        blocks = self.get_streaming_blocks(session_id, layer_idx)
        num_blocks = len(blocks) if blocks else 0
        if num_blocks == 0:
            return None, [], None, None, None

        metadata = self._streaming_mgr.session_metadata.get(session_id, {}).get(layer_idx)
        if metadata is None:
            return None, [], None, None, None

        active_meta = metadata[:num_blocks]          # CPU slice view

        # state_code 2 == COMPRESSED
        compressed_mask = active_meta[:, 3] == 2     # CPU compare
        
        max_anchor_idx = None
        max_valid_len = None
        # Phase 32: GPU block indices cache check
        if compressed_mask.any():                    # CPU any() — no CUDA sync
            cpu_indices = active_meta[compressed_mask, 0]
            cpu_anchors = active_meta[compressed_mask, 1]
            cpu_seq_lens = active_meta[compressed_mask, 2]
            max_anchor_idx = int(cpu_anchors.max().item())
            max_valid_len = int(cpu_seq_lens.max().item())
            session_dict = self.decode_workspace.setdefault(session_id, {})
            indices_gpu_cache = session_dict.setdefault("indices_gpu", {})
            cached_val = indices_gpu_cache.get(layer_idx)
            if cached_val is not None:
                cached_cpu_ind, cached_gpu_ind, cached_cpu_anc, cached_gpu_anc = cached_val
                if cached_cpu_ind.shape[0] == cpu_indices.shape[0] and torch.equal(cached_cpu_ind, cpu_indices):
                    block_indices_tensor = cached_gpu_ind
                    anchor_indices_gpu = cached_gpu_anc
                else:
                    block_indices_tensor = cpu_indices.to(device)
                    anchor_indices_gpu = cpu_anchors.to(device)
                    indices_gpu_cache[layer_idx] = (cpu_indices, block_indices_tensor, cpu_anchors, anchor_indices_gpu)
            else:
                block_indices_tensor = cpu_indices.to(device)
                anchor_indices_gpu = cpu_anchors.to(device)
                indices_gpu_cache[layer_idx] = (cpu_indices, block_indices_tensor, cpu_anchors, anchor_indices_gpu)
        else:
            block_indices_tensor = None
            anchor_indices_gpu = None

        # Get ALL non-compressed, non-paged blocks as dense context.
        # Phase 32 backwards optimization: since blocks are compressed chronologically,
        # we can traverse backwards and break as soon as we see a COMPRESSED/PAGED block.
        dense_blocks = [block for block in blocks if block.state not in ("COMPRESSED", "PAGED")]

        return block_indices_tensor, dense_blocks, anchor_indices_gpu, max_anchor_idx, max_valid_len

    def assemble_dense_window_kv(
        self,
        session_id: str,
        layer_idx: int,
        dense_blocks: list,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Lightweight dense-window-only KV assembler for the decode hot path.

        ONLY processes non-compressed (ACCUMULATING / SUBMITTED) blocks.
        Compressed blocks are handled by block_indices in the sparse kernel.

        Pre-allocates static workspace tensors to eliminate torch.cat dynamic malloc
        and fragmenting allocator memory growth on Apple Silicon (MPS).
        """
        if not dense_blocks:
            return None, None

        # 1. Compute total length of dense tokens
        L_dense = 0
        for blk in dense_blocks:
            L_dense += 1
            if blk.active_k is not None:
                L_dense += blk.active_k.shape[2]
            elif getattr(blk, "active_k_cpu", None) is not None:
                L_dense += blk.active_k_cpu.shape[2]

        # 2. Retrieve or allocate workspaces
        session_dict = self.decode_workspace.setdefault(session_id, {})
        dense_k_cache = session_dict.setdefault("dense_workspace_k", {})
        dense_v_cache = session_dict.setdefault("dense_workspace_v", {})
        workspace_k = dense_k_cache.get(layer_idx)
        workspace_v = dense_v_cache.get(layer_idx)

        if (workspace_k is None 
            or workspace_k.shape[1] != self.kv_heads 
            or workspace_k.dtype != dtype 
            or workspace_k.shape[2] < L_dense):
            
            # Align allocation length to multiples of 512 to prevent VRAM fragmentation
            alloc_len = ((L_dense + 511) // 512) * 512
            workspace_k = torch.zeros((1, self.kv_heads, alloc_len, self.head_dim), device=self.device, dtype=dtype)
            workspace_v = torch.zeros((1, self.kv_heads, alloc_len, self.head_dim), device=self.device, dtype=dtype)
            dense_k_cache[layer_idx] = workspace_k
            dense_v_cache[layer_idx] = workspace_v

        # 3. Copy tokens directly into the pre-allocated workspace slices
        curr_idx = 0
        for blk in dense_blocks:
            workspace_k[:, :, curr_idx : curr_idx + 1].copy_(blk.anchor_kv[:, 0].unsqueeze(2), non_blocking=True)
            workspace_v[:, :, curr_idx : curr_idx + 1].copy_(blk.anchor_kv[:, 1].unsqueeze(2), non_blocking=True)
            curr_idx += 1

            if blk.active_k is not None:
                active_len = blk.active_k.shape[2]
                workspace_k[:, :, curr_idx : curr_idx + active_len].copy_(blk.active_k, non_blocking=True)
                workspace_v[:, :, curr_idx : curr_idx + active_len].copy_(blk.active_v, non_blocking=True)
                curr_idx += active_len
            elif getattr(blk, "active_k_cpu", None) is not None:
                active_len = blk.active_k_cpu.shape[2]
                workspace_k[:, :, curr_idx : curr_idx + active_len].copy_(blk.active_k_cpu, non_blocking=True)
                workspace_v[:, :, curr_idx : curr_idx + active_len].copy_(blk.active_v_cpu, non_blocking=True)
                curr_idx += active_len

        # 4. Return zero-allocation slice views of the static workspace
        return workspace_k[:, :, :L_dense], workspace_v[:, :, :L_dense]

    # ── High-throughput decode KV assembly ────────────────────────────────────

    def assemble_decode_kv(
        self,
        session_id: str,
        layer_idx: int,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Vectorized KV assembly for the decode hot path.

        Design goals vs. the old block-by-block loop in diffkv_attention.py:
          1. Single workspace tensor reused across decode steps (no per-step malloc).
          2. Anchor tokens copied via vectorized index_put_ (not a Python for-loop).
          3. Compressed blocks satisfied from ReconstructionPool where possible
             (zero GEMM cost on hit); batch GEMM for all misses in one bmm call.
          4. Dense active_k/v blocks copied in one per-block copy (unchanged).

        Returns
        -------
        k_tensor : [1, kv_heads, total_seq_len, head_dim]  or None if no blocks
        v_tensor : [1, kv_heads, total_seq_len, head_dim]  or None
        """
        blocks = self.get_streaming_blocks(session_id, layer_idx)
        if not blocks:
            return None, None

        # ── 1. Check workspace state and snapshot dirty blocks only ──
        session_dict = self.decode_workspace.setdefault(session_id, {})
        assembled_kv_cache = session_dict.setdefault("assembled_kv", {})
        ws = assembled_kv_cache.get(layer_idx)
        need_full_snapshot = (ws is None)

        snapshots = []
        if need_full_snapshot:
            for b in blocks:
                snapshots.append(BlockSnapshot(b))
        else:
            for b in blocks:
                if b.dirty:
                    snapshots.append(BlockSnapshot(b))

        # ── 2. Compute total sequence length in O(1) from the last block ─────────
        last_block = blocks[-1]
        lock = getattr(last_block, "_lock", None)
        if lock is not None:
            with lock:
                last_U = last_block.U
                last_active_k = last_block.active_k
                last_active_k_cpu = getattr(last_block, "active_k_cpu", None)
                last_U_len = last_U.shape[0] if last_U is not None else 0
                last_active_len = 0
                if last_active_k is not None:
                    last_active_len = last_active_k.shape[2]
                elif last_active_k_cpu is not None:
                    last_active_len = last_active_k_cpu.shape[2]
        else:
            last_U = last_block.U
            last_active_k = last_block.active_k
            last_active_k_cpu = getattr(last_block, "active_k_cpu", None)
            last_U_len = last_U.shape[0] if last_U is not None else 0
            last_active_len = 0
            if last_active_k is not None:
                last_active_len = last_active_k.shape[2]
            elif last_active_k_cpu is not None:
                last_active_len = last_active_k_cpu.shape[2]

        last_token_count = last_U_len if last_U_len > 0 else last_active_len
        total_seq_len = last_block.anchor_idx + 1 + last_token_count

        # ── 3. Allocate or resize persistent workspace with vectorized copy ──────
        ws = assembled_kv_cache.get(layer_idx)
        if ws is None:
            # Allocate with 5% headroom to reduce future reallocations while minimising
            # persistent VRAM waste. Reallocation is rare (vectorised GPU copy) and cheap.
            alloc_len = max(total_seq_len + max(total_seq_len // 20, 64), 128)
            k_ws = torch.zeros(
                (1, self.kv_heads, alloc_len, self.head_dim),
                dtype=dtype, device=self.device
            )
            v_ws = torch.zeros(
                (1, self.kv_heads, alloc_len, self.head_dim),
                dtype=dtype, device=self.device
            )
            assembled_kv_cache[layer_idx] = (k_ws, v_ws)
            ws = (k_ws, v_ws)
            # If it's a completely new workspace, all blocks must be written
            for b_snap in snapshots:
                b_snap.dirty = True
        elif ws[0].shape[2] < total_seq_len:
            # Vectorized GPU-to-GPU copy from old workspace to resized workspace
            old_k, old_v = ws
            old_len = old_k.shape[2]
            alloc_len = max(total_seq_len + max(total_seq_len // 20, 64), 128)
            k_ws = torch.zeros(
                (1, self.kv_heads, alloc_len, self.head_dim),
                dtype=dtype, device=self.device
            )
            v_ws = torch.zeros(
                (1, self.kv_heads, alloc_len, self.head_dim),
                dtype=dtype, device=self.device
            )
            k_ws[..., :old_len, :] = old_k
            v_ws[..., :old_len, :] = old_v
            assembled_kv_cache[layer_idx] = (k_ws, v_ws)
            ws = (k_ws, v_ws)
            # Keep dirty blocks as they are; no need to mark past blocks dirty!

        k_ws, v_ws = ws

        # ── 4. Separate dirty blocks into categories using O(1) offsets ──────────
        anchor_positions = []     # position index in k_ws
        anchor_k_list   = []     # [kv_heads, head_dim] tensors
        anchor_v_list   = []

        compressed_hits   = []   # (b_snap, pool_slot, start_pos)
        compressed_misses = []   # (b_snap, start_pos)

        dense_copies = []        # (b_snap, start_pos, length)

        miss_pool_idxs = []
        dirty_blocks = []
        hit_slots = []

        for b_snap in snapshots:
            if not b_snap.dirty:
                continue

            b_anchor_pos = b_snap.b.anchor_idx
            b_content_pos = b_anchor_pos + 1

            if b_snap.U is not None and b_snap.V is not None:
                block_len = b_snap.U.shape[0]
            elif b_snap.active_k is not None:
                block_len = b_snap.active_k.shape[2]
            elif b_snap.active_k_cpu is not None:
                block_len = b_snap.active_k_cpu.shape[2]
            else:
                block_len = 0

            dirty_blocks.append(b_snap)

            anchor_positions.append(b_anchor_pos)
            anchor_k_list.append(b_snap.anchor_kv[0, 0])
            anchor_v_list.append(b_snap.anchor_kv[0, 1])

            if b_snap.U is not None and b_snap.V is not None:
                compressed_misses.append((b_snap, b_content_pos))
            elif b_snap.active_k is not None:
                dense_copies.append((b_snap, b_content_pos, block_len))
            elif b_snap.active_k_cpu is not None:
                dense_copies.append((b_snap, b_content_pos, block_len))

        # ── 5. Vectorized anchor copy for dirty blocks ─────────────────────────
        if anchor_positions:
            pos_t = torch.tensor(anchor_positions, device=self.device, dtype=torch.long)
            ak_t  = torch.stack(anchor_k_list, dim=0)   # [N, kv_heads, head_dim]
            av_t  = torch.stack(anchor_v_list, dim=0)
            k_ws[0, :, pos_t, :] = ak_t.transpose(0, 1)   # [kv_heads, N, head_dim]
            v_ws[0, :, pos_t, :] = av_t.transpose(0, 1)

        # ── 6. Compressed hits (no-op since recon_pool removed) ────────────────

        # ── 7. Compressed misses — batched GEMM then write into ws ─────────────
        if compressed_misses:
            # Group misses by sequence length to handle variable-size adaptive blocks perfectly
            groups = {}
            for item in compressed_misses:
                b_snap, start_pos = item
                seq_len = b_snap.U.shape[0]
                groups.setdefault(seq_len, []).append(item)

            for seq_len, group_items in groups.items():
                B_grp = len(group_items)
                grp_blocks = [t[0] for t in group_items]
                grp_starts = [t[1] for t in group_items]

                # Dynamic Rank Reconstruction: find the maximum dynamic rank in the batch to avoid redundant compute
                max_k = max([getattr(b_snap, "dynamic_rank", self.rank) for b_snap in grp_blocks])
                if max_k <= 0:
                    max_k = self.rank
                max_k = min(max_k, self.rank)

                # FP16 low-rank reconstruction: direct GPU arithmetic without slow FP32 conversions
                stacked_U = torch.stack([b_snap.U for b_snap in grp_blocks], dim=0) # [B_grp, S, R]
                stacked_V = torch.stack([b_snap.V for b_snap in grp_blocks], dim=0) # [B_grp, R, H_kv * D * 2]

                # Slice along rank dimension for fast BMM
                if max_k < self.rank:
                    stacked_U = stacked_U[:, :, :max_k]
                    stacked_V = stacked_V[:, :max_k, :]

                stacked_scale = torch.tensor(
                    [b_snap.scale for b_snap in grp_blocks], device=self.device, dtype=dtype
                ).view(B_grp, 1, 1)
                stacked_anchor = torch.stack(
                    [b_snap.anchor_kv.reshape(-1).to(dtype) for b_snap in grp_blocks], dim=0
                ).unsqueeze(1) # [B_grp, 1, H_kv * D * 2]

                recon_flat = torch.bmm(stacked_U, stacked_V) * stacked_scale + stacked_anchor

                if not torch.isfinite(recon_flat).all():
                    recon_flat = torch.nan_to_num(recon_flat, nan=0.0, posinf=0.0, neginf=0.0)

                recon = recon_flat.view(B_grp, -1, 2, self.kv_heads, self.head_dim)
                recon_k = recon[:, :, 0].permute(0, 2, 1, 3)
                recon_v = recon[:, :, 1].permute(0, 2, 1, 3)

                for i, (b_snap, start_pos) in enumerate(zip(grp_blocks, grp_starts)):
                    k_ws[0, :, start_pos:start_pos + seq_len, :] = recon_k[i, :, :seq_len, :]
                    v_ws[0, :, start_pos:start_pos + seq_len, :] = recon_v[i, :, :seq_len, :]

        # ── 8. Dense active blocks — slice copy ───────────────────────────────
        for b_snap, start_pos, block_len in dense_copies:
            if b_snap.active_k is not None:
                k_ws[0, :, start_pos:start_pos + block_len, :] = b_snap.active_k[0]
                v_ws[0, :, start_pos:start_pos + block_len, :] = b_snap.active_v[0]
            elif b_snap.active_k_cpu is not None:
                k_ws[0, :, start_pos:start_pos + block_len, :] = b_snap.active_k_cpu[0].to(k_ws.device, non_blocking=True)
                v_ws[0, :, start_pos:start_pos + block_len, :] = b_snap.active_v_cpu[0].to(v_ws.device, non_blocking=True)

        # ── 9. Reset dirty flags on processed blocks under lock, if they haven't changed since snapshot ──
        for b_snap in dirty_blocks:
            lock = getattr(b_snap.b, "_lock", None)
            if lock is not None:
                with lock:
                    if (b_snap.b.U is b_snap.U) and (b_snap.b.active_k is b_snap.active_k) and (getattr(b_snap.b, "active_k_cpu", None) is b_snap.active_k_cpu):
                        b_snap.b.dirty = False
            else:
                if (b_snap.b.U is b_snap.U) and (b_snap.b.active_k is b_snap.active_k) and (getattr(b_snap.b, "active_k_cpu", None) is b_snap.active_k_cpu):
                    b_snap.b.dirty = False

        # ── 10. Return a view of the valid portion (no copy) ───────────────────
        return k_ws[:, :, :total_seq_len, :], v_ws[:, :, :total_seq_len, :]

    # ── Sequence length ───────────────────────────────────────────────────────

    def get_seq_len(self, session_id: str, layer_idx: int = 0) -> int:
        if session_id not in self.session_blocks:
            return 0
        total = 0
        for block in self.session_blocks[session_id][layer_idx]:
            total += 1  # anchor
            if block.U is not None:
                total += block.U.shape[0]
            if block.active_k is not None:
                total += block.active_k.shape[2]
        return total

    # ── Phase 6: raw block access ─────────────────────────────────────────────

    def get_raw_blocks(self, session_id: str, layer_idx: int) -> list:
        """
        Return the raw KVBlock list. Used by fused_sparse_attention_decode.
        Touches each block in the pager to prevent spurious eviction.
        """
        if session_id not in self.session_blocks:
            return []
        blocks = self.session_blocks[session_id][layer_idx]
        for idx, block in enumerate(blocks):
            self.pager.touch(session_id, layer_idx, idx)
        return blocks

    # ── Dense KV reconstruction (legacy path for prefill) ────────────────────

    def get_kv(self, session_id: str, layer_idx: int
               ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Reconstruct full dense KV for prefill path. Uses ReconstructionCache
        to avoid repeat GEMM for already-reconstructed blocks.
        """
        if self.get_seq_len(session_id, layer_idx) > 16384:
            raise RuntimeError(f"[DIFFKV] FATAL MEMORY GUARD: get_kv() called for sequence > 16k tokens (session {session_id}). Full-sequence dense reconstruction bypassed to prevent 20GB OOM spike.")

        if session_id not in self.session_blocks or \
           not self.session_blocks[session_id][layer_idx]:
            return None, None

        blocks = self.session_blocks[session_id][layer_idx]
        k_list, v_list = [], []

        for idx, block in enumerate(blocks):
            self.pager.touch(session_id, layer_idx, idx)

            # Anchor
            k_list.append(block.anchor_kv[:, 0].unsqueeze(2))
            v_list.append(block.anchor_kv[:, 1].unsqueeze(2))

            if block.U is not None and block.V is not None:
                # Check reconstruction cache first
                cached_k, cached_v = None, None
                if getattr(self, "recon_cache", None) is not None:
                    cached_k, cached_v = self.recon_cache.get(block)
                if cached_k is not None:
                    k_list.append(cached_k)
                    v_list.append(cached_v)
                else:
                    anchor_flat = block.anchor_kv.reshape(-1).to(torch.float16)
                    recon = TritonDiffKV.reconstruct_lowrank(
                        block.U, block.V, anchor_flat, scale=block.scale
                    )
                    hds  = block.anchor_kv.shape[2]
                    hdim = block.anchor_kv.shape[3]
                    recon = recon.view(1, -1, 2, hds, hdim)
                    # Use .contiguous().clone() so the cached tensors are fully independent:
                    # .transpose() creates non-contiguous views, and without cloning they'd
                    # alias reconstruct_lowrank's output which may be reused by a future call.
                    recon_k = recon[:, :, 0].transpose(1, 2).contiguous().clone()
                    recon_v = recon[:, :, 1].transpose(1, 2).contiguous().clone()
                    # Cache for reuse
                    if getattr(self, "recon_cache", None) is not None:
                        self.recon_cache.put(block, recon_k, recon_v)
                    k_list.append(recon_k)
                    v_list.append(recon_v)

            if block.active_k is not None:
                k_list.append(block.active_k)
                v_list.append(block.active_v)

        full_k = torch.cat(k_list, dim=2) if len(k_list) > 1 else k_list[0]
        full_v = torch.cat(v_list, dim=2) if len(v_list) > 1 else v_list[0]
        return full_k, full_v

    # ── KV write (from attention forward) ────────────────────────────────────

    def set_kv(self, session_id: str, layer_idx: int,
               k: torch.Tensor, v: torch.Tensor):
        if session_id not in self.session_blocks:
            self.init_session(session_id)

        blocks = self.session_blocks[session_id][layer_idx]

        # ---- Initial prefill phase ----------------------------------------
        if len(blocks) == 0:
            seq_len = k.shape[2]
            for start_idx in range(0, seq_len, self.block_size):
                end_idx = min(start_idx + self.block_size, seq_len)
                chunk_k = k[:, :, start_idx:end_idx]
                chunk_v = v[:, :, start_idx:end_idx]
                anchor_kv = torch.stack(
                    [chunk_k[:, :, 0], chunk_v[:, :, 0]], dim=1
                )
                block = KVBlock(
                    anchor_idx=start_idx,
                    anchor_kv=anchor_kv,
                    token_indices=list(range(start_idx, end_idx)),
                )
                if chunk_k.shape[2] > 1:
                    active_k = chunk_k[:, :, 1:]
                    active_v = chunk_v[:, :, 1:]
                    block.active_k = active_k
                    block.active_v = active_v
                blocks.append(block)
                self.pager.register_block(session_id, layer_idx, len(blocks) - 1, block)

            # Compress oldest prefill blocks (outside recency window)
            full_blocks = [
                b for b in blocks
                if b.anchor_idx > 0 and b.active_k is not None and b.active_k.shape[2] >= self.block_size - 1
            ]
            while len(full_blocks) > self.dense_recency_blocks:
                oldest = full_blocks.pop(0)
                self._submit_compression(oldest, oldest.active_k, oldest.active_v)
            return

        # ---- Continuous decode phase: append 1 token ----------------------
        last_block = blocks[-1]
        curr_k = k[:, :, -1:]
        curr_v = v[:, :, -1:]

        if last_block.U is not None:
            # Last block already compressed — start a new block
            anchor_idx = last_block.anchor_idx + self.block_size
            anchor_kv  = torch.stack([curr_k[:, :, 0], curr_v[:, :, 0]], dim=1)
            new_block  = KVBlock(
                anchor_idx=anchor_idx, anchor_kv=anchor_kv,
                token_indices=[anchor_idx],
            )
            blocks.append(new_block)
            self.pager.register_block(session_id, layer_idx, len(blocks) - 1, new_block)
            return

        if last_block.active_k is None:
            last_block.active_k = curr_k
            last_block.active_v = curr_v
        else:
            last_block.active_k = torch.cat([last_block.active_k, curr_k], dim=2)
            last_block.active_v = torch.cat([last_block.active_v, curr_v], dim=2)

        # Invalidate recon cache for this block since active_k changed
        recon_cache = getattr(self, "recon_cache", None)
        if recon_cache is not None:
            recon_cache.invalidate(last_block)

        # Compress oldest full dense blocks outside the recency window
        full_dense = [
            b for b in blocks
            if b.anchor_idx > 0 and b.U is None and b.active_k is not None
            and b.active_k.shape[2] >= self.block_size - 1
        ]
        while len(full_dense) > self.dense_recency_blocks:
            oldest = full_dense.pop(0)
            self._submit_compression(oldest, oldest.active_k, oldest.active_v)

        # Trigger pager to check budget
        self.pager.maybe_evict()

    # ── Compression helpers ───────────────────────────────────────────────────

    def _submit_compression(self, block: KVBlock,
                            k: torch.Tensor, v: torch.Tensor):
        """Route to async or sync compressor."""
        if self._async:
            self._compressor.submit(block, k, v)
        else:
            self._compress_block_sync(block, k, v)

    def _compress_block_sync(self, block: KVBlock,
                             k: torch.Tensor, v: torch.Tensor):
        """Synchronous SVD compression (used by AsyncCompressor worker).
        Fixed rank -- simple, stable, predictable.
        """
        input_device = k.device
        if input_device.type == "cpu":
            anchor_kv_local = getattr(block, "anchor_kv_cpu", None)
            if anchor_kv_local is None:
                anchor_kv_local = block.anchor_kv.cpu()
        else:
            anchor_kv_local = block.anchor_kv

        # ── Geometric Median Anchor Selection ──
        # Form full block including the old anchor
        full_k = torch.cat([anchor_kv_local[:, 0].unsqueeze(2), k], dim=2)
        full_v = torch.cat([anchor_kv_local[:, 1].unsqueeze(2), v], dim=2)
        S_total = full_k.shape[2]
        
        K_f = full_k[0].permute(1, 0, 2).reshape(S_total, -1).float() # [S_total, heads * head_dim]
        diff = K_f.unsqueeze(0) - K_f.unsqueeze(1)
        sq_dist = (diff ** 2).sum(-1)
        median_idx = sq_dist.sum(-1).argmin().item()
        
        if median_idx > 0:
            # Swap token at index 0 (old anchor) and median_idx in full_k and full_v
            tmp_k = full_k[:, :, 0].clone()
            tmp_v = full_v[:, :, 0].clone()
            full_k[:, :, 0] = full_k[:, :, median_idx]
            full_v[:, :, 0] = full_v[:, :, median_idx]
            full_k[:, :, median_idx] = tmp_k
            full_v[:, :, median_idx] = tmp_v
            
            # Update local anchor
            anchor_kv_local = torch.stack([full_k[:, :, 0], full_v[:, :, 0]], dim=1)
            if input_device.type == "cpu":
                block.anchor_kv_cpu = anchor_kv_local
                block.anchor_kv = anchor_kv_local.to(block.anchor_kv.device)
            else:
                block.anchor_kv = anchor_kv_local
            
            # Update active tokens for SVD
            k = full_k[:, :, 1:]
            v = full_v[:, :, 1:]

        anchor_flat = anchor_kv_local.reshape(-1).float().to(input_device)
        seq_len  = k.shape[2]
        heads    = k.shape[1]
        head_dim = k.shape[3]
        feat_dim = 2 * heads * head_dim

        stacked     = torch.stack([k[0].transpose(0, 1), v[0].transpose(0, 1)], dim=1)
        flat_tokens = stacked.reshape(seq_len, feat_dim).float()
        deltas      = flat_tokens - anchor_flat.unsqueeze(0)

        # Token-wise Norm-Normalization (row-wise)
        token_norms = deltas.norm(dim=1)
        token_norms = torch.clamp(token_norms, min=1e-5)
        normalized_deltas = deltas / token_norms.unsqueeze(1)


        # Per-layer rank with optional early-layer boost.
        # self.config is the DiffKVConfig instance set in __init__.
        _cfg = getattr(self, "config", None)
        _early_boost     = getattr(_cfg, "early_layer_rank_boost", False)
        _max_rank_early  = getattr(_cfg, "max_rank_early", 0)
        _layer_idx_safe  = block.layer_idx if block.layer_idx is not None else 0
        rank = get_layer_rank(
            _layer_idx_safe, self.num_layers, self.rank,
            early_boost=_early_boost, max_rank_early=_max_rank_early,
        )
        lr_delta = compress_lowrank(normalized_deltas, rank)

        # Scale U by token norms to perform token-wise denormalization when reconstructed
        U_scaled = lr_delta.U.float() * token_norms.unsqueeze(1)
        U_scaled = U_scaled.to(torch.float16)

        if self.rank:
            import os as _local_os
            if _local_os.environ.get("DIFFKV_DIAGNOSTICS", "0") == "1":
                print(f"[DiffKV SVD Debug] Block anchor_idx={block.anchor_idx} layer={block.layer_idx}: "
                      f"scale={lr_delta.scale:.4f} cos_sim={lr_delta.cosine_sim:.6f} norm_drift={lr_delta.norm_drift:.6f} "
                      f"dyn_rank={lr_delta.dynamic_rank}")

        is_background = threading.current_thread().name.startswith("DiffKV-Compressor")

        if is_background:
            # CPU-only background SVD to guarantee thread safety
            block.U_cpu          = U_scaled.cpu()
            block.V_cpu          = lr_delta.V.to(torch.float16).cpu()
            block.scale          = lr_delta.scale
            block.cosine_sim     = lr_delta.cosine_sim
            block.norm_drift     = lr_delta.norm_drift
            block.dynamic_rank   = getattr(lr_delta, "dynamic_rank", self.rank)

            block.dirty    = True

            if hasattr(block, 'state'):
                block.state = "CPU_COMPRESSED"
                # Signal main-thread finalizer that there is work to do
        else:
            gpu_device = block.anchor_kv.device
            u_gpu = U_scaled.to(gpu_device)
            v_gpu = lr_delta.V.to(torch.float16).to(gpu_device)

            lock = getattr(block, "_lock", None)
            if lock is None:
                lock = threading.Lock()
                try:
                    block._lock = lock
                except AttributeError:
                    pass

            if lock is not None:
                with lock:
                    block.U          = u_gpu
                    block.V          = v_gpu
                    block.scale      = lr_delta.scale
                    block.cosine_sim = lr_delta.cosine_sim
                    block.norm_drift = lr_delta.norm_drift
                    block.dynamic_rank = getattr(lr_delta, "dynamic_rank", self.rank)

                    block.active_k = None
                    block.active_v = None
                    block.active_k_cpu = None
                    block.active_v_cpu = None
                    block.dirty    = True

                    if hasattr(block, 'state'):
                        block.state = "COMPRESSED"
            else:
                block.U          = u_gpu
                block.V          = v_gpu
                block.scale      = lr_delta.scale
                block.cosine_sim = lr_delta.cosine_sim
                block.norm_drift = lr_delta.norm_drift
                block.dynamic_rank = getattr(lr_delta, "dynamic_rank", self.rank)

                block.active_k = None
                block.active_v = None
                block.active_k_cpu = None
                block.active_v_cpu = None
                block.dirty    = True

                if hasattr(block, 'state'):
                    block.state = "COMPRESSED"

            # Check if block's session is still active/resident
            session_id = getattr(block, 'session_id', None)
            session_active = True
            if session_id is not None:
                if self._streaming_mgr is not None:
                    session_active = session_id in self._streaming_mgr.session_blocks
                else:
                    session_active = session_id in self.session_blocks

            if session_active:
                # Phase 28 Native Block Pool Integration
                if hasattr(self, 'native_pool') and self.native_pool is not None:
                    try:
                        if getattr(block, 'pool_idx', None) is None:
                            block.pool_idx = self.native_pool.allocate_block()
                        self.native_pool.write_block(
                            pool_idx=block.pool_idx,
                            U=block.U,
                            V=block.V,
                            anchor_K=self._get_rotated_anchor_k(session_id, block.anchor_kv[0, 0], block.anchor_idx),
                            anchor_V=block.anchor_kv[0, 1],
                            scale=block.scale,
                            seq_len=block.U.shape[0]
                        )
                    except Exception as e:
                        # Log warning but do not crash generation, as we can still decode using the standard PyTorch path!
                        print(f"[DiffKV] WARNING: Failed to write block to NativeBlockPool: {e}")

                if self._streaming_mgr is not None and getattr(block, 'session_id', None) is not None and getattr(block, 'layer_idx', None) is not None:
                    self._streaming_mgr.update_metadata_state(block.session_id, block.layer_idx, block)

        self.total_compressions += 1
        self.total_cosine_sim   += lr_delta.cosine_sim
        self.total_norm_drift   += lr_delta.norm_drift
        self.rank_histogram[rank] = self.rank_histogram.get(rank, 0) + 1

        fp16_bytes = seq_len * feat_dim * 2
        lr_bytes   = U_scaled.numel() * 2 + lr_delta.V.numel() * 2
        self.vram_saved_bytes += (fp16_bytes - lr_bytes)

    def _get_rotated_anchor_k(self, session_id, anchor_k, anchor_idx):
        """
        Applies RoPE to anchor_k at position anchor_idx using cached cos/sin in decode_workspace.
        """
        session_dict = self.decode_workspace.get(session_id)
        if session_dict is not None:
            cos = session_dict.get("rope_cos")
            sin = session_dict.get("rope_sin")
            if cos is not None and sin is not None and anchor_idx < cos.shape[1]:
                cos_val = cos[0, anchor_idx].to(anchor_k.device, dtype=anchor_k.dtype).view(1, -1)
                sin_val = sin[0, anchor_idx].to(anchor_k.device, dtype=anchor_k.dtype).view(1, -1)
                
                # Apply RoPE
                half_d = anchor_k.shape[-1] // 2
                k1 = anchor_k[..., :half_d]
                k2 = anchor_k[..., half_d:]
                k_rot = torch.cat([-k2, k1], dim=-1)
                return anchor_k * cos_val + k_rot * sin_val
        return anchor_k


    # ── Diagnostics ───────────────────────────────────────────────────────────

    def runtime_summary(self) -> dict:
        pager_s = self.pager.summary()
        recon_cache = getattr(self, "recon_cache", None)
        recon_s = recon_cache.summary() if recon_cache is not None else {}
        comp_s  = self._compressor.summary()
        avg_cos   = (self.total_cosine_sim / max(1, self.total_compressions))
        avg_drift = (self.total_norm_drift  / max(1, self.total_compressions))
        return {
            "sessions":              len(self.session_blocks),
            "total_compressions":    self.total_compressions,
            "avg_cosine_sim":        round(avg_cos, 4),
            "avg_norm_drift":        round(avg_drift, 4),
            "vram_saved_mb":         round(self.vram_saved_bytes / 1e6, 2),
            "fixed_rank":            self.rank,
            "rank_histogram":        dict(sorted(self.rank_histogram.items())),
            "pager":                 pager_s,
            "recon_cache":           recon_s,
            "async_compressor":      comp_s,
        }

