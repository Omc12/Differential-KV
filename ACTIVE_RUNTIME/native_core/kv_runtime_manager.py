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

    Normal schedule (early_boost=False, default, base_rank=16):
      Layers 0-15%:   base_rank          (e.g. 16)
      Layers 15-50%:  base_rank          (e.g. 16)
      Layers 50-79%:  max(6, round(0.75 * base_rank))   (e.g. 12)
      Layers 79%+:    max(8, round(0.50 * base_rank))   (e.g. 8)

    Minimum floor raised from 4 -> 6 for mid layers and 4 -> 8 for final layers:
    at rank 4 the SVD approximation for formula/number blocks degrades enough
    that digit sequences cannot be reproduced reliably. rank 8 is the practical
    minimum for faithful exact-value recall.

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
    elif ratio < 0.79:     # layers 14-22 — slightly reduced, min floor raised to 6
        return max(6, round(0.75 * base_rank))
    else:                  # layers 22-28 — concentrated final layers, min floor raised to 8
        return max(8, round(0.50 * base_rank))



# ─────────────────────────────────────────────────────────────────────────────
# KVBlock definition (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class KVBlock:
    """Physically stores compressed KV memory for one block of tokens."""
    anchor_idx: int
    anchor_kv:  torch.Tensor          # [1, 2, heads, head_dim]
    anchor_kv_cpu: Optional[torch.Tensor] = None
    _U:          Optional[torch.Tensor] = None   # [block_size, rank]
    _V:          Optional[torch.Tensor] = None   # [rank, feat_dim]
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
    pool:     Any = None

    _residual_K_positions: Optional[torch.Tensor] = None
    _residual_K_values:    Optional[torch.Tensor] = None
    _residual_V_positions: Optional[torch.Tensor] = None
    _residual_V_values:    Optional[torch.Tensor] = None

    _U_sem_int4:           Optional[torch.Tensor] = None
    _U_sem_scale:          Optional[torch.Tensor] = None
    _U_fact_fp16:          Optional[torch.Tensor] = None
    _n_semantic:           int = 0

    _fact_anchors_K:       Optional[torch.Tensor] = None
    _fact_anchors_V:       Optional[torch.Tensor] = None
    _fact_anchor_positions: Optional[torch.Tensor] = None

    @property
    def U(self):
        if self._U is not None:
            return self._U
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            seq_len = int(pool.seq_lens[pool_idx].item())
            rank = self.dynamic_rank if self.dynamic_rank > 0 else pool.U.shape[2]
            U_int8 = pool.U[pool_idx, :seq_len, :rank]
            scale_u = pool.U_scale[pool_idx]
            return U_int8.to(scale_u.dtype) * scale_u
        return None

    @U.setter
    def U(self, val):
        self._U = val

    @property
    def V(self):
        if self._V is not None:
            return self._V
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            rank = self.dynamic_rank if self.dynamic_rank > 0 else pool.V_KV.shape[2]
            vk = pool.V_KV[pool_idx, 0, :rank]
            vv = pool.V_KV[pool_idx, 1, :rank]
            vk_flat = vk.reshape(rank, -1)
            vv_flat = vv.reshape(rank, -1)
            return torch.cat([vk_flat, vv_flat], dim=1)
        return None

    @V.setter
    def V(self, val):
        self._V = val

    @property
    def residual_K_positions(self):
        if self._residual_K_positions is not None:
            return self._residual_K_positions
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            return pool.residual_K_positions[pool_idx]
        return None

    @residual_K_positions.setter
    def residual_K_positions(self, val):
        self._residual_K_positions = val

    @property
    def residual_K_values(self):
        if self._residual_K_values is not None:
            return self._residual_K_values
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            return pool.residual_K_values[pool_idx]
        return None

    @residual_K_values.setter
    def residual_K_values(self, val):
        self._residual_K_values = val

    @property
    def residual_V_positions(self):
        if self._residual_V_positions is not None:
            return self._residual_V_positions
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            return pool.residual_V_positions[pool_idx]
        return None

    @residual_V_positions.setter
    def residual_V_positions(self, val):
        self._residual_V_positions = val

    @property
    def residual_V_values(self):
        if self._residual_V_values is not None:
            return self._residual_V_values
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            return pool.residual_V_values[pool_idx]
        return None

    @residual_V_values.setter
    def residual_V_values(self, val):
        self._residual_V_values = val

    @property
    def U_sem_int4(self):
        if self._U_sem_int4 is not None:
            return self._U_sem_int4
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            seq_len = int(pool.seq_lens[pool_idx].item())
            write_seq = (seq_len + 1) // 2
            n_sem = int(pool.n_semantic[pool_idx].item())
            return pool.U_sem[pool_idx, :write_seq, :n_sem]
        return None

    @U_sem_int4.setter
    def U_sem_int4(self, val):
        self._U_sem_int4 = val

    @property
    def U_sem_scale(self):
        if self._U_sem_scale is not None:
            return self._U_sem_scale
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            n_sem = int(pool.n_semantic[pool_idx].item())
            return pool.U_sem_scale[pool_idx, :n_sem]
        return None

    @U_sem_scale.setter
    def U_sem_scale(self, val):
        self._U_sem_scale = val

    @property
    def U_fact_fp16(self):
        if self._U_fact_fp16 is not None:
            return self._U_fact_fp16
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            seq_len = int(pool.seq_lens[pool_idx].item())
            rank = self.dynamic_rank if self.dynamic_rank > 0 else pool.U_fact.shape[2]
            n_sem = int(pool.n_semantic[pool_idx].item())
            n_fact = rank - n_sem
            if n_fact <= 0:
                return torch.empty((seq_len, 0), device=pool.device, dtype=pool.dtype)
            return pool.U_fact[pool_idx, :seq_len, :n_fact]
        return None

    @U_fact_fp16.setter
    def U_fact_fp16(self, val):
        self._U_fact_fp16 = val

    @property
    def n_semantic(self):
        if self._n_semantic > 0:
            return self._n_semantic
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            return int(pool.n_semantic[pool_idx].item())
        return 0

    @n_semantic.setter
    def n_semantic(self, val):
        self._n_semantic = val

    @property
    def fact_anchors_K(self):
        if self._fact_anchors_K is not None:
            return self._fact_anchors_K
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            return pool.fact_anchors_K[pool_idx]
        return None

    @fact_anchors_K.setter
    def fact_anchors_K(self, val):
        self._fact_anchors_K = val

    @property
    def fact_anchors_V(self):
        if self._fact_anchors_V is not None:
            return self._fact_anchors_V
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            return pool.fact_anchors_V[pool_idx]
        return None

    @fact_anchors_V.setter
    def fact_anchors_V(self, val):
        self._fact_anchors_V = val

    @property
    def fact_anchor_positions(self):
        if self._fact_anchor_positions is not None:
            return self._fact_anchor_positions
        if getattr(self, "pool_idx", None) is not None and getattr(self, "pool", None) is not None:
            pool = self.pool
            pool_idx = self.pool_idx
            return pool.fact_anchor_positions[pool_idx]
        return None

    @fact_anchor_positions.setter
    def fact_anchor_positions(self, val):
        self._fact_anchor_positions = val

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
        self._factual_stores: dict = {}

        # Per-session SRL custom configuration settings
        self.session_configs: dict = {}

        # session_id -> layer_idx -> List[KVBlock]
        self.session_blocks: Dict[str, Dict[int, List[KVBlock]]] = {}

        self.block_size           = 64
        if rank >= head_dim:
            self.rank = head_dim // 2
            print(f"[DiffKV] WARNING: Configured SVD rank {rank} is >= head_dim {head_dim}. "
                  f"Capping SVD rank to {self.rank} (head_dim // 2) to preserve accuracy and avoid memory waste.")
        else:
            self.rank = rank
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

        # pool_rank MUST support the maximum rank used by any layer (which might be boosted
        # up to 2× self.rank via early_layer_rank_boost in early layers).
        _cfg = getattr(self, "config", None)
        _early_boost = getattr(_cfg, "early_layer_rank_boost", False)
        _max_rank_early = getattr(_cfg, "max_rank_early", 0)
        if not _early_boost:
            _early_boost = os.environ.get("DIFFKV_EARLY_LAYER_RANK_BOOST", "0") == "1"
        if _max_rank_early == 0:
            try:
                _max_rank_early = int(os.environ.get("DIFFKV_MAX_RANK_EARLY", "0"))
            except ValueError:
                _max_rank_early = 0

        max_possible_rank = max(
            get_layer_rank(l, self.num_layers, self.rank, early_boost=_early_boost, max_rank_early=_max_rank_early)
            for l in range(self.num_layers)
        )
        import math
        pool_rank = int(math.ceil(max_possible_rank * 1.5))
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
        
        # ── MPS memory pressure adaptation ────────────────────────────────────────
        # On Apple Silicon (MPS), aggressively reduce pool allocation when using 'low' preset
        # to prevent OOM errors on systems with 8GB unified memory (4GB MPS limit).
        is_mps = str(self.device) == "mps" or (hasattr(self.device, "type") and self.device.type == "mps")
        is_low_preset = (self.config is not None and getattr(self.config, "preset", "mid") == "low")
        
        # Debug: always log device and preset detection
        print(f"[DiffKV Memory] Device: {self.device} (is_mps={is_mps}), Preset: {getattr(self.config, 'preset', 'unknown')} (is_low={is_low_preset})")
        
        if is_mps and is_low_preset:
            # For MPS + low preset: cap pool at 256MB (more aggressive) and reduce expected tokens by 75%
            pool_budget_bytes = min(pool_budget_bytes, 256 * 1024 ** 2)
            expected_tokens = expected_tokens // 4  # 75% reduction
            print(f"[DiffKV] MPS + low preset detected: reducing pool budget to {pool_budget_bytes // (1024**2)}MB "
                  f"for {expected_tokens} expected tokens to fit 4GB MPS limit")
        else:
            # Clamp pool between 128MB and the budget ceiling.
            #
            # This ceiling is not a VRAM reservation — the pool is lazy, so it
            # only ever allocates the slots a session actually fills.  What the
            # ceiling really sets is max_blocks (below), i.e. the LONGEST
            # CONTEXT the pool can represent:
            #     max_ctx ~= (ceiling / bytes_per_block / num_layers) * block_size
            # So a ceiling that is too low silently caps context instead of
            # saving memory, and the failure looks like pool exhaustion rather
            # than OOM.
            #
            # 4GB was sized around rank 16.  Raising the default rank to 32
            # (MLX parity) grows bytes_per_block ~2x and would have dropped the
            # ceiling on a 48-layer model from ~211K to ~107K tokens — breaking
            # the 128K evaluation.  8GB restores ~200K of headroom at zero
            # allocation cost.  MPS keeps 4GB: unified memory is the hard
            # constraint there, and MLX manages its own pool anyway.
            _ceiling_gb = float(os.environ.get("DIFFKV_POOL_BUDGET_GB", "8"))
            pool_budget_bytes = max(
                128 * 1024 ** 2, min(int(_ceiling_gb * 1024 ** 3), pool_budget_bytes)
            )
        
        min_blocks = 2048 if self.serving_mode == "lightweight" else (4096 if self.serving_mode == "balanced" else 8000)
        
        # MPS + low preset: override min_blocks to prevent OOM
        if is_mps and is_low_preset:
            min_blocks = 512  # Even smaller minimum for memory-constrained devices
        
        dynamic_max_blocks = max(min_blocks, min(65536, pool_budget_bytes // bytes_per_block))
        self.max_blocks = dynamic_max_blocks

        # Surface the context ceiling this pool implies.  max_blocks is shared
        # across layers, so the reachable context is (max_blocks / num_layers)
        # blocks per layer.  Exhausting it does not raise OOM — it just stops
        # accepting compressed blocks — so print it rather than let a long run
        # fail obscurely.  Note bytes_per_block above deliberately excludes the
        # residual arrays; the pool's own accounting includes them, so report
        # the real per-slot cost here too.
        _max_res = getattr(self.config, "max_residual_tokens", 8) if self.config is not None else 8
        _res_bytes = _max_res * (2 + 2 + self.kv_heads * self.head_dim * 2 * 2)
        _true_bpb = bytes_per_block + _res_bytes
        _blocks_per_layer = max(1, dynamic_max_blocks // max(self.num_layers, 1))
        print(
            f"[DiffKV Memory] Pool: max_blocks={dynamic_max_blocks} "
            f"({_blocks_per_layer}/layer x {self.num_layers} layers "
            f"~= {_blocks_per_layer * pool_block_size:,} tokens max context), "
            f"rank={self.rank} (pool_rank={pool_rank}), "
            f"max_residual={_max_res}, "
            f"{_true_bpb / 1024:.0f} KB/slot, "
            f"budget={pool_budget_bytes / 1024**3:.1f} GB, "
            f"worst-case {dynamic_max_blocks * _true_bpb / 1e9:.1f} GB if every slot fills "
            f"(lazy: only used slots are allocated)"
        )

        self.native_pool = NativeBlockPool(
            max_blocks=dynamic_max_blocks,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            rank=pool_rank,
            max_seq_len=pool_block_size,
            device=self.device,
            dtype=torch.float16,
            num_layers=self.num_layers,
            lazy=True,
            max_residual_tokens=self.config.max_residual_tokens,
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
        self.pager.manager = self
        self.decode_workspace = {}

        # On Apple Silicon/MPS, we enable async background SVD with safe CPU-offloaded SVD preprocessing
        if self.device == "mps" or (isinstance(self.device, torch.device) and self.device.type == "mps") or "mps" in str(self.device):
            print("[DiffKV] Auto-detected Apple Silicon / MPS device. Enabling CPU-offloaded async background SVD.")
            self._async = self.config.async_svd
            # Default MPS approximate attention to ON for Apple Silicon.
            # The fused_decode_mps Project-Then-Attend path avoids per-token RoPE
            # reconstruction over compressed blocks (the main decode bottleneck on MPS).
            # Dense window tokens still receive exact pre-rotated attention.
            # Override with DIFFKV_MPS_APPROXIMATE_ATTN=0 to disable.
            import os as _local_os
            if _local_os.environ.get("DIFFKV_MPS_APPROXIMATE_ATTN") is None:
                _local_os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
                print("[DiffKV] Enabled MPS approximate attention fast-path (fused_decode_mps). "
                      "Set DIFFKV_MPS_APPROXIMATE_ATTN=0 to disable.")
        else:
            self._async = self.config.async_svd

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
                # 512-token recency window: keeps the most recent 512 tokens dense for
                # exact attention, compressing everything older. This is sufficient for
                # typical response lengths and yields clear VRAM savings at 4K+ contexts.
                # Increase to 1024 via recency_window=1024 if generation quality drifts.
                recency_window=int(os.environ.get("DIFFKV_RECENCY_WINDOW", "512")),
            )
            self._streaming_mgr.manager = self
        else:
            self._streaming_mgr = None

        _recency_window = int(os.environ.get("DIFFKV_RECENCY_WINDOW", "512"))
        # Compute a generous upper bound for the dense window workspace.
        # During chunked prefill, active_k can temporarily hold up to 2*block_size − 1 tokens
        # per block (e.g. 127 for block_size=64).  Formula:
        #   n_max_blocks  = ceil(recency_window / block_size) + 3   (safety margin)
        #   max_per_block = 2 * block_size + 1                      (anchor + 2× active)
        #   max_dense_len = n_max_blocks × max_per_block
        # This is independent of model architecture — it only depends on block_size and
        # recency_window, both of which are system parameters (not per-model constants).
        _n_max_blocks = (_recency_window + self.block_size - 1) // self.block_size + 3
        _max_per_block = 2 * self.block_size + 1
        self.max_dense_len = int(
            os.environ.get("DIFFKV_MAX_DENSE_LEN", str(_n_max_blocks * _max_per_block))
        )

        self.max_residual = self.native_pool.max_residual_tokens

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

        # Fix 1: Lazy pool allocation — size to actual session context on first use.
        if getattr(self, "native_pool", None) is not None:
            pool = self.native_pool
            growth_factor = 1.5
            block_size = self.micro_block_size if self.streaming_ingest else self.block_size
            block_size = max(block_size, 257)

            if not pool._allocated:
                # First session: allocate pool sized to this session's context.
                # max_tokens_hint is the prompt_len passed from batch_engine.
                # Fall back to prefill_len, then to the serving_mode expected size.
                hint = max_tokens_hint or prefill_len or None
                pool.ensure_allocated(hint)
            elif max_tokens_hint is not None:
                # Pool already allocated: grow if this session needs more space
                needed_blocks = int((max_tokens_hint / block_size) * self.num_layers * growth_factor)
                if needed_blocks > pool.current_blocks:
                    print(f"[DiffKV] Growing block pool from {pool.current_blocks} → "
                          f"{needed_blocks} blocks for session {session_id}")
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

        # Truncate captured prefill/decode K/V states to target_len
        if hasattr(self, "_prefill_kv_capture") and session_id in self._prefill_kv_capture:
            if target_len <= 0:
                self._prefill_kv_capture.pop(session_id, None)
            else:
                session_cap = self._prefill_kv_capture[session_id]
                for layer_idx in list(session_cap.keys()):
                    K_cap, V_cap = session_cap[layer_idx]
                    if K_cap.shape[2] > target_len:
                        session_cap[layer_idx][0] = K_cap[:, :, :target_len, :]
                    if V_cap.shape[2] > target_len:
                        session_cap[layer_idx][1] = V_cap[:, :, :target_len, :]

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
            self._factual_stores.pop(session_id, None)
        else:
            srl_state = self._session_srl.get(session_id)
            if srl_state is not None:
                kept_slots = set()
                if self._streaming_mgr is not None and session_id in self._streaming_mgr.session_blocks:
                    blocks_layer0 = self._streaming_mgr.session_blocks[session_id].get(0, [])
                    for block in blocks_layer0:
                        if block.pool_idx is not None:
                            kept_slots.add(block.pool_idx)
                if hasattr(srl_state, "rollback_to"):
                    srl_state.rollback_to(target_len, kept_slots)

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

        # Gather all COMPRESSED pool slot IDs and anchor indexes for this session (from layer 0)
        blocks_layer0 = self.get_streaming_blocks(session_id, 0)
        slot_ids = []
        anchor_idxs = []
        for b in blocks_layer0:
            if getattr(b, "pool_idx", None) is not None and getattr(b, "state", "") == "COMPRESSED":
                slot_ids.append(b.pool_idx)
                anchor_idxs.append(b.anchor_idx)

        if not slot_ids:
            return  # No compressed blocks yet — skip

        try:
            from native_core.srl.semantic_index import build_semantic_index
            from native_core.srl.chunk_graph import build_chunk_graph
            from native_core.srl.inverted_index import build_inverted_index
            from native_core.srl.session_srl_state import SessionSRLState

            # ── 1. Semantic index ───────────────────────────────────────
            sem_index = build_semantic_index(pool, slot_ids)

            # Get session config if any to dynamically set values on SessionSRLState
            session_config = getattr(self, "session_configs", {}).setdefault(session_id, {})
            default_k_min = int(_os.environ.get("DIFFKV_SRL_K_MIN", "20"))
            default_k_max = int(_os.environ.get("DIFFKV_SRL_K_MAX", "200"))
            default_threshold = self.config.srl_threshold
            default_overlap_threshold = float(_os.environ.get("DIFFKV_SRL_OVERLAP_THRESHOLD", "0.15"))

            default_graph_hop_decay = float(_os.environ.get("DIFFKV_SRL_GRAPH_HOP_DECAY", "0.5"))
            default_srl_age_penalty = self.config.srl_age_penalty

            k_min = session_config.get("srl_k_min", default_k_min)
            k_max = session_config.get("srl_k_max", default_k_max)
            routing_threshold = session_config.get("srl_threshold", default_threshold)
            overlap_threshold = session_config.get("srl_overlap_threshold", default_overlap_threshold)
            graph_hop_decay = session_config.get("srl_graph_hop_decay", default_graph_hop_decay)
            srl_age_penalty = session_config.get("srl_age_penalty", default_srl_age_penalty)

            # ── 3. Inverted token index ─────────────────────────────────
            token_ids_cpu = self._session_token_ids.get(session_id)
            if token_ids_cpu is not None:
                if hasattr(self, "_prefill_kv_capture") and session_id in self._prefill_kv_capture:
                    cap = self._prefill_kv_capture[session_id]
                    if cap:
                        first_layer = list(cap.keys())[0]
                        seq_len_kv = cap[first_layer][0].shape[2]
                        if token_ids_cpu.numel() > seq_len_kv:
                            token_ids_cpu = token_ids_cpu[:seq_len_kv]
                        elif token_ids_cpu.numel() < seq_len_kv:
                            seq_len_kv = token_ids_cpu.numel()
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
                    block_anchor_idxs = anchor_idxs,
                )
                inv_index._tokenizer_ref = self.tokenizer
            else:
                from native_core.srl.inverted_index import InvertedTokenIndex
                inv_index = InvertedTokenIndex(index={}, important_vocab=set())
                inv_index._tokenizer_ref = self.tokenizer

            # ── 2. Chunk graph ──────────────────────────────────────────
            chunk_graph = build_chunk_graph(
                sem_index.desc_matrix,
                sem_index.slot_ids,
                K_semantic=6,
                K_temporal=2,
                inv_index=inv_index,
                overlap_threshold=overlap_threshold,
                blocks=blocks_layer0,
                cached_len=cached_len,
            )

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
                overlap_threshold = overlap_threshold,
                graph_hop_decay   = graph_hop_decay,
                srl_age_penalty   = srl_age_penalty,
            )
            srl_state.ordered_anchor_idxs = [b.anchor_idx for b in blocks_layer0 if getattr(b, "pool_idx", None) is not None and getattr(b, "state", "") == "COMPRESSED"]
            srl_state.cached_len = cached_len
            if existing_srl is not None:
                srl_state.concept_tok_1 = getattr(existing_srl, "concept_tok_1", -1)
                srl_state.concept_tok_2 = getattr(existing_srl, "concept_tok_2", -1)
                srl_state.segment_ids = dict(getattr(existing_srl, "segment_ids", {}))
            srl_state.nothing_found = nothing_found
            srl_state.current_query_tokens = current_query_tokens
            if hasattr(self, "_last_prefill_q") and session_id in self._last_prefill_q:
                srl_state.last_prefill_q = self._last_prefill_q[session_id]
            self._session_srl[session_id] = srl_state

            # ── 6. Assemble and Build Factual Store (Solution 4) ──
            # Gated by DIFFKV_FACTUAL_STORE=1.  Default is OFF to match the MLX
            # wrapper and the documented default behaviour.  When disabled:
            #   - No CPU KV duplicate is retained (_prefill_kv_capture is empty).
            #   - No FactualExactStore is built or queried during decode.
            #   - The factual-logit machinery in hf_diffkv_wrapper.py is skipped
            #     because kv_manager._factual_stores will not contain this session.
            _factual_enabled = _os.environ.get("DIFFKV_FACTUAL_STORE", "0") == "1"
            if not getattr(self, "_factual_store_logged", False):
                _tag = "ENABLED" if _factual_enabled else "DISABLED (set DIFFKV_FACTUAL_STORE=1 to enable)"
                print(f"[DiffKV] Factual store: {_tag}")
                self._factual_store_logged = True
            try:
                from native_core.srl.factual_store import FactualExactStore
                if (
                    _factual_enabled
                    and hasattr(self, "_prefill_kv_capture")
                    and session_id in self._prefill_kv_capture
                ):
                    factual_store = FactualExactStore(session_id)
                    prefill_kv = self._prefill_kv_capture[session_id]
                    
                    prime_slots = set()
                    if chunk_graph is not None and getattr(chunk_graph, "cluster_centers_tensor", None) is not None:
                        prime_slots = set(chunk_graph.cluster_centers_tensor.tolist())
                        
                    factual_store.build(
                        prefill_kv=prefill_kv,
                        token_ids=token_ids_cpu,
                        W_proj=pool.W_proj,
                        stop_token_ids=self._stop_token_ids,
                        slot_ids=slot_ids,
                        block_size=index_block_size,
                        inv_index=inv_index,
                        semantic_prime_slots=prime_slots,
                        block_anchor_idxs=anchor_idxs
                    )
                    self._factual_stores[session_id] = factual_store
                    srl_state.prompt_eagle_scores = getattr(factual_store, "eagle_scores", None)
                    srl_state.setup_sas_and_eqa(token_ids_cpu, self._stop_token_ids, self.tokenizer)
            except Exception as fe:
                print(f"[SRL] WARNING: Failed to build FactualExactStore: {fe}")


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
        soon as the forward pass for that layer returns, while SVD publication is
        deferred until the prefill boundary. This:
          1. Eliminates all torch.cat accumulation (zero extra allocations).
          2. Preserves exact causal attention across prefill chunks.
          3. Keeps the ingest representation block-aligned for one final SVD batch.

        Factual KV capture: CPU copies of K/V are only retained when
        DIFFKV_FACTUAL_STORE=1.  When the flag is absent/"0" the factual-store
        path is skipped entirely, saving O(context_len) host RAM and the
        associated D2H traffic.  This matches the MLX wrapper's gating and the
        documented default behaviour.
        """
        _factual_enabled = os.environ.get("DIFFKV_FACTUAL_STORE", "0") == "1"

        # Only accumulate CPU KV copies when the factual store is enabled.
        # This avoids retaining a full host-RAM duplicate of every prefill K/V
        # and the torch.cat O(N²) allocation spike at long contexts.
        if _factual_enabled:
            if not hasattr(self, "_prefill_kv_capture"):
                self._prefill_kv_capture = {}
            session_cap = self._prefill_kv_capture.setdefault(session_id, {})
            # If we have multiple chunks, concatenate them along chunk_len (dim=2)
            if layer_idx not in session_cap:
                session_cap[layer_idx] = [K.clone().cpu(), V.clone().cpu()]
            else:
                session_cap[layer_idx][0] = torch.cat([session_cap[layer_idx][0], K.clone().cpu()], dim=2)
                session_cap[layer_idx][1] = torch.cat([session_cap[layer_idx][1], V.clone().cpu()], dim=2)

        # Stream directly — ingest_streaming handles block alignment, SVD, pool writes.
        # This is unconditional: the block pool must always be fed regardless of factual
        # store setting.
        self.ingest_streaming(session_id, layer_idx, K, V)

    def compress_prefill_kv(self, session_id: str) -> None:
        """
        Compatibility barrier retained for callers that invoke it after each
        prefill chunk.  SVD publication is intentionally deferred until
        compress_deferred_prefill_blocks() at the prefill boundary.

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

        # O(1) early exit — no blocks waiting. Avoids full session*layer*block scan
        # on every decode token once all prefill blocks are finalized.
        with self._pending_lock:
            if self._pending_cpu_blocks <= 0:
                return
            _pending_before = self._pending_cpu_blocks

        # Timing: measure upload cost. Slow calls (> 2ms) indicate real CPU->GPU
        # work still happening during decode — a sign prefill barrier didn't drain.
        import time as _time
        _t0 = _time.perf_counter()

        # Scan all resident sessions and layers
        for session_id, layers in list(self._streaming_mgr.session_blocks.items()):
            for layer_idx, blocks in list(layers.items()):
                for block in blocks:
                    if getattr(block, "state", None) == "CPU_COMPRESSED":
                        try:
                            # Perform the GPU/Metal upload on the main thread
                            gpu_device = block.anchor_kv.device if block.anchor_kv is not None else self.device
                            u_cpu = getattr(block, "U_cpu", None)
                            v_cpu = getattr(block, "V_cpu", None)

                            if u_cpu is None or v_cpu is None:
                                # Still waiting for compressor to populate — skip for now
                                continue

                            if getattr(block, "anchor_kv_cpu", None) is not None:
                                block.anchor_kv = block.anchor_kv_cpu.to(gpu_device)
                                block.anchor_kv_cpu = None

                            block.U = u_cpu.to(gpu_device)
                            block.V = v_cpu.to(gpu_device)
                            block.residual_K_positions = getattr(block, "residual_K_positions", None)
                            if block.residual_K_positions is not None:
                                block.residual_K_positions = block.residual_K_positions.to(gpu_device)
                            block.residual_K_values = getattr(block, "residual_K_values", None)
                            if block.residual_K_values is not None:
                                block.residual_K_values = block.residual_K_values.to(gpu_device)
                            block.residual_V_positions = getattr(block, "residual_V_positions", None)
                            if block.residual_V_positions is not None:
                                block.residual_V_positions = block.residual_V_positions.to(gpu_device)
                            block.residual_V_values = getattr(block, "residual_V_values", None)
                            if block.residual_V_values is not None:
                                block.residual_V_values = block.residual_V_values.to(gpu_device)

                            block.U_sem_int4 = getattr(block, "U_sem_int4_cpu", None)
                            if block.U_sem_int4 is not None:
                                block.U_sem_int4 = block.U_sem_int4.to(gpu_device)
                            block.U_sem_scale = getattr(block, "U_sem_scale_cpu", None)
                            if block.U_sem_scale is not None:
                                block.U_sem_scale = block.U_sem_scale.to(gpu_device)
                            block.U_fact_fp16 = getattr(block, "U_fact_fp16_cpu", None)
                            if block.U_fact_fp16 is not None:
                                block.U_fact_fp16 = block.U_fact_fp16.to(gpu_device)

                            fact_anc_K = getattr(block, "fact_anchors_K_cpu", None)
                            block.fact_anchors_K = fact_anc_K.to(gpu_device) if fact_anc_K is not None else None
                            fact_anc_V = getattr(block, "fact_anchors_V_cpu", None)
                            block.fact_anchors_V = fact_anc_V.to(gpu_device) if fact_anc_V is not None else None
                            fact_anc_pos = getattr(block, "fact_anchor_positions_cpu", None)
                            block.fact_anchor_positions = fact_anc_pos.to(gpu_device) if fact_anc_pos is not None else None

                            # Clean up temporary CPU tensors
                            block.U_cpu = None
                            block.V_cpu = None
                            block.U_sem_int4_cpu = None
                            block.U_sem_scale_cpu = None
                            block.U_fact_fp16_cpu = None
                            block.fact_anchors_K_cpu = None
                            block.fact_anchors_V_cpu = None
                            block.fact_anchor_positions_cpu = None

                            # Write to native pool
                            if hasattr(self, 'native_pool') and self.native_pool is not None:
                                if getattr(block, 'pool_idx', None) is None:
                                    block.pool_idx = self.native_pool.allocate_block()
                                block.pool = self.native_pool
                                self.native_pool.write_block(
                                    pool_idx=block.pool_idx,
                                    U=block.U,
                                    V=block.V,
                                    anchor_K=self._get_rotated_anchor_k(session_id, block.anchor_kv[0, 0], block.anchor_idx),
                                    anchor_V=block.anchor_kv[0, 1],
                                    scale=block.scale,
                                    seq_len=block.U.shape[0],
                                    residual_K_positions=block.residual_K_positions,
                                    residual_K_values=block.residual_K_values,
                                    residual_V_positions=block.residual_V_positions,
                                    residual_V_values=block.residual_V_values,
                                    U_sem_int4=block.U_sem_int4,
                                    U_sem_scale=block.U_sem_scale,
                                    U_fact_fp16=block.U_fact_fp16,
                                    n_semantic=getattr(block, "n_semantic", 0),
                                )
                                # Clear local GPU tensors on block to prevent VRAM leak
                                block.U = None
                                block.V = None
                                block.U_sem_int4 = None
                                block.U_sem_scale = None
                                block.U_fact_fp16 = None
                                block.residual_K_positions = None
                                block.residual_K_values = None
                                block.residual_V_positions = None
                                block.residual_V_values = None

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

        # Log only when genuinely doing upload work (> 2ms) and telemetry is enabled.
        # Consistent < 1ms readings confirm the stall was pure lock overhead (now fixed).
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        if _elapsed_ms > 2.0 and os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
            print(f"[DiffKV] finalize_compressed_blocks: {_elapsed_ms:.1f}ms "
                  f"(was pending={_pending_before})")

    def clear_session(self, session_id: str):
        # Issue 6 fix: Invalidate the gather-KV workspace cache for this session
        # by bumping routing_version BEFORE freeing blocks.  Any in-flight decode
        # step that checks cached_val[0] == current_version will see a mismatch
        # and re-gather fresh tensors rather than using stale ones from freed slots.
        if hasattr(self, 'decode_workspace') and session_id in self.decode_workspace:
            ws = self.decode_workspace[session_id]
            if isinstance(ws, dict):
                ws["routing_version"] = ws.get("routing_version", 0) + 1
        # Now fully clear the workspace entry
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
        self._factual_stores.pop(session_id, None)
        self._session_token_ids.pop(session_id, None)
        # Per-session content caches keyed by block anchor.  These are derived
        # from _session_token_ids, so they are stale the moment it is dropped.
        for _cache_attr in ("_block_rank_cache", "_res_capture_boost_rows"):
            _cache = getattr(self, _cache_attr, None)
            if _cache is not None:
                _cache.pop(session_id, None)
        if hasattr(self, "_prefill_kv_capture"):
            self._prefill_kv_capture.pop(session_id, None)
        if hasattr(self, "attention_score_cache"):
            self.attention_score_cache.clear_session(session_id)
        if hasattr(self, "_last_prefill_q"):
            self._last_prefill_q.pop(session_id, None)

        # Trigger garbage collection and empty MPS/CUDA cache
        import gc
        gc.collect()
        _empty_cache(self.device)

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
                b_snap.pool = None
                
                # Clone active GPU buffers and move to CPU
                if getattr(b, "_active_buf_k", None) is not None:
                    b_snap._active_buf_k = b._active_buf_k.clone().cpu()
                    fill = getattr(b, "_active_fill", 0)
                    if fill > 0:
                        b_snap.active_k = b_snap._active_buf_k[:, :, :fill, :]
                    else:
                        b_snap.active_k = None
                elif getattr(b, "active_k", None) is not None:
                    b_snap.active_k = b.active_k.clone().cpu()

                if getattr(b, "_active_buf_v", None) is not None:
                    b_snap._active_buf_v = b._active_buf_v.clone().cpu()
                    fill = getattr(b, "_active_fill", 0)
                    if fill > 0:
                        b_snap.active_v = b_snap._active_buf_v[:, :, :fill, :]
                    else:
                        b_snap.active_v = None
                elif getattr(b, "active_v", None) is not None:
                    b_snap.active_v = b.active_v.clone().cpu()

                # Clone CPU-pinned uncompressed caches
                if getattr(b, "active_k_cpu", None) is not None:
                    b_snap.active_k_cpu = b.active_k_cpu.clone().cpu()
                if getattr(b, "active_v_cpu", None) is not None:
                    b_snap.active_v_cpu = b.active_v_cpu.clone().cpu()

                if getattr(b, "anchor_kv", None) is not None:
                    b_snap.anchor_kv = b.anchor_kv.clone().cpu()
                if getattr(b, "anchor_kv_cpu", None) is not None:
                    b_snap.anchor_kv_cpu = b.anchor_kv_cpu.clone().cpu()

                # Fetch U and V from pool to keep them in snapshot as CPU tensors
                if getattr(b, "pool_idx", None) is not None and getattr(self, "native_pool", None) is not None:
                    b.pool = self.native_pool
                    u_gpu = b.U
                    v_gpu = b.V
                    if u_gpu is not None:
                        b_snap._U = u_gpu.cpu()
                    if v_gpu is not None:
                        b_snap._V = v_gpu.cpu()
                    # Mark pool_idx as None so it gets re-allocated upon restore
                    b_snap.pool_idx = None

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
                device = self.device
                if getattr(b, "_active_buf_k", None) is not None:
                    b_restore._active_buf_k = b._active_buf_k.clone().to(device)
                    fill = getattr(b, "_active_fill", 0)
                    if fill > 0:
                        b_restore.active_k = b_restore._active_buf_k[:, :, :fill, :]
                    else:
                        b_restore.active_k = None
                elif getattr(b, "active_k", None) is not None:
                    b_restore.active_k = b.active_k.clone().to(device)

                if getattr(b, "_active_buf_v", None) is not None:
                    b_restore._active_buf_v = b._active_buf_v.clone().to(device)
                    fill = getattr(b, "_active_fill", 0)
                    if fill > 0:
                        b_restore.active_v = b_restore._active_buf_v[:, :, :fill, :]
                    else:
                        b_restore.active_v = None
                elif getattr(b, "active_v", None) is not None:
                    b_restore.active_v = b.active_v.clone().to(device)

                # Clone CPU-pinned uncompressed caches
                if getattr(b, "active_k_cpu", None) is not None:
                    b_restore.active_k_cpu = b.active_k_cpu.clone()
                if getattr(b, "active_v_cpu", None) is not None:
                    b_restore.active_v_cpu = b.active_v_cpu.clone()

                if getattr(b, "anchor_kv", None) is not None:
                    b_restore.anchor_kv = b.anchor_kv.clone().to(device)
                if getattr(b, "anchor_kv_cpu", None) is not None:
                    b_restore.anchor_kv_cpu = b.anchor_kv_cpu.clone()

                # Re-allocate slots in pool for compressed blocks and write CPU tensors back to pool
                if getattr(b, "_U", None) is not None and getattr(self, "native_pool", None) is not None:
                    b_restore.pool_idx = self.native_pool.allocate_block()
                    b_restore.pool = self.native_pool
                    u_gpu = b._U.to(device)
                    v_gpu = b._V.to(device)
                    self.native_pool.write_block(
                        pool_idx=b_restore.pool_idx,
                        U=u_gpu,
                        V=v_gpu,
                        anchor_K=self._get_rotated_anchor_k(session_id, b_restore.anchor_kv[0, 0], b_restore.anchor_idx),
                        anchor_V=b_restore.anchor_kv[0, 1],
                        scale=b_restore.scale,
                        seq_len=u_gpu.shape[0],
                        residual_K_positions=getattr(b, "residual_K_positions", None),
                        residual_K_values=getattr(b, "residual_K_values", None),
                        residual_V_positions=getattr(b, "residual_V_positions", None),
                        residual_V_values=getattr(b, "residual_V_values", None),
                        U_sem_int4=getattr(b, "U_sem_int4", None),
                        U_sem_scale=getattr(b, "U_sem_scale", None),
                        U_fact_fp16=getattr(b, "U_fact_fp16", None),
                        n_semantic=getattr(b, "n_semantic", 0),
                        fact_anchors_K=getattr(b, "fact_anchors_K", None),
                        fact_anchors_V=getattr(b, "fact_anchors_V", None),
                        fact_anchor_positions=getattr(b, "fact_anchor_positions", None),
                    )
                    # Clear local GPU tensors on block again to prevent VRAM leak
                    b_restore._U = None
                    b_restore._V = None

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
                        # Issue 6 fix: invalidate any cached gather-KV tensors that
                        # reference this checkpoint's pool slots before freeing.
                        session_id = checkpoint.get("session_id") or checkpoint_id
                        if hasattr(self, 'decode_workspace') and session_id in self.decode_workspace:
                            ws = self.decode_workspace[session_id]
                            if isinstance(ws, dict):
                                ws["routing_version"] = ws.get("routing_version", 0) + 1
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
        - Streams blocks during ingest; prefill compression is published after the
          final chunk, while decode still compresses eligible blocks incrementally.
        - Dense footprint is bounded during decode; prefill retains raw history for
          exact causal attention.
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
            # CRITICAL SRL ALIGNMENT FIX: Capture decode token K/V states.
            # Guard: only accumulate when the SRL index hasn't been finalized yet,
            # AND when SRL is actually enabled (W_proj is not None).
            # When W_proj is None, finalize_srl_index() returns early without
            # populating _session_srl, so we must also check W_proj here to
            # prevent O(N²) torch.cat+.cpu() on every decode token × all layers.
            _pool = getattr(self, "native_pool", None)
            _srl_active = (_pool is not None and getattr(_pool, "W_proj", None) is not None)
            if _srl_active and session_id not in self._session_srl:
                if not hasattr(self, "_prefill_kv_capture"):
                    self._prefill_kv_capture = {}
                session_cap = self._prefill_kv_capture.setdefault(session_id, {})
                if layer_idx not in session_cap:
                    session_cap[layer_idx] = [k.clone().cpu(), v.clone().cpu()]
                else:
                    session_cap[layer_idx][0] = torch.cat([session_cap[layer_idx][0], k.cpu()], dim=2)
                    session_cap[layer_idx][1] = torch.cat([session_cap[layer_idx][1], v.cpu()], dim=2)
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
            cpu_indices_key = tuple(cpu_indices.tolist())
            if cached_val is not None:
                cached_key, cached_gpu_ind, cached_gpu_anc = cached_val
                if cached_key == cpu_indices_key:
                    block_indices_tensor = cached_gpu_ind
                    anchor_indices_gpu = cached_gpu_anc
                else:
                    block_indices_tensor = cpu_indices.to(device)
                    anchor_indices_gpu = cpu_anchors.to(device)
                    indices_gpu_cache[layer_idx] = (cpu_indices_key, block_indices_tensor, anchor_indices_gpu)
            else:
                block_indices_tensor = cpu_indices.to(device)
                anchor_indices_gpu = cpu_anchors.to(device)
                indices_gpu_cache[layer_idx] = (cpu_indices_key, block_indices_tensor, anchor_indices_gpu)
        else:
            block_indices_tensor = None
            anchor_indices_gpu = None

        # Get non-compressed, non-paged, non-submitted blocks as dense context.
        # SUBMITTED blocks are excluded: they are partial blocks awaiting async CPU
        # compression. They have no pool_idx so cannot be served by the sparse kernel,
        # but including them as dense overflows max_dense_len for long contexts (16 dense
        # blocks >> workspace of ~5 blocks), triggering trim that drops block 0 and
        # causes the model to output EOS. Dropping them temporarily is safe: neighboring
        # full GPU-compressed blocks cover the same context region via sparse attention,
        # and SUBMITTED blocks will be promoted to COMPRESSED shortly after decode starts.
        dense_blocks = [block for block in blocks if block.state == "ACCUMULATING"]

        return block_indices_tensor, dense_blocks, anchor_indices_gpu, max_anchor_idx, max_valid_len

    def assemble_dense_window_kv(
        self,
        session_id: str,
        layer_idx: int,
        dense_blocks: list,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Lightweight dense-window-only KV assembler for the decode hot path.

        ONLY processes non-compressed (ACCUMULATING / SUBMITTED) blocks.
        Compressed blocks are handled by block_indices in the sparse kernel.

        Returns (workspace_k, workspace_v, L_dense) where workspace_k/v always have
        shape [1, kv_heads, max_dense_len, head_dim] regardless of how many tokens
        are valid this step.  L_dense is the count of valid tokens (Python int).
        Callers mask positions >= L_dense with -inf before softmax.
        """
        if not dense_blocks:
            return None, None, 0

        # 1. Compute per-block sizes, then trim oldest blocks if total > max_dense_len.
        # During chunked prefill, blk.active_k.shape[2] can reach 2*block_size - 1, so
        # the total may occasionally exceed the workspace.  We preserve recency by
        # dropping the OLDEST blocks (front of the list) until the content fits.
        def _blk_len(blk):
            a = (blk.active_k.shape[2] if blk.active_k is not None
                 else (blk.active_k_cpu.shape[2] if getattr(blk, "active_k_cpu", None) is not None else 0))
            return 1 + a  # anchor + active

        blk_sizes = [_blk_len(b) for b in dense_blocks]
        L_dense   = sum(blk_sizes)

        if L_dense > self.max_dense_len:
            # Drop oldest blocks until L_dense fits.  Emit a one-time warning so it's visible.
            if not getattr(self, "_dense_trim_warned", False):
                print(
                    f"[DiffKV] WARNING: dense window ({L_dense} tokens) exceeds workspace "
                    f"({self.max_dense_len}).  Trimming oldest blocks to fit.  "
                    f"Set DIFFKV_MAX_DENSE_LEN={L_dense + 64} to suppress.",
                    flush=True,
                )
                self._dense_trim_warned = True
            while blk_sizes and L_dense > self.max_dense_len:
                # Never drop block 0 (anchor_idx == 0): losing the system prompt / question
                # causes the model to output EOS immediately.
                if len(dense_blocks) > 1 and dense_blocks[0].anchor_idx == 0:
                    # Protect block 0 — try dropping the second-oldest instead
                    if len(dense_blocks) > 2:
                        L_dense -= blk_sizes.pop(1)
                        dense_blocks = [dense_blocks[0]] + dense_blocks[2:]
                    else:
                        break  # only block 0 and one other left — stop trimming
                else:
                    L_dense -= blk_sizes.pop(0)
                    dense_blocks = dense_blocks[1:]
            if not dense_blocks:
                return None, None, 0

        # 2. Retrieve or allocate workspaces.
        # Static-shape invariant: always allocate to self.max_dense_len (= recency_window +
        # block_size) so the returned tensor always has the same shape regardless of how many
        # dense tokens are actually valid this step.  The caller masks out positions >= L_dense
        # before softmax so padding zeros contribute nothing to attention output.
        # This makes tensor shapes inside the decode forward pass completely static, enabling
        # the CUDAGraphDecodeRunner to capture the forward once and replay it every step.
        session_dict = self.decode_workspace.setdefault(session_id, {})
        dense_k_cache = session_dict.setdefault("dense_workspace_k", {})
        dense_v_cache = session_dict.setdefault("dense_workspace_v", {})
        workspace_k = dense_k_cache.get(layer_idx)
        workspace_v = dense_v_cache.get(layer_idx)

        dense_start_pos_dict = session_dict.setdefault("dense_start_pos", {})
        last_start_pos = dense_start_pos_dict.get(layer_idx)
        start_pos = dense_blocks[0].anchor_idx

        new_alloc = (workspace_k is None
            or workspace_k.shape[1] != self.kv_heads
            or workspace_k.dtype != dtype
            or last_start_pos is None
            or last_start_pos != start_pos)

        if new_alloc:
            if workspace_k is None or workspace_k.shape[1] != self.kv_heads or workspace_k.dtype != dtype:
                # Allocate once to the session-level cap — shape never changes after this.
                workspace_k = torch.zeros((1, self.kv_heads, self.max_dense_len, self.head_dim), device=self.device, dtype=dtype)
                workspace_v = torch.zeros((1, self.kv_heads, self.max_dense_len, self.head_dim), device=self.device, dtype=dtype)
                dense_k_cache[layer_idx] = workspace_k
                dense_v_cache[layer_idx] = workspace_v
            dense_start_pos_dict[layer_idx] = start_pos

        dense_offsets = session_dict.setdefault("dense_offsets", {})

        # Fix 1: Detect block growth between decode steps.
        # dense_offsets stores (offset, expected_active_len) per (layer_idx, anchor_idx).
        # If any block has grown (e.g. 26→27 tokens since last alloc), the old offset's
        # slot is too small — force new_alloc=True so we recompute all offsets and copy
        # all blocks into a freshly-laid-out workspace.  This prevents the
        #   RuntimeError: tensor a (26) must match tensor b (27)
        # crash that killed responses mid-generation in long chat sessions.
        if not new_alloc:
            for blk in dense_blocks:
                key = (layer_idx, blk.anchor_idx)
                cached_entry = dense_offsets.get(key)
                if cached_entry is not None and isinstance(cached_entry, tuple):
                    _cached_offset, _cached_alen = cached_entry
                    cur_alen = blk.active_k.shape[2] if blk.active_k is not None else (
                        blk.active_k_cpu.shape[2] if getattr(blk, "active_k_cpu", None) is not None else 0
                    )
                    if cur_alen != _cached_alen:
                        # Block grew — invalidate workspace layout, force full rewrite
                        new_alloc = True
                        # Workspace is always max_dense_len; no reallocation needed.
                        dense_start_pos_dict[layer_idx] = start_pos
                        break

        # 3. Copy only dirty blocks directly into the pre-allocated workspace slices.
        # Safety net: if a single block's active_k still exceeds the remaining workspace
        # (should not happen after trimming above, but guarded defensively), clip it.
        curr_idx = 0
        for blk in dense_blocks:
            if curr_idx >= self.max_dense_len:
                break  # workspace full — skip remaining blocks
            key = (layer_idx, blk.anchor_idx)
            if new_alloc:
                workspace_k[:, :, curr_idx : curr_idx + 1].copy_(blk.anchor_kv[:, 0].unsqueeze(2), non_blocking=True)
                workspace_v[:, :, curr_idx : curr_idx + 1].copy_(blk.anchor_kv[:, 1].unsqueeze(2), non_blocking=True)

                if blk.active_k is not None:
                    active_len = min(blk.active_k.shape[2], self.max_dense_len - curr_idx - 1)
                    if active_len > 0:
                        workspace_k[:, :, curr_idx + 1 : curr_idx + 1 + active_len].copy_(blk.active_k[:, :, :active_len], non_blocking=True)
                        workspace_v[:, :, curr_idx + 1 : curr_idx + 1 + active_len].copy_(blk.active_v[:, :, :active_len], non_blocking=True)
                    dense_offsets[key] = (curr_idx, active_len)
                    curr_idx += 1 + active_len
                elif getattr(blk, "active_k_cpu", None) is not None:
                    active_len = min(blk.active_k_cpu.shape[2], self.max_dense_len - curr_idx - 1)
                    if active_len > 0:
                        workspace_k[:, :, curr_idx + 1 : curr_idx + 1 + active_len].copy_(blk.active_k_cpu[:, :, :active_len], non_blocking=True)
                        workspace_v[:, :, curr_idx + 1 : curr_idx + 1 + active_len].copy_(blk.active_v_cpu[:, :, :active_len], non_blocking=True)
                    dense_offsets[key] = (curr_idx, active_len)
                    curr_idx += 1 + active_len
                else:
                    dense_offsets[key] = (curr_idx, 0)  # anchor only
                    curr_idx += 1

                blk.dirty = False

            else:
                cached_entry = dense_offsets.get(key)
                if cached_entry is not None and isinstance(cached_entry, tuple):
                    offset = cached_entry[0]
                elif isinstance(cached_entry, int):
                    offset = cached_entry
                else:
                    offset = curr_idx
                    dense_offsets[key] = (offset, 0)

                if blk.dirty:
                    workspace_k[:, :, offset : offset + 1].copy_(blk.anchor_kv[:, 0].unsqueeze(2), non_blocking=True)
                    workspace_v[:, :, offset : offset + 1].copy_(blk.anchor_kv[:, 1].unsqueeze(2), non_blocking=True)

                    if blk.active_k is not None:
                        active_len = min(blk.active_k.shape[2], self.max_dense_len - offset - 1)
                        if active_len > 0:
                            workspace_k[:, :, offset + 1 : offset + 1 + active_len].copy_(blk.active_k[:, :, :active_len], non_blocking=True)
                            workspace_v[:, :, offset + 1 : offset + 1 + active_len].copy_(blk.active_v[:, :, :active_len], non_blocking=True)
                        dense_offsets[key] = (offset, active_len)
                    elif getattr(blk, "active_k_cpu", None) is not None:
                        active_len = min(blk.active_k_cpu.shape[2], self.max_dense_len - offset - 1)
                        if active_len > 0:
                            workspace_k[:, :, offset + 1 : offset + 1 + active_len].copy_(blk.active_k_cpu[:, :, :active_len], non_blocking=True)
                            workspace_v[:, :, offset + 1 : offset + 1 + active_len].copy_(blk.active_v_cpu[:, :, :active_len], non_blocking=True)
                        dense_offsets[key] = (offset, active_len)
                    else:
                        dense_offsets[key] = (offset, 0)

                    blk.dirty = False

                if blk.active_k is not None:
                    curr_idx = min(offset + 1 + blk.active_k.shape[2], self.max_dense_len)
                elif getattr(blk, "active_k_cpu", None) is not None:
                    curr_idx = min(offset + 1 + blk.active_k_cpu.shape[2], self.max_dense_len)
                else:
                    curr_idx = min(offset + 1, self.max_dense_len)


        # 4. Return the FULL fixed-size workspace + L_dense as a scalar + trimmed block list.
        # Shape is always [1, kv_heads, max_dense_len, head_dim] — static across every
        # decode step.  Positions >= L_dense contain stale/zero data; the caller masks
        # those positions with -inf before softmax so they get exactly 0 attention weight.
        # dense_blocks may have been trimmed (oldest dropped); return the surviving list so
        # the caller's position-tensor loop uses the same trimmed set.
        return workspace_k, workspace_v, L_dense, dense_blocks

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
        blocks = self.get_streaming_blocks(session_id, layer_idx)
        if not blocks:
            return 0
        total = 0
        for block in blocks:
            total += 1  # anchor
            if getattr(block, "token_indices", None) is not None and len(block.token_indices) > 0:
                total += len(block.token_indices) - 1
            elif getattr(block, "pool_idx", None) is not None and getattr(block, "pool", None) is not None:
                total += int(block.pool.seq_lens[block.pool_idx].item())
            elif getattr(block, "_U", None) is not None:
                total += block._U.shape[0]
            elif block.U is not None:
                total += block.U.shape[0]
            elif block.active_k is not None:
                total += block.active_k.shape[2]
            elif getattr(block, "active_k_cpu", None) is not None:
                total += block.active_k_cpu.shape[2]
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
                anchor_flat = block.anchor_kv.reshape(-1).to(torch.float16)
                recon = TritonDiffKV.reconstruct_lowrank(
                    block.U, block.V, anchor_flat, scale=block.scale
                )
                hds  = block.anchor_kv.shape[2]
                hdim = block.anchor_kv.shape[3]
                recon = recon.view(1, -1, 2, hds, hdim)
                recon_k = recon[:, :, 0].transpose(1, 2).contiguous().clone()
                recon_v = recon[:, :, 1].transpose(1, 2).contiguous().clone()
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
        block.anchor_kv_cpu = block.anchor_kv.cpu() if block.anchor_kv is not None else None
        if self._async:
            self._compressor.submit(block, k, v)
        else:
            self._compress_block_sync(block, k, v)

    def _preprocess_block_for_compression(self, block: KVBlock, k: torch.Tensor, v: torch.Tensor):
        if getattr(block, "skip_compression", False):
            return None
        input_device = k.device
        anchor_kv_local = block.anchor_kv
        if anchor_kv_local is not None:
            anchor_kv_local = anchor_kv_local.to(input_device)
        else:
            anchor_kv_local = getattr(block, "anchor_kv_cpu", None)
            if anchor_kv_local is not None:
                anchor_kv_local = anchor_kv_local.to(input_device)
                # Restore to block.anchor_kv if on main thread
                is_background = threading.current_thread().name.startswith("DiffKV-Compressor")
                if not is_background:
                    block.anchor_kv = anchor_kv_local
        if anchor_kv_local is None:
            print(f"[DiffKV DEBUG] _compress_block_sync: anchor_kv_local is None! block={getattr(block, 'anchor_idx', 'unknown')}, skip={getattr(block, 'skip_compression', 'unknown')}, anchor_kv={getattr(block, 'anchor_kv', 'unknown')}, anchor_kv_cpu={getattr(block, 'anchor_kv_cpu', 'unknown')}")

        # ── Learned Landmark Scoring ──
        # Form full block including the old anchor
        full_k = torch.cat([anchor_kv_local[:, 0].unsqueeze(2), k], dim=2)
        full_v = torch.cat([anchor_kv_local[:, 1].unsqueeze(2), v], dim=2)
        S_total = full_k.shape[2]

        # ── PER-TOKEN ROTATION FIX (gated: DIFFKV_ROTATE_AT_INGEST) ────────────────────────────────
        # Rotate each token t in full_k by its within-block offset t (K only)
        has_rope = True
        if has_rope:
            device = full_k.device
            dtype = full_k.dtype
            D = full_k.shape[3]
            half_d = D // 2
            
            rope_theta = 10000.0
            if hasattr(self, "model") and self.model is not None:
                rope_theta = getattr(self.model.config, "rope_theta", 10000.0)
            if "qwen" in str(getattr(self, "model_id", "")).lower():
                rope_theta = 1000000.0
            
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
            t_coords = torch.arange(S_total, device=device, dtype=torch.float32)
            angles = t_coords.unsqueeze(1) * inv_freq.unsqueeze(0)
            cos_a = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(1)
            sin_a = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(1)
            
            k_half = torch.zeros_like(full_k)
            k_half[..., :half_d] = -full_k[..., half_d:]
            k_half[..., half_d:] = full_k[..., :half_d]
            
            cos_a_full = torch.cat([cos_a, cos_a], dim=-1)
            sin_a_full = torch.cat([sin_a, sin_a], dim=-1)
            
            full_k = full_k * cos_a_full + k_half * sin_a_full

        session_id = getattr(block, "session_id", None)

        block_token_ids = []
        if session_id is not None and session_id in self._session_token_ids:
            all_tids = self._session_token_ids[session_id]
            for pos in getattr(block, "token_indices", []):
                if 0 <= pos < len(all_tids):
                    block_token_ids.append(int(all_tids[pos].item()))

        scores = torch.zeros(S_total, device=full_k.device)
        
        # 1. Key norm / activation magnitude (saliency)
        k_norms = full_k[0].norm(dim=-1).mean(dim=0)  # [S_total]
        scores = scores + k_norms
        
        # 2. Semantic Centrality (graph centrality equivalent within block)
        K_f = full_k[0].permute(1, 0, 2).reshape(S_total, -1).float()  # [S_total, heads * head_dim]
        K_f = K_f / (K_f.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = K_f @ K_f.T  # [S_total, S_total]
        centrality = sim_matrix.sum(dim=-1)
        scores = scores + centrality.to(scores.device) * 0.5
        
        # 3. Numeric / Entity / Stopword priority
        stop_words = getattr(self, "_stop_token_ids", set())
        for idx in range(min(S_total, len(block_token_ids))):
            tid = block_token_ids[idx]
            if tid not in stop_words:
                scores[idx] += 2.0  # content word boost
            if 48 <= tid <= 57:  # basic ASCII digit check
                scores[idx] += 3.0

        landmark_idx = int(scores.argmax().item())
        
        if landmark_idx > 0:
            # Swap token at index 0 (old anchor) and landmark_idx in full_k and full_v
            tmp_k = full_k[:, :, 0].clone()
            tmp_v = full_v[:, :, 0].clone()
            full_k[:, :, 0] = full_k[:, :, landmark_idx]
            full_v[:, :, 0] = full_v[:, :, landmark_idx]
            full_k[:, :, landmark_idx] = tmp_k
            full_v[:, :, landmark_idx] = tmp_v
            
            # Update local anchor
            anchor_kv_local = torch.stack([full_k[:, :, 0], full_v[:, :, 0]], dim=1)
            is_background = threading.current_thread().name.startswith("DiffKV-Compressor")
            if input_device.type == "cpu" or is_background:
                block.anchor_kv_cpu = anchor_kv_local
                if not is_background:
                    gpu_dev = block.anchor_kv.device if block.anchor_kv is not None else self.device
                    block.anchor_kv = anchor_kv_local.to(gpu_dev)
            else:
                block.anchor_kv = anchor_kv_local
            
            # Update active tokens for SVD
            k = full_k[:, :, 1:]
            v = full_v[:, :, 1:]
        else:
            # Update local anchor with the unrotated version of the original anchor
            anchor_kv_local = torch.stack([full_k[:, :, 0], full_v[:, :, 0]], dim=1)
            is_background = threading.current_thread().name.startswith("DiffKV-Compressor")
            if input_device.type == "cpu" or is_background:
                block.anchor_kv_cpu = anchor_kv_local
                if not is_background:
                    gpu_dev = block.anchor_kv.device if block.anchor_kv is not None else self.device
                    block.anchor_kv = anchor_kv_local.to(gpu_dev)
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

        # Per-layer rank with early-layer boost
        _cfg = getattr(self, "config", None)
        _early_boost     = getattr(_cfg, "early_layer_rank_boost", False)
        _max_rank_early  = getattr(_cfg, "max_rank_early", 0)
        _layer_idx_safe  = getattr(block, "layer_idx", 0)
        if _layer_idx_safe is None:
            _layer_idx_safe = 0
        rank = get_layer_rank(
            _layer_idx_safe, self.num_layers, self.rank,
            early_boost=_early_boost, max_rank_early=_max_rank_early,
        )

        # Check if block qualifies for rank boosting (1.5x)
        boost_rank = False
        if block_token_ids and getattr(self, "tokenizer", None) is not None:
            try:
                block_text = self.tokenizer.decode(block_token_ids)
                # 1. Any digit
                if any(c.isdigit() for c in block_text):
                    boost_rank = True
                else:
                    # 2. Math formula markers
                    import re
                    re_math_boost = re.compile(
                        r'[\+\-\*\/=]|\$\$|\\\[|\\\(|\\begin\{|\\alpha|\\beta|\\gamma|\\delta|\\sum|\\int|\\frac|\\sqrt|_\{|\^'
                    )
                    if re_math_boost.search(block_text):
                        boost_rank = True
                    else:
                        # 3. Key definition keywords
                        re_definitions_boost = re.compile(
                            r'\b(?:is|are|we)\s+(?:defined|referred|called|known)\s+(?:as|by)\b|\brefers?\s+to\b|\b(?:denotes?|stands\s+for|represents?)\b|\bwe\s+define\b|\b(?:let\s+us|let)\s+define\b',
                            re.IGNORECASE
                        )
                        if re_definitions_boost.search(block_text):
                            boost_rank = True
            except Exception:
                pass

        if boost_rank:
            import math
            rank = int(math.ceil(rank * 1.5))
            if rank > seq_len:
                rank = seq_len

        return normalized_deltas, token_norms, deltas, rank, block_token_ids, k, v, anchor_kv_local

    def _postprocess_compressed_block(self, block: KVBlock, U_scaled: torch.Tensor, V: torch.Tensor,
                                     scale: float, cosine_sim: float, norm_drift: float, dynamic_rank: int,
                                     fact_anchors_K_val: torch.Tensor, fact_anchors_V_val: torch.Tensor,
                                     fact_anchor_positions_val: torch.Tensor,
                                     k_orig: torch.Tensor, v_orig: torch.Tensor, anchor_kv_local: torch.Tensor, rank: int):
        is_background = threading.current_thread().name.startswith("DiffKV-Compressor")

        if is_background:
            # CPU-only background SVD to guarantee thread safety
            block.U_cpu          = U_scaled.cpu()
            block.V_cpu          = V.to(torch.float16).cpu()
            block.scale          = scale
            block.cosine_sim     = cosine_sim
            block.norm_drift     = norm_drift
            block.dynamic_rank   = dynamic_rank

            # Solution 2 CPU components
            block.U_sem_int4_cpu = None
            block.U_sem_scale_cpu = None
            block.U_fact_fp16_cpu = None
            block.n_semantic     = 0

            # Solution 3 CPU components
            block.fact_anchors_K_cpu = fact_anchors_K_val.cpu()
            block.fact_anchors_V_cpu = fact_anchors_V_val.cpu()
            block.fact_anchor_positions_cpu = fact_anchor_positions_val.cpu()

            block.dirty    = True

            if hasattr(block, 'state'):
                block.state = "CPU_COMPRESSED"
        else:
            gpu_device = block.anchor_kv.device if block.anchor_kv is not None else self.device
            u_gpu = U_scaled.to(gpu_device)
            v_gpu = V.to(torch.float16).to(gpu_device)

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
                    block.scale      = scale
                    block.cosine_sim = cosine_sim
                    block.norm_drift = norm_drift
                    block.dynamic_rank = dynamic_rank

                    # Solution 2 components
                    block.U_sem_int4 = None
                    block.U_sem_scale = None
                    block.U_fact_fp16 = None
                    block.n_semantic     = 0

                    # Solution 3 components
                    block.fact_anchors_K = fact_anchors_K_val.to(gpu_device)
                    block.fact_anchors_V = fact_anchors_V_val.to(gpu_device)
                    block.fact_anchor_positions = fact_anchor_positions_val.to(gpu_device)

                    block.active_k = None
                    block.active_v = None
                    block.active_k_cpu = None
                    block.active_v_cpu = None
                    block._active_buf_k = None
                    block._active_buf_v = None
                    block.dirty    = True

                    if hasattr(block, 'state'):
                        block.state = "COMPRESSED"
            else:
                block.U          = u_gpu
                block.V          = v_gpu
                block.scale      = scale
                block.cosine_sim = cosine_sim
                block.norm_drift = norm_drift
                block.dynamic_rank = dynamic_rank

                # Solution 2 components
                block.U_sem_int4 = None
                block.U_sem_scale = None
                block.U_fact_fp16 = None
                block.n_semantic     = 0

                # Solution 3 components
                block.fact_anchors_K = fact_anchors_K_val.to(gpu_device)
                block.fact_anchors_V = fact_anchors_V_val.to(gpu_device)
                block.fact_anchor_positions = fact_anchor_positions_val.to(gpu_device)

                block.active_k = None
                block.active_v = None
                block.active_k_cpu = None
                block.active_v_cpu = None
                block._active_buf_k = None
                block._active_buf_v = None
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
                        block.pool = self.native_pool
                        self.native_pool.write_block(
                            pool_idx=block.pool_idx,
                            U=block.U,
                            V=block.V,
                            anchor_K=self._get_rotated_anchor_k(session_id, anchor_kv_local[0, 0], block.anchor_idx),
                            anchor_V=anchor_kv_local[0, 1],
                            scale=block.scale,
                            seq_len=block.U.shape[0],
                            residual_K_positions=block.residual_K_positions,
                            residual_K_values=block.residual_K_values,
                            residual_V_positions=block.residual_V_positions,
                            residual_V_values=block.residual_V_values,
                            U_sem_int4=block.U_sem_int4,
                            U_sem_scale=block.U_sem_scale,
                            U_fact_fp16=block.U_fact_fp16,
                            n_semantic=getattr(block, 'n_semantic', 0),
                            fact_anchors_K=block.fact_anchors_K,
                            fact_anchors_V=block.fact_anchors_V,
                            fact_anchor_positions=block.fact_anchor_positions,
                        )
                        # Clear local GPU tensors on block to prevent VRAM leak
                        block.U = None
                        block.V = None
                        block.anchor_kv = None
                        block.U_sem_int4 = None
                        block.U_sem_scale = None
                        block.U_fact_fp16 = None
                        block.residual_K_positions = None
                        block.residual_K_values = None
                        block.residual_V_positions = None
                        block.residual_V_values = None
                        block.fact_anchors_K = None
                        block.fact_anchors_V = None
                        block.fact_anchor_positions = None
                    except Exception as e:
                        print(f"[DiffKV] WARNING: Failed to write block to NativeBlockPool: {e}")

                if self._streaming_mgr is not None and getattr(block, 'session_id', None) is not None and getattr(block, 'layer_idx', None) is not None:
                    self._streaming_mgr.update_metadata_state(block.session_id, block.layer_idx, block)

        self.total_compressions += 1
        self.total_cosine_sim   += cosine_sim
        self.total_norm_drift   += norm_drift
        self.rank_histogram[rank] = self.rank_histogram.get(rank, 0) + 1

        seq_len = U_scaled.shape[0]
        feat_dim = V.shape[1]
        fp16_bytes = seq_len * feat_dim * 2
        lr_bytes   = U_scaled.numel() * 2 + V.numel() * 2
        self.vram_saved_bytes += (fp16_bytes - lr_bytes)

    def _compress_block_sync(self, block: KVBlock, k: torch.Tensor, v: torch.Tensor):
        """Synchronous SVD compression helper (Sequential fallback)."""
        res = self._preprocess_block_for_compression(block, k, v)
        if res is None:
            return
            
        normalized_deltas, token_norms, deltas, rank, block_token_ids, k_orig, v_orig, anchor_kv_local = res
        
        lr_delta = compress_lowrank(normalized_deltas, rank)
        
        # Populate SVD residuals on block
        block.residual_K_positions = lr_delta.residual_K_positions
        block.residual_K_values = lr_delta.residual_K_values
        block.residual_V_positions = lr_delta.residual_V_positions
        block.residual_V_values = lr_delta.residual_V_values
        
        U_scaled = lr_delta.U.float() * token_norms.unsqueeze(1)
        U_scaled = U_scaled.to(torch.float16)
        if not torch.isfinite(U_scaled).all():
            U_scaled = torch.nan_to_num(U_scaled, nan=0.0, posinf=65504.0, neginf=-65504.0)

        recon_deltas = (lr_delta.U.float() @ lr_delta.V.float()) * lr_delta.scale
        recon_errors = (deltas - recon_deltas.to(deltas.device)).norm(dim=1)
        
        seq_len = k_orig.shape[2]
        top_k_val = min(3, seq_len)
        heads = k_orig.shape[1]
        head_dim = k_orig.shape[3]
        fact_anchors_K_val = torch.zeros((3, heads, head_dim), device=k_orig.device, dtype=k_orig.dtype)
        fact_anchors_V_val = torch.zeros((3, heads, head_dim), device=v_orig.device, dtype=v_orig.dtype)
        fact_anchor_positions_val = torch.full((3,), -1, device=k_orig.device, dtype=torch.int16)
        
        if top_k_val > 0:
            top_3 = torch.topk(recon_errors, k=top_k_val)
            for j, pos_val in enumerate(top_3.indices):
                fact_anchors_K_val[j] = k_orig[0, :, pos_val, :]
                fact_anchors_V_val[j] = v_orig[0, :, pos_val, :]
                fact_anchor_positions_val[j] = pos_val.to(torch.int16)

        self._postprocess_compressed_block(
            block, U_scaled, lr_delta.V, lr_delta.scale, lr_delta.cosine_sim, lr_delta.norm_drift,
            getattr(lr_delta, "dynamic_rank", self.rank),
            fact_anchors_K_val, fact_anchors_V_val, fact_anchor_positions_val,
            k_orig, v_orig, anchor_kv_local, rank
        )

    def _compress_blocks_batch(self, items):
        """Batch compression using batched Randomized SVD (C3)"""
        preprocessed = []
        for block, k_cpu, v_cpu, event in items:
            try:
                if event is not None:
                    event.synchronize()
                
                res = self._preprocess_block_for_compression(block, k_cpu, v_cpu)
                if res is not None:
                    preprocessed.append((block, *res))
            except Exception as e:
                print(f"[AsyncCompressor] Preprocess failed for block anchor={getattr(block, 'anchor_idx', '?')}: {e}")
                self._compressor._adjust_pending(-1)

        if not preprocessed:
            return

        deltas_list = [res[1] for res in preprocessed]
        max_rank = max(res[4] for res in preprocessed)
        
        deltas_batch = torch.stack(deltas_list, dim=0)
        
        from native_core.compression.lowrank import compress_lowrank_batch
        U_batch, S_batch, Vh_batch, scale_batch = compress_lowrank_batch(deltas_batch, max_rank)
        
        for i, (block, normalized_deltas, token_norms, deltas_raw, block_rank, block_token_ids, k_orig, v_orig, anchor_kv_local) in enumerate(preprocessed):
            try:
                U_i = U_batch[i]
                S_i = S_batch[i]
                Vh_i = Vh_batch[i]
                scale_i = float(scale_batch[i].item())
                
                total_energy = (S_i ** 2).sum().item()
                k_sel = block_rank
                if total_energy > 1e-9:
                    cum = torch.cumsum(S_i ** 2, dim=0)
                    threshold = 0.999 * total_energy
                    idx = torch.where(cum >= threshold)[0]
                    if idx.numel() > 0:
                        k_sel = max(4, min(int(idx[0].item() + 1), block_rank))
                
                U_k = U_i[:, :k_sel] * S_i[:k_sel].unsqueeze(0)
                Vh_k = Vh_i[:k_sel, :]
                
                U_k_fp16 = U_k.to(torch.float16)
                Vh_k_fp16 = Vh_k.to(torch.float16)
                
                if not torch.isfinite(U_k_fp16).all():
                    U_k_fp16 = torch.nan_to_num(U_k_fp16, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(Vh_k_fp16).all():
                    Vh_k_fp16 = torch.nan_to_num(Vh_k_fp16, nan=0.0, posinf=0.0, neginf=0.0)
                
                U_scaled = U_k_fp16.float() * token_norms.unsqueeze(1)
                U_scaled = U_scaled.to(torch.float16)
                if not torch.isfinite(U_scaled).all():
                    U_scaled = torch.nan_to_num(U_scaled, nan=0.0, posinf=65504.0, neginf=-65504.0)
                
                recon_deltas = (U_k_fp16.float() @ Vh_k_fp16.float()) * scale_i
                recon_errors = (deltas_raw - recon_deltas.to(deltas_raw.device)).norm(dim=1)
                
                # ── SVD Residual Correction (C10 remediation parity) ──
                n, d = normalized_deltas.shape
                half_d = d // 2
                delta_K = deltas_raw[:, :half_d]
                delta_V = deltas_raw[:, half_d:]
                recon_K = recon_deltas[:, :half_d]
                recon_V = recon_deltas[:, half_d:]

                error_K = (delta_K - recon_K).norm(dim=1)
                error_V = (delta_V - recon_V).norm(dim=1)

                norm_K = delta_K.norm(dim=1).clamp(min=1e-8)
                norm_V = delta_V.norm(dim=1).clamp(min=1e-8)

                rel_error_K = error_K / norm_K
                rel_error_V = error_V / norm_V

                # Default SVD error threshold and residual fraction
                error_threshold = 0.08
                n_max_residual = int(n * 0.15)

                # OPT-A: Adaptive residual budget — 3-tier block classifier by median reconstruction error.
                median_err_K = float(torch.median(rel_error_K).item()) if rel_error_K.numel() > 0 else 0.0
                median_err_V = float(torch.median(rel_error_V).item()) if rel_error_V.numel() > 0 else 0.0
                max_median_err = max(median_err_K, median_err_V)
                if max_median_err < 0.05:
                    n_max_residual = min(8, n_max_residual)
                elif max_median_err < 0.15:
                    n_max_residual = min(16, n_max_residual)

                # Content-aware residual capture (C10 token boosting / table capture parity)
                # Matches compress_layer_blocks_gpu and MLX wrapper behavior
                if block_token_ids and getattr(self, "tokenizer", None) is not None and len(block_token_ids) == n:
                    try:
                        _sid = getattr(block, "session_id", None)
                        _cached_boost = None
                        if _sid is not None:
                            _boost_cache = getattr(self, "_res_capture_boost_rows", None)
                            if _boost_cache is None:
                                _boost_cache = self._res_capture_boost_rows = {}
                            _session_boosts = _boost_cache.setdefault(_sid, {})
                            _cached_boost = _session_boosts.get(block.anchor_idx)

                        if _cached_boost is not None:
                            boost_row, n_boosted = _cached_boost
                        else:
                            from native_core.compression.residual_capture import compute_boost_multipliers
                            _tok = self.tokenizer
                            _cache = getattr(self, "_res_capture_decode_cache", None)
                            if _cache is None:
                                _cache = self._res_capture_decode_cache = {}
                            tok_strs = []
                            for _tid in block_token_ids:
                                _s = _cache.get(_tid)
                                if _s is None:
                                    _s = _cache[_tid] = _tok.decode([_tid])
                                tok_strs.append(_s)
                            _all = self._session_token_ids.get(_sid) if getattr(self, "_session_token_ids", None) is not None else None
                            _total = int(_all.numel()) if _all is not None else len(block_token_ids)
                            _ckey = (_sid, _total)
                            _counts_cache = getattr(self, "_res_capture_counts", None)
                            if _counts_cache is None:
                                _counts_cache = self._res_capture_counts = {}
                            _counts = _counts_cache.get(_ckey)
                            if _counts is None and _all is not None:
                                _counts = {}
                                for _t in _all.tolist():
                                    _counts[_t] = _counts.get(_t, 0) + 1
                                _counts_cache.clear()
                                _counts_cache[_ckey] = _counts
                            boost_row, n_boosted = compute_boost_multipliers(
                                tok_strs, block_token_ids, _counts or {}, _total)
                            if _sid is not None:
                                _session_boosts[block.anchor_idx] = (boost_row, n_boosted)

                        if boost_row is not None and n_boosted > 0:
                            _bt = torch.tensor(boost_row, device=rel_error_K.device, dtype=rel_error_K.dtype)
                            rel_error_K = rel_error_K * _bt
                            rel_error_V = rel_error_V * _bt
                            try:
                                _margin = int(_os.environ.get("DIFFKV_RESIDUAL_FLOOR_MARGIN", "4"))
                            except Exception:
                                _margin = 4
                            n_max_residual = max(n_max_residual, min(n, n_boosted + _margin))
                    except Exception:
                        pass

                residual_K_pos = None
                residual_K_vals = None
                residual_V_pos = None
                residual_V_vals = None

                if n > 0 and n_max_residual > 0:
                    top_k_K = torch.topk(rel_error_K, k=min(n_max_residual, n))
                    top_k_V = torch.topk(rel_error_V, k=min(n_max_residual, n))
                    
                    mask_K = (top_k_K.values > error_threshold) & (error_K[top_k_K.indices] > 1e-4)
                    residual_K_pos = top_k_K.indices[mask_K]
                    
                    mask_V = (top_k_V.values > error_threshold) & (error_V[top_k_V.indices] > 1e-4)
                    residual_V_pos = top_k_V.indices[mask_V]
                    
                    device = deltas_raw.device
                    if residual_K_pos.numel() > 0:
                        res_K_vals = (delta_K - recon_K)[residual_K_pos]
                        if token_norms is not None:
                            res_K_vals = res_K_vals * token_norms.cpu()[residual_K_pos.cpu()].unsqueeze(1)
                        residual_K_vals = res_K_vals.to(torch.float16).to(device)
                        residual_K_pos = residual_K_pos.to(torch.int16).to(device)
                    else:
                        residual_K_pos = None
                        residual_K_vals = None
                        
                    if residual_V_pos.numel() > 0:
                        res_V_vals = (delta_V - recon_V)[residual_V_pos]
                        if token_norms is not None:
                            res_V_vals = res_V_vals * token_norms.cpu()[residual_V_pos.cpu()].unsqueeze(1)
                        residual_V_vals = res_V_vals.to(torch.float16).to(device)
                        residual_V_pos = residual_V_pos.to(torch.int16).to(device)
                    else:
                        residual_V_pos = None
                        residual_V_vals = None

                block.residual_K_positions = residual_K_pos
                block.residual_K_values = residual_K_vals
                block.residual_V_positions = residual_V_pos
                block.residual_V_values = residual_V_vals

                seq_len = k_orig.shape[2]
                top_k_val = min(3, seq_len)
                heads = k_orig.shape[1]
                head_dim = k_orig.shape[3]
                fact_anchors_K_val = torch.zeros((3, heads, head_dim), device=k_orig.device, dtype=k_orig.dtype)
                fact_anchors_V_val = torch.zeros((3, heads, head_dim), device=v_orig.device, dtype=v_orig.dtype)
                fact_anchor_positions_val = torch.full((3,), -1, device=k_orig.device, dtype=torch.int16)
                
                if top_k_val > 0:
                    top_3 = torch.topk(recon_errors, k=top_k_val)
                    for j, pos_val in enumerate(top_3.indices):
                        fact_anchors_K_val[j] = k_orig[0, :, pos_val, :]
                        fact_anchors_V_val[j] = v_orig[0, :, pos_val, :]
                        fact_anchor_positions_val[j] = pos_val.to(torch.int16)
                
                recon_norm = recon_deltas.norm()
                raw_norm = deltas_raw.norm().clamp(min=1e-8)
                cosine_sim = float((torch.sum(deltas_raw * recon_deltas.to(deltas_raw.device)) / (raw_norm * recon_norm + 1e-8)).item())
                norm_drift = float((recon_norm / raw_norm).item())
                
                self._postprocess_compressed_block(
                    block, U_scaled, Vh_k_fp16, scale_i, cosine_sim, norm_drift, k_sel,
                    fact_anchors_K_val, fact_anchors_V_val, fact_anchor_positions_val,
                    k_orig, v_orig, anchor_kv_local, block_rank
                )
            except Exception as e:
                print(f"[AsyncCompressor] Postprocess failed for block anchor={getattr(block, 'anchor_idx', '?')}: {e}")
                self._compressor._adjust_pending(-1)

    def _get_rotated_anchor_k(self, session_id, anchor_k, anchor_idx):
        """
        Applies RoPE to anchor_k at position anchor_idx using cached cos/sin in decode_workspace.
        """
        # Return unrotated anchor_k because the decode attention paths (Python fallback, C++, Metal)
        # always apply RoPE on-the-fly at runtime. Storing it pre-rotated in the pool causes
        # double-rotation bugs during attention decoding.
        return anchor_k


    # ── Diagnostics ───────────────────────────────────────────────────────────

    def runtime_summary(self) -> dict:
        pager_s = self.pager.summary()
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
            "async_compressor":      comp_s,
        }

    @property
    def sessions(self) -> dict:
        sessions_dict = {}
        for session_id in list(self.session_blocks.keys()):
            num_blocks = [len(self.session_blocks[session_id][l]) for l in range(self.num_layers)]
            seq_len = self.get_session_sequence_length(session_id)
            recency = 512
            if getattr(self, "_streaming_mgr", None) is not None:
                recency = getattr(self._streaming_mgr, "recency_window", 512)
            dense_len = min(seq_len, recency)
            dense_lens = [dense_len] * self.num_layers

            comp_res_n = []
            for l in range(self.num_layers):
                layer_res_n = []
                for b in self.session_blocks[session_id][l]:
                    slot_idx = getattr(b, "slot_idx", -1)
                    if slot_idx >= 0 and self.native_pool is not None:
                        res_pos = self.native_pool.residual_K_positions[slot_idx]
                        n_valid = int((res_pos >= 0).sum().item())
                        layer_res_n.append(n_valid)
                    else:
                        layer_res_n.append(0)
                comp_res_n.append(layer_res_n)

            sessions_dict[session_id] = {
                "num_blocks": num_blocks,
                "dense_lens": dense_lens,
                "comp_res_n": comp_res_n,
            }
        return sessions_dict

    def close(self):
        """Explicitly break all circular references and release pool memory to prevent leaks."""
        # 1. Reset and delete the native block pool
        if hasattr(self, "native_pool") and self.native_pool is not None:
            try:
                self.native_pool.reset()
            except Exception as e:
                print(f"[DiffKV] Warning during pool reset: {e}")
            self.native_pool = None
            
        # 2. Break block references to pool
        if hasattr(self, "session_blocks") and self.session_blocks:
            for session_id, layers in list(self.session_blocks.items()):
                for layer_idx, blocks in list(layers.items()):
                    for block in blocks:
                        try:
                            block.pool = None
                            block.pool_idx = None
                            block.active_k = None
                            block.active_v = None
                            block.active_k_cpu = None
                            block.active_v_cpu = None
                            block.anchor_kv = None
                            block.anchor_kv_cpu = None
                            block.U = None
                            block.V = None
                        except Exception:
                            pass
            self.session_blocks.clear()

        # 3. Stop background compressor
        if hasattr(self, "_compressor") and self._compressor is not None:
            try:
                self._compressor.stop()
            except Exception:
                pass
            self._compressor = None

        self._streaming_mgr = None
        self.tokenizer = None
        self.pager = None

    def __del__(self):
        self.close()
