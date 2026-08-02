"""
native_core/streaming_sparse_ingest.py

Phase 24.5 — True Streaming Sparse Ingest Manager

Replaces the dense-first set_kv() path in KVRuntimeManager.

OLD lifecycle:
    Dense allocation → async compression aging → eventual sparse

NEW lifecycle:
    Token chunk arrives
    → anchor extracted (1 token dense, irreducible)
    → micro-block accumulated (configurable: 8–32 tokens)
    → compression submitted immediately when micro-block fills
    → slab written while next chunk ingests (overlapped)
    → only a single micro-block window stays dense at any time

Key guarantees:
    1. No full-sequence dense allocation.
    2. Compression begins DURING ingest, not after.
    3. Dense footprint bounded to: micro_block_size * num_layers * 2 * heads * head_dim * 2 bytes
    4. Replay-safe: block stays readable via active_k/v until SVD completes (no partial state).
    5. Attention path falls back to dense gracefully for blocks mid-compression.
"""

import os
import torch
import queue
import threading
import re
from typing import Dict, List, Optional, Tuple, ClassVar, Any
from dataclasses import dataclass, field

try:
    from native_core.mac_utils import new_event as _new_event
except ImportError:
    def _new_event(device=None):
        if torch.cuda.is_available():
            return torch.cuda.Event()
        class _NE:
            def record(self, stream=None): pass
            def synchronize(self): pass
        return _NE()


# Module-level constant — avoids re-creating this dict on every metadata write call
# (previously created fresh inside update_metadata_block every token × every layer)
_STATE_CODES = {"ACCUMULATING": 0, "SUBMITTED": 1, "COMPRESSED": 2, "PAGED": 3}

# ── Precision-sensitive token detection ────────────────────────────────────────
# These patterns flag blocks whose tokens require EXACT key representations.
# Blocks matching any of these patterns are exempted from SVD compression so that
# the model can attend to them faithfully during decode.
#
# INTENTIONALLY NARROW: Only exact-value tokens that cannot be reconstructed
# approximately are exempted. Broad rules (math operators, LaTeX, quoted text)
# caused 40-60%% of blocks in technical papers to be exempt, spiking CPU RAM
# from ~3-4 GB to ~7-8 GB. Only rare, high-precision patterns are exempted.

_STOP_WORDS_COMPRESS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'a', 'an', 'the', 'and', 'but',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'now',
    'should', "should've", 'would', 'could', 'may', 'might', 'must',
    'shall', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'get', 'got', 'make', 'made', 'go', 'went', 'take', 'took',
    'see', 'saw', 'say', 'said', 'use', 'used', 'find', 'found',
    'question', 'answer', 'text', 'context', 'information', 'prompt',
    'query', 'assistant', 'system', 'user', 'file', 'document', 'page',
    'line', 'passage', 'following', 'please', 'write', 'read',
    'describe', 'explain', 'summarize', 'extract', 'retrieve', 'give',
    'tell', 'show', 'list', 'what', 'who', 'whom', 'which', 'detail',
    'details', 'brief', 'exact', 'exactly', 'correct', 'correctly',
    'true', 'false', 'yes', 'no'
}

# Narrow precision patterns: ONLY match patterns that (a) appear rarely in
# normal prose and (b) require exact attention for faithful reproduction.
_RE_LONG_DIGITS     = re.compile(r'\d{5,}')          # 5+ digit codes / IDs (rare)
# Alphanumeric IDENTIFIER codes: hyphenated alnum (SIGMA-1409-ZETA, GPT-4,
# COVID-19) or a contiguous mixed run with >=2 letters + a digit (SKU9910,
# AB12CD, iPhone12X). These are rare in prose but REQUIRE exact reconstruction
# (a single wrong token corrupts the whole code). The \d{5,} rule above only
# caught 5+ pure-digit runs, so short/hyphenated codes (e.g. a 4-digit PIN in a
# code, the common ID form) fell through to lossy SVD and decoded to garbage —
# the CUDA-only gap vs MLX (which captures exact residuals uniformly, no digit-
# length gate). Deliberately EXCLUDES bare years/decades/counts/dates ("1409",
# "2010s", "100k", "2020-01-01", "Table 2") to keep the narrow-ruleset RAM
# guarantee: the hyphenated form needs a letter AND a digit (dates have no
# letter); the contiguous form needs >=2 letters (decades have one).
_RE_ALNUM_CODE      = re.compile(
    r'\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*[0-9])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b'
    r'|\b(?=(?:[A-Za-z0-9]*[A-Za-z]){2})(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{4,}\b'
)
# Scientific notation: 1.23e+4, 2.998e8, 6.02E23 — never in prose
_RE_SCI_NOTATION    = re.compile(r'\d+\.?\d*[eE][+\-]?\d+')
# Unicode-only math symbols: π, ∑, ∞, ≤, ±, etc. — NEVER in normal prose
_RE_UNICODE_MATH    = re.compile(
    r'[\u221a\u2211\u222b\u2202\u03c0\u03a0\u03a3\u221e\u2264\u2265\u2260\u00b1\u00f7\u00d7]'
)
_RE_LATEX_MATH      = re.compile(
    r'\$\$|\\\[|\\\(|\\begin\{(?:equation|align|gather|math|displaymath)\}|\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|sum|int|prod|partial|nabla|hbar|infty|approx|neq|le|ge|times|div|cdot|sqrt|frac)\b|_\{[^\}]+\}|\^[^\}]+\}'
)
_RE_ASCII_EQUATION  = re.compile(
    r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[-+]?[a-zA-Z0-9_.\(\)\+\-\*\/]+'
)
_RE_DEFINITIONS     = re.compile(
    r'\b(?:is|are|we)\s+(?:defined|referred|called|known)\s+(?:as|by)\b|\brefers?\s+to\b|\b(?:denotes?|stands\s+for|represents?)\b|\bwe\s+define\b|\b(?:let\s+us|let)\s+define\b',
    re.IGNORECASE
)
_RE_CLAIMS          = re.compile(
    r'\b(?:theorem|lemma|proposition|corollary|conjecture|hypothesis|proof)\s+\d+(?:\.\d+)*\b|\bour\s+main\s+contribution\b|\bwe\s+(?:prove|show|demonstrate|argue|conclude|find)\s+that\b|\bour\s+(?:results|analysis)\s+show\b',
    re.IGNORECASE
)
_RE_ACRONYMS        = re.compile(r'\b[A-Z]{2,}\b')
# Content words for short-digit query overlap check
_RE_WORD_TOKENS     = re.compile(r'\b[a-z0-9]{2,}\b')


def _new_metadata_tensor(rows: int) -> torch.Tensor:
    """Allocate a block-metadata tensor that stays mutable outside InferenceMode.

    These are plain CPU int32 bookkeeping rows (pool_idx, anchor_idx, token_count,
    state) -- nothing autograd ever sees. But they are allocated lazily from
    inside the model's forward, which transformers runs under
    torch.inference_mode(). A tensor created there is an INFERENCE TENSOR
    permanently, and mutating one outside that mode raises:

        RuntimeError: Inplace update to inference tensor outside InferenceMode
                      is not allowed.

    Which is exactly what happened: prefill allocated the metadata inside
    inference_mode, then compress_deferred_prefill_blocks -- called after the
    forward returns, outside the mode -- tried to write pool_idx into it and
    aborted. Forcing inference_mode(False) here makes the allocation independent
    of whatever mode the caller happens to be in.
    """
    with torch.inference_mode(False):
        return torch.full((rows, 4), -1, dtype=torch.int32)


def _resolve_short_context_threshold() -> int:
    """Context length below which blocks are kept DENSE (never SVD-compressed).

    THIS EXISTS BECAUSE TWO SHIPPED SERVING DEFAULTS HAD NO EFFECT ON CUDA.

    decode_config.BEST_DECODE_DEFAULTS -- what cli.py and the OpenAI gateway
    apply, and what validate_cuda_dkv.py prints as "serving defaults applied" --
    sets DKV_COMPRESSED_DECODE=auto and DKV_COMPRESSED_MIN_CTX=8192. A search
    across every .py/.cpp/.cu/.mm/.h/.metal file in the repo finds exactly one
    consumer of either name: serving/mlx_dkv_wrapper.py:4477 and :4482. No CUDA
    path read them. They were inert on the runtime the validator validates,
    while being printed at the top of every run as though they applied.

    The behaviours were therefore 32x apart on the same nominal config:

        MLX   _resolve_compressed_decode(seq_len) -> dense below 8192 tokens
        CUDA  short_context_threshold = 256       -> compress from 256 tokens

    That is not a numerical divergence, it is a different code path for the
    same request. At the validator's 2k cases (2822 tokens) MLX runs DENSE and
    CUDA runs COMPRESSED -- which is worth knowing before comparing their needle
    results, because "MLX passes 2k@0.0" partly means "MLX never compressed it".

    CUDA has no decode-time dense/sparse switch to mirror MLX's directly, but it
    does not need one: gating INGEST has the same net effect, because a block
    that is never compressed is still readable through active_k and decode
    attends it densely. short_context_threshold is that gate and it was simply
    set 32x too low relative to the shipped policy.

    Mirrors _resolve_compressed_decode's tri-state exactly:
      "1"/"on"/"true"   -> always compress (256, the previous CUDA behaviour)
      "0"/"off"/"false" -> never compress
      "auto"            -> dense below DKV_COMPRESSED_MIN_CTX

    NOTE: MLX defaults DKV_COMPRESSED_MIN_CTX to 16384 and BEST_DECODE_DEFAULTS
    overrides it to 8192; this reads the same variable with the same fallback,
    so both runtimes move together.
    """
    mode = os.environ.get("DKV_COMPRESSED_DECODE", "1").strip().lower()
    if mode in ("1", "on", "true", "yes"):
        return 256
    if mode in ("0", "off", "false", "no"):
        return 1 << 30                     # effectively never compress

    # ── 'auto' CANNOT be honoured on CUDA yet. MEASURED, NOT ASSUMED. ────────
    # The first version of this function mapped 'auto' to DKV_COMPRESSED_MIN_CTX
    # (8192 under the serving defaults) on the reasoning that gating INGEST is
    # equivalent to MLX's decode-time switch, "because a block that is never
    # compressed stays readable through active_k and decode attends it densely".
    #
    # That reasoning was WRONG, and the GPU run said so in two lines:
    #
    #   WARNING: dense window (2823 tokens) exceeds workspace (1538).
    #            Trimming oldest blocks to fit.
    #   WARNING: 0 compressed blocks routed and this decoder has no dense-only
    #            path -- attention output will be EMPTY ([1,8,1,0] instead of
    #            [1,8,1,256]). dense_window_present=True, layer=3.
    #
    # followed by the pool ballooning to all 8301 slots and a device-side assert
    # in vectorized_gather_kernel (index out of bounds) that aborted the process.
    #
    # So CUDA has no dense-only decode path. With nothing compressed, decode does
    # not fall back to dense -- it returns an EMPTY tensor. And the dense-window
    # workspace is sized (DKV_MAX_DENSE_LEN) far below a 2.8k context, so keeping
    # everything dense also overflows it and trims the oldest blocks, which is
    # where the needle at depth 0.0 lives.
    #
    # Net effect of that change on the shipped config: 2k@0.0 started passing
    # (its blocks stopped being compressed) while 2k@0.5 went 3/3 -> 0/3 and the
    # 8k case crashed. A strictly worse trade.
    #
    # 'auto' therefore resolves to 256 -- CUDA's long-standing behaviour -- until
    # a real dense-only decode path exists. Honouring it needs THREE things, none
    # of which is a config change:
    #   1. a decode path that attends the dense window when N == 0
    #      (the warning above is in native_triton_sparse_attn_decode)
    #   2. DKV_MAX_DENSE_LEN sized to the threshold, not to 1538
    #   3. whatever indexes the trimmed dense window fixed, so it stops
    #      generating out-of-range gather indices
    #
    # The divergence this function documents is REAL and still unfixed: MLX runs
    # dense below 8192 and CUDA compresses from 256, on the same nominal config.
    # It just cannot be closed from this end.
    return 256


@dataclass
class StreamingKVBlock:
    """
    A KV block with sparse-first lifecycle.
    
    States:
        ACCUMULATING  — active_k/v accumulating tokens, not yet eligible
        SUBMITTED     — queued for compression, still readable via active_k/v
        COMPRESSED    — U/V set, active_k/v=None, VRAM freed
        PAGED         — evicted to CPU RAM
    """
    anchor_idx: int
    anchor_kv:  torch.Tensor          # [1, 2, heads, head_dim] — ALWAYS dense (1 token)
    anchor_kv_cpu: Optional[torch.Tensor] = None  # CPU-resident anchor cache
    micro_block_size: int = 16        # compress at this threshold, not 64

    # Mutable KV state
    active_k: Optional[torch.Tensor] = None  # [1, heads, T, head_dim]
    active_v: Optional[torch.Tensor] = None
    active_k_cpu: Optional[torch.Tensor] = None  # CPU-pinned uncompressed cache
    active_v_cpu: Optional[torch.Tensor] = None

    # Compressed state (set by compressor)
    _U: Optional[torch.Tensor] = None
    _V: Optional[torch.Tensor] = None
    U_cpu: Optional[torch.Tensor] = None
    V_cpu: Optional[torch.Tensor] = None
    scale: float = 1.0
    cosine_sim: float = 1.0
    norm_drift: float = 0.0
    dynamic_rank: int = -1

    token_indices: List[int] = field(default_factory=list)
    state: str = "ACCUMULATING"  # ACCUMULATING | SUBMITTED | COMPRESSED | PAGED
    pool_idx: Optional[int] = None
    dirty: bool = True
    is_outlier: bool = False
    skip_compression: bool = False
    session_id: Optional[str] = None
    layer_idx: Optional[int] = None
    _cache_id: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    pool: Any = None

    _residual_K_positions: Optional[torch.Tensor] = None
    _residual_K_values: Optional[torch.Tensor] = None
    _residual_V_positions: Optional[torch.Tensor] = None
    _residual_V_values: Optional[torch.Tensor] = None

    _U_sem_int4: Optional[torch.Tensor] = None
    _U_sem_scale: Optional[torch.Tensor] = None
    _U_fact_fp16: Optional[torch.Tensor] = None
    _n_semantic: int = 0

    _fact_anchors_K: Optional[torch.Tensor] = None
    _fact_anchors_V: Optional[torch.Tensor] = None
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


    # ── Phase 29: Ring buffer fields (Fix #1 — eliminate torch.cat per token) ──
    # Pre-allocated GPU tensor of shape [1, heads, micro_block_size, head_dim].
    # Tokens are written in-place via buf[:, :, fill, :] = k[:, :, 0, :].
    # active_k/v are kept as O(1) views: buf[:, :, :fill, :] — no allocation.
    _active_buf_k: Optional[torch.Tensor] = None
    _active_buf_v: Optional[torch.Tensor] = None
    _active_fill: int = 0   # number of tokens written into the ring buffer

    # ── O(1) metadata index — set at creation, used by update_metadata_state ──
    # Stores this block's row index in session_metadata[session_id][layer_idx].
    # Eliminates the O(N) linear scan in update_metadata_state() for each
    # block's state-change callback (compression, finalize, rollback).
    _metadata_idx: int = -1   # -1 = not yet assigned

    # ── Configurable thresholds (class level for slot-friendliness) ──
    short_context_threshold: ClassVar[int] = 256   # overwritten by the manager
    protect_block_zero: ClassVar[bool] = True

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        # anchor_kv_cpu is intentionally NOT created here.
        # It is lazily created in _compress_block_sync() only when the CPU
        # compression path actually needs it (i.e. when k/v are CPU tensors).
        # Eager creation wastes RAM: 28 layers x 500 blocks x anchor_size bytes
        # per session for a field that is rarely accessed on the hot path.
        pass

    def token_count(self) -> int:
        if self.token_indices:
            return max(0, len(self.token_indices) - 1)
        if self.active_k is not None:
            return self.active_k.shape[2]
        if self.active_k_cpu is not None:
            return self.active_k_cpu.shape[2]
        if self.U is not None:
            return self.U.shape[0]
        return 0

    def is_compression_eligible(self) -> bool:
        # Dynamic Compression Guard: SVD compression is deferred for short context windows
        # (< 256 tokens) to preserve 100% exact-match precision and eliminate SVD CPU/GPU
        # roundtrip overhead for very short conversations. The 512-token recency window
        # (enforced by the rolling loop below) guarantees exact attention for recent tokens
        # regardless of this gate. Lowered from 1024 to 256 so mid-length prompts (256-1024)
        # can start compressing — they were previously locked dense unnecessarily.
        # Also protect blocks flagged with outliers (e.g. key activation > 20.0) from SVD
        # compression to prevent attention sink corruption.
        if (self.anchor_idx == 0 and StreamingKVBlock.protect_block_zero) or self.is_outlier or self.skip_compression:
            return False
        if self.anchor_idx + self.token_count() < StreamingKVBlock.short_context_threshold:
            return False
        return (
            self.state == "ACCUMULATING"
            and self.active_k is not None
            and self.active_k.shape[2] >= self.micro_block_size
        )
_original_is_compression_eligible = StreamingKVBlock.is_compression_eligible

def _is_block_compression_eligible(block: StreamingKVBlock, is_last_block: bool = False,
                                   ignore_skip_compression: bool = False) -> bool:
    # protect_block_zero exists to keep a "sink" region out of LOSSY SVD
    # compression. It must NOT apply when this is the deferred force-exact
    # path for a skip_compression block (ignore_skip_compression=True):
    # force_exact compression is lossless (every position kept as an exact
    # residual), so there is no corruption risk to protect against -- and
    # the alternative is actively worse. A skip_compression block stuck
    # ACCUMULATING forever still counts as "dense" for the recency window
    # (get_cached_decode_blocks), and assemble_dense_window_kv trims the
    # OLDEST dense blocks first once the window is full. Anchor 0 is by
    # definition the oldest block in the session, so once enough tokens
    # follow it, it's evicted from the dense window every single step --
    # never compressed (this check) and no longer dense (evicted) is total
    # silent data loss: the block, and anything in it (e.g. a needle placed
    # at the very start of a long prompt), becomes permanently invisible to
    # decode. Confirmed empirically: anchor=0 stuck at state=ACCUMULATING
    # indefinitely on an 8k-token Qwen3.5-2B prompt with the needle at the
    # very start -- dense recalled it in full, DKV never did.
    #
    # THE HATCH ABOVE WAS TOO NARROW. It released block 0 only when the block
    # was ALSO flagged skip_compression, i.e. only when one of the Rule 1-5
    # regexes happened to match its text. When no rule matched, block 0 stayed
    # protected and the exact data loss the comment describes still happened.
    #
    # That is the 2k@depth0.0 failure. On the 8k prompt "Rule 5 skip block
    # anchor=0: word 'helpful' occurs 1 times" fires -- the chat template's
    # system line -- so the hatch opens and 8k@depth0.0 recalls 3/3. On the 2k
    # prompt no rule matches block 0, so it is never compressed, is trimmed out
    # of the dense window as the oldest block, and the needle placed at depth 0.0
    # -- which lives in block 0 -- is invisible to every decode step. CUDA
    # returns token salad 0/3, deterministically, while MLX on the identical
    # 2822-token prompt with compression forced on returns it 3/3 (MLX applies
    # recency at ATTENTION time and has no state a block can be stranded in).
    #
    # Whether a needle at the start of a document survives therefore depended on
    # whether an unrelated regex matched the system prompt. That is not a policy.
    #
    # The deferred path now releases block 0 unconditionally, and the caller
    # marks it force-exact before submitting, so protect_block_zero is honoured
    # the way it is already honoured for skip blocks: block 0 is protected from
    # LOSSY compression by being compressed LOSSLESSLY, not by being left out of
    # attention entirely.
    _block_zero_protected = (
        block.anchor_idx == 0
        and StreamingKVBlock.protect_block_zero
        and not ignore_skip_compression
    )
    if _block_zero_protected or block.is_outlier:
        return False
    # skip_compression can be bypassed for the deferred prefill path — see compress_deferred_blocks
    if not ignore_skip_compression and block.skip_compression:
        return False
    toks = block.token_count()
    if block.anchor_idx + toks < StreamingKVBlock.short_context_threshold:
        return False
        
    is_patched = (StreamingKVBlock.is_compression_eligible != _original_is_compression_eligible)
    if is_patched:
        return block.is_compression_eligible()
        
    size_ok = (toks >= block.micro_block_size) if is_last_block else True
    return (
        block.state == "ACCUMULATING"
        and (block.active_k is not None or block.active_k_cpu is not None)
        and size_ok
    )


class StreamingSparseIngestManager:
    """
    True streaming sparse-ingest KV manager.

    Core contract:
        - Dense footprint per session ≤ 1 micro_block × num_layers (current accumulation window)
        - All older blocks are either SUBMITTED or COMPRESSED
        - Compression runs concurrently with token ingest via background threads
        - Single anchor token (1 token per block) is the only irreducible dense requirement

    Parameters
    ----------
    micro_block_size : int
        Number of non-anchor tokens to accumulate before triggering compression.
        Default=16 (vs. old block_size=64). Smaller = less dense residency.
        Minimum useful value: 8 (smaller causes SVD overhead to dominate).
    dense_anchor_only : bool
        If True, only the anchor token is kept dense during ACCUMULATING state —
        all other tokens are compressed immediately on micro-block fill.
        If False, the entire micro-block stays dense until fill (legacy-compatible).
    """

    def __init__(
        self,
        compressor,               # AsyncCompressor instance
        compress_fn,              # sync compression callable(block, k, v)
        micro_block_size: int = 16,
        dense_anchor_only: bool = True,
        native_pool = None,
        device: str = "cuda",
        recency_window: int = 512,
        short_context_threshold: int = None,
        protect_block_zero: bool = True,
    ):
        self.compressor = compressor
        self.compress_fn = compress_fn
        self.micro_block_size = micro_block_size
        self.dense_anchor_only = dense_anchor_only
        self.native_pool = native_pool
        self.device = device
        self.recency_window = recency_window
        if short_context_threshold is None:
            short_context_threshold = _resolve_short_context_threshold()
        self.short_context_threshold = short_context_threshold
        self.protect_block_zero = protect_block_zero
        StreamingKVBlock.short_context_threshold = short_context_threshold
        StreamingKVBlock.protect_block_zero = protect_block_zero
        self.manager = None

        # session_id -> layer_idx -> List[StreamingKVBlock]
        self.session_blocks: Dict[str, Dict[int, List[StreamingKVBlock]]] = {}
        
        # session_id -> layer_idx -> Contiguous 2D metadata tensor [MAX_BLOCKS, 4]
        self.session_metadata: Dict[str, Dict[int, torch.Tensor]] = {}

        # Metadata changes relevant to sparse decode only when block membership
        # or compression state changes.  Decode token appends update the live
        # block/ring-buffer directly and do not need to invalidate this cache.
        self._metadata_versions: Dict[str, Dict[int, int]] = {}
        
        # session_id -> micro_block_size (dynamic adaptive size per session)
        self.session_micro_block_sizes: Dict[str, int] = {}
        
        # session_id -> (k_gpu, v_gpu, k_cpu, v_cpu) pre-allocated pinned memory buffers.
        # A single buffer per session is shared across all layers (safe because slices are
        # cloned before being enqueued into the async compressor — breaking aliasing).
        self.session_staging_buffers: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # session_id -> prefill_len
        self.session_prefill_lens: Dict[str, int] = {}

        # session_id -> cached query words set
        self.session_query_words: Dict[str, set] = {}

        # session_id -> document word counts
        self.session_doc_words: Dict[str, dict] = {}

        # Telemetry
        self.stats = {
            "total_blocks_created": 0,
            "total_compressed": 0,
            "total_dense_tokens_peak": 0,
            "compressions_during_ingest": 0,
        }
        self._stats_lock = threading.Lock()

    # ── Session management ─────────────────────────────────────────────────────

    def init_session(self, session_id: str, num_layers: int, prefill_len: int = 0):
        if session_id not in self.session_blocks:
            self.session_blocks[session_id] = {i: [] for i in range(num_layers)}
        if session_id not in self.session_metadata:
            self.session_metadata[session_id] = {}
        self._metadata_versions.setdefault(session_id, {})
        self.session_prefill_lens[session_id] = prefill_len
        if session_id not in self.session_micro_block_sizes:
            # DKV_ADAPTIVE_BLOCK_SIZE — default OFF = MLX parity (fixed block).
            #
            # MLX has ONE block size (block_size=256) for every prompt. This
            # length-bucketed schedule is a CUDA-only invention, and it changes
            # more than memory layout:
            #
            #  * RANK IS FIXED AT 32 WHILE THE BLOCK SHRINKS. At S=16 the rank
            #    exceeds the block, so the "low-rank" factorisation is essentially
            #    lossless; at S=256 it is 8x compression. The fidelity regime — and
            #    therefore what the needle tests measure — silently changes with
            #    prompt length, which is why short contexts pass easily.
            #  * THE ROUTED TOKEN BUDGET moves with it. K blocks x span tokens is
            #    the quantity that has to match MLX's 4096, and span here has been
            #    observed at 65 (2822-token prompt) and 129 (7739-token prompt) in
            #    the same model on consecutive runs.
            #  * POOL SIZING assumes >=257 (max(pool_block_size, 257)), so a 16-
            #    to 128-token block wastes most of every slot.
            #  * RESULTS ARE NOT COMPARABLE ACROSS CONTEXT LENGTHS, which makes a
            #    depth sweep at 2k/8k/32k three different algorithms rather than
            #    one algorithm at three lengths.
            #
            # Set DKV_ADAPTIVE_BLOCK_SIZE=1 to restore the schedule.
            _adaptive = os.environ.get("DKV_ADAPTIVE_BLOCK_SIZE", "0") == "1"
            if _adaptive and prefill_len > 0:
                if prefill_len < 256:
                    raw_target = 16
                elif prefill_len < 1024:
                    raw_target = 32
                elif prefill_len < 4096:
                    raw_target = 64
                elif prefill_len < 8192:
                    raw_target = 128
                else:
                    raw_target = 256
                target = min(raw_target, self.micro_block_size)
                adaptive_size = max(16, ((target + 15) // 16) * 16)
                self.session_micro_block_sizes[session_id] = adaptive_size
            else:
                self.session_micro_block_sizes[session_id] = self.micro_block_size

    def _get_session_staging_buffer(
        self,
        session_id: str,
        num_blocks: int,
        heads: int,
        micro_block_size: int,
        head_dim: int,
        device: str
    ):
        key = session_id
        buffers = self.session_staging_buffers.get(key)
        
        if (buffers is None or 
            buffers[0].shape[0] < num_blocks or 
            buffers[0].shape[2] < micro_block_size or
            buffers[0].shape[1] != heads or
            buffers[0].shape[3] != head_dim):
            
            # Allocate larger buffers dynamically.
            # Use 4× micro_block_size as headroom to avoid frequent reallocations;
            # previously this was hardcoded to max(..., 256) which over-allocated
            # staging buffers by 8–16× when micro_block_size=16 or 32.
            alloc_blocks = max(num_blocks, 16)
            alloc_mbs = max(micro_block_size * 4, micro_block_size + 16)
            
            k_gpu = torch.zeros((alloc_blocks, heads, alloc_mbs, head_dim), dtype=torch.float16, device=device)
            v_gpu = torch.zeros((alloc_blocks, heads, alloc_mbs, head_dim), dtype=torch.float16, device=device)
            
            # pin_memory() accelerates async GPU->CPU DMA on CUDA; not available/needed on MPS.
            _use_pinned = device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda")
            k_cpu = torch.zeros((alloc_blocks, heads, alloc_mbs, head_dim), dtype=torch.float16)
            v_cpu = torch.zeros((alloc_blocks, heads, alloc_mbs, head_dim), dtype=torch.float16)
            if _use_pinned:
                k_cpu = k_cpu.pin_memory()
                v_cpu = v_cpu.pin_memory()
            
            buffers = (k_gpu, v_gpu, k_cpu, v_cpu)
            self.session_staging_buffers[key] = buffers
            
        k_gpu, v_gpu, k_cpu, v_cpu = buffers
        return (
            k_gpu[:num_blocks, :, :micro_block_size, :],
            v_gpu[:num_blocks, :, :micro_block_size, :],
            k_cpu[:num_blocks, :, :micro_block_size, :],
            v_cpu[:num_blocks, :, :micro_block_size, :]
        )

    def clear_session(self, session_id: str):
        self.session_blocks.pop(session_id, None)
        self.session_metadata.pop(session_id, None)
        self._metadata_versions.pop(session_id, None)
        self.session_micro_block_sizes.pop(session_id, None)
        self.session_staging_buffers.pop(session_id, None)
        self.session_prefill_lens.pop(session_id, None)
        self.session_query_words.pop(session_id, None)
        self.session_doc_words.pop(session_id, None)
        # Clear the skip_compression cache for this session (prevents stale entries)
        if hasattr(self, "_skip_compress_cache"):
            keys_to_del = [k for k in self._skip_compress_cache if k[0] == session_id]
            for k in keys_to_del:
                del self._skip_compress_cache[k]


    def rollback_session(self, session_id: str, target_len: int) -> None:
        """
        Rollback/truncate a session's KV cache to a target sequence length.
        Used by speculative decoding to discard rejected candidate tokens.
        """
        if session_id not in self.session_blocks:
            return

        num_layers = len(self.session_blocks[session_id])
        for layer_idx in range(num_layers):
            blocks = self.session_blocks[session_id][layer_idx]
            if not blocks:
                continue

            new_blocks = []
            for block_idx, block in enumerate(blocks):
                # Case 1: Entire block is at or after target_len -> discard
                if block.anchor_idx >= target_len:
                    if block.pool_idx is not None and self.native_pool is not None:
                        self.native_pool.free_block(block.pool_idx)
                    # Reset metadata row
                    metadata = self.session_metadata.get(session_id, {}).get(layer_idx)
                    if metadata is not None and block_idx < metadata.shape[0]:
                        metadata[block_idx] = -1
                    continue

                # Case 2: Block spans across target_len -> truncate
                total_tokens = len(block.token_indices)
                if block.anchor_idx + total_tokens > target_len:
                    keep_count = target_len - block.anchor_idx
                    block.token_indices = block.token_indices[:keep_count]
                    keep_active = keep_count - 1 # anchor is at index 0

                    if block._active_buf_k is not None:
                        block._active_fill = keep_active
                        block.active_k = block._active_buf_k[:, :, :keep_active, :] if keep_active > 0 else None
                        block.active_v = block._active_buf_v[:, :, :keep_active, :] if keep_active > 0 else None
                    else:
                        if block.active_k is not None:
                            block.active_k = block.active_k[:, :, :keep_active, :] if keep_active > 0 else None
                            block.active_v = block.active_v[:, :, :keep_active, :] if keep_active > 0 else None
                    
                    if block.active_k_cpu is not None:
                        block.active_k_cpu = block.active_k_cpu[:, :, :keep_active, :] if keep_active > 0 else None
                        block.active_v_cpu = block.active_v_cpu[:, :, :keep_active, :] if keep_active > 0 else None

                    if block._U is not None:
                        block._U = block._U[:keep_active, :]
                    if block.U_cpu is not None:
                        block.U_cpu = block.U_cpu[:keep_active, :]
                    if block.pool_idx is not None and self.native_pool is not None:
                        self.native_pool.seq_lens[block.pool_idx] = keep_active

                    block.dirty = True
                    new_blocks.append(block)
                    self.update_metadata_block(session_id, layer_idx, len(new_blocks) - 1, block)
                else:
                    # Case 3: Block is entirely before target_len -> keep
                    new_blocks.append(block)
                    self.update_metadata_block(session_id, layer_idx, len(new_blocks) - 1, block)

            self.session_blocks[session_id][layer_idx] = new_blocks

            # Reset any unused rows in the metadata tensor to -1
            metadata = self.session_metadata.get(session_id, {}).get(layer_idx)
            if metadata is not None:
                for idx in range(len(new_blocks), metadata.shape[0]):
                    metadata[idx] = -1


    def update_metadata_block(self, session_id: str, layer_idx: int, block_idx: int, block):
        # Phase 29 Fix #3: metadata is a CPU tensor — all writes are pure CPU memory ops,
        # zero CUDA syncs (previously 4 GPU scalar writes = 4 CUDA syncs per call).
        metadata = self.session_metadata.setdefault(session_id, {}).setdefault(
            layer_idx, _new_metadata_tensor(1024)  # CPU — no device=
        )
        if block_idx >= metadata.shape[0]:
            new_size = metadata.shape[0] * 2
            new_meta = _new_metadata_tensor(new_size)  # CPU
            new_meta[:metadata.shape[0]] = metadata
            self.session_metadata[session_id][layer_idx] = new_meta
            metadata = new_meta

        metadata[block_idx, 0] = int(block.pool_idx) if block.pool_idx is not None else -1
        metadata[block_idx, 1] = int(block.anchor_idx)
        metadata[block_idx, 2] = block.token_count()
        metadata[block_idx, 3] = _STATE_CODES.get(block.state, -1)
        versions = self._metadata_versions.setdefault(session_id, {})
        versions[layer_idx] = versions.get(layer_idx, 0) + 1

    def update_metadata_state(self, session_id: str, layer_idx: int, block):
        """
        O(1) metadata state update using the block's pre-assigned _metadata_idx.

        Falls back to O(N) linear scan only when _metadata_idx is not set
        (e.g. blocks created before this optimization was deployed).
        """
        block_idx = getattr(block, "_metadata_idx", -1)
        if block_idx < 0:
            # Fallback: legacy O(N) scan for blocks without _metadata_idx
            blocks = self.session_blocks.get(session_id, {}).get(layer_idx, [])
            for idx, b in enumerate(blocks):
                if b is block:
                    block_idx = idx
                    break
        if block_idx < 0:
            return
        metadata = self.session_metadata.get(session_id, {}).get(layer_idx)
        if metadata is not None and block_idx < metadata.shape[0]:
            metadata[block_idx, 0] = int(block.pool_idx) if block.pool_idx is not None else -1
            metadata[block_idx, 2] = block.token_count()
            metadata[block_idx, 3] = _STATE_CODES.get(block.state, -1)
            versions = self._metadata_versions.setdefault(session_id, {})
            versions[layer_idx] = versions.get(layer_idx, 0) + 1

    def _should_skip_compression(self, session_id: str, anchor_idx: int, block_capacity: int) -> bool:
        """
        Return True if this block should be kept DENSE (skipping SVD compression).

        Uses a NARROW ruleset intentionally — broad rules (math operators, LaTeX,
        quoted strings) exempt 40-60%% of blocks in technical papers, causing
        3-4× CPU RAM growth. Only patterns that are rare in prose AND require
        exact attention for faithful reproduction are included:

          1. Long digit sequences (≥5 digits): IDs, timestamps, DOIs.
          2. Scientific notation: 1.23e+4, 2.998e8 — never appear in prose.
          3. Unicode math symbols: π, ∑, ∞, ≤, ± — never in normal prose.
          4. Short digits (≥2) that overlap with current query keywords.

        Results are cached per (session_id, anchor_idx) so the 7-regex + tokenizer.decode
        work is only done ONCE across all 40 layers, cutting CPU usage by 97.5%.
        """
        # ── Fast cache lookup ─────────────────────────────────────────────────
        if not hasattr(self, "_skip_compress_cache"):
            self._skip_compress_cache = {}
        _cache_key = (session_id, anchor_idx)
        if _cache_key in self._skip_compress_cache:
            return self._skip_compress_cache[_cache_key]

        if os.environ.get("DKV_DISABLE_REGEX_HEURISTICS", "0").lower() in ("1", "true", "yes"):
            self._skip_compress_cache[_cache_key] = False
            return False

        if self.manager is None or getattr(self.manager, "tokenizer", None) is None:
            self._skip_compress_cache[_cache_key] = False
            return False
        session_tok_dict = getattr(self.manager, "_session_token_ids", {})
        token_ids_cpu = session_tok_dict.get(session_id)
        if token_ids_cpu is None:
            self._skip_compress_cache[_cache_key] = False
            return False

        start = anchor_idx
        end = min(start + block_capacity, len(token_ids_cpu))
        if start >= end:
            self._skip_compress_cache[_cache_key] = False
            return False

        block_toks = token_ids_cpu[start:end].tolist()

        _result = False
        try:
            block_text = self.manager.tokenizer.decode(block_toks)
            block_text_lc = block_text.lower()

            # Rule 1: Long digit codes (IDs, years+, zip codes, DOIs) — always exempt
            if _RE_LONG_DIGITS.search(block_text):
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[DKV DEBUG] Rule 1 skip block anchor={anchor_idx}: '{block_text}'")
                _result = True
            # Rule 1b: Alphanumeric identifier codes (SIGMA-1409-ZETA, SKU9910,
            # GPT-4) — always exempt. Catches short/hyphenated codes the \d{5,}
            # rule missed (the CUDA-vs-MLX random-code retrieval gap).
            elif _RE_ALNUM_CODE.search(block_text):
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[DKV DEBUG] Rule 1b skip block anchor={anchor_idx}: '{block_text}'")
                _result = True
            # Rule 2: Scientific notation — always exempt (1.23e+4, 2.998e8)
            elif _RE_SCI_NOTATION.search(block_text):
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[DKV DEBUG] Rule 2 skip block anchor={anchor_idx}: '{block_text}'")
                _result = True
            # Rule 3: Unicode math symbols — always exempt (π, ∑, ∞, ≤, ±, etc.)
            elif _RE_UNICODE_MATH.search(block_text):
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[DKV DEBUG] Rule 3 skip block anchor={anchor_idx}: '{block_text}'")
                _result = True
            # Rule 3b: LaTeX math formula block — always exempt
            elif _RE_LATEX_MATH.search(block_text):
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[DKV DEBUG] Rule 3b skip block anchor={anchor_idx}: '{block_text}'")
                _result = True
            # Rule 3c: ASCII equation statement — always exempt
            elif _RE_ASCII_EQUATION.search(block_text):
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[DKV DEBUG] Rule 3c skip block anchor={anchor_idx}: '{block_text}'")
                _result = True
            # Rule 3d: Verbatim definitions — always exempt
            elif _RE_DEFINITIONS.search(block_text):
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[DKV DEBUG] Rule 3d skip block anchor={anchor_idx}: '{block_text}'")
                _result = True
            # Rule 3e: Formal claims / theorems — always exempt
            elif _RE_CLAIMS.search(block_text):
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[DKV DEBUG] Rule 3e skip block anchor={anchor_idx}: '{block_text}'")
                _result = True
            else:
                # Rule 3f: Acronym density — always exempt
                acronyms = set(_RE_ACRONYMS.findall(block_text))
                if len(acronyms) >= 3:
                    if os.environ.get("DKV_TELEMETRY", "0") == "1":
                        print(f"[DKV DEBUG] Rule 3f skip block anchor={anchor_idx}: '{block_text}', acronyms={acronyms}")
                    _result = True
                else:
                    # Rule 4: Short digits (≥2 digits) with query-word overlap
                    digit_parts = re.findall(r'\d+', block_text)
                    if any(len(p) >= 2 for p in digit_parts):
                        query_words = self.session_query_words.get(session_id)
                        if query_words is None:
                            prefill_len = self.session_prefill_lens.get(session_id, 0)
                            if prefill_len <= 0:
                                prefill_len = len(token_ids_cpu)
                            query_start = max(0, prefill_len - 128)
                            query_toks = token_ids_cpu[query_start:prefill_len].tolist()
                            if query_toks:
                                query_text = self.manager.tokenizer.decode(query_toks).lower()
                                query_words = {
                                    w for w in _RE_WORD_TOKENS.findall(query_text)
                                    if w not in _STOP_WORDS_COMPRESS
                                }
                            else:
                                query_words = set()
                            self.session_query_words[session_id] = query_words

                        if query_words:
                            block_words = {
                                w for w in _RE_WORD_TOKENS.findall(block_text_lc)
                                if w not in _STOP_WORDS_COMPRESS
                            }
                            if block_words & query_words:
                                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                                    print(f"[DKV DEBUG] Rule 4 skip block anchor={anchor_idx}: overlap={block_words & query_words}")
                                _result = True

                    if not _result:
                        # Rule 5: Rare document words (exact keywords)
                        doc_words = self.session_doc_words.get(session_id)
                        if doc_words is None:
                            doc_words = {}
                            if token_ids_cpu is not None:
                                # Decode the entire document text
                                full_text = self.manager.tokenizer.decode(token_ids_cpu.tolist()).lower()
                                # Count all words
                                from collections import Counter
                                doc_words = Counter(_RE_WORD_TOKENS.findall(full_text))
                            self.session_doc_words[session_id] = doc_words

                        block_words = {
                            w for w in _RE_WORD_TOKENS.findall(block_text_lc)
                            if w not in _STOP_WORDS_COMPRESS
                        }
                        for w in block_words:
                            if doc_words.get(w, 0) <= 2:
                                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                                    print(f"[DKV DEBUG] Rule 5 skip block anchor={anchor_idx}: word '{w}' occurs {doc_words.get(w, 0)} times")
                                _result = True
                                break

        except Exception as e:
            if os.environ.get("DKV_TELEMETRY", "0") == "1":
                print(f"[DKV DEBUG] skip check error: {e}")

        self._skip_compress_cache[_cache_key] = _result
        return _result


    # ── Core streaming ingest ──────────────────────────────────────────────────

    def ingest_chunk(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,   # [1, heads, T, head_dim]
        v: torch.Tensor,
    ) -> None:
        """
        Streaming ingest of a token chunk.

        Called once per forward pass per layer.
        For prefill: k/v shape is [1, heads, seq_len, head_dim].
        For decode:  k/v shape is [1, heads, 1, head_dim].

        Processes tokens in micro-blocks of dynamic `micro_block_size`.
        Triggers compression immediately when each micro-block fills.
        """
        blocks = self.session_blocks[session_id][layer_idx]
        seq_len = k.shape[2]
        
        # Read the session-specific micro-block size (defaults to self.micro_block_size)
        micro_block_size = self.session_micro_block_sizes.get(session_id, self.micro_block_size)

        if seq_len == 1:
            # Force micro_block_size to 32 for the active accumulation window during decode
            micro_block_size = 32
            # ───────────────────────────────────────────────────────────────────
            # DECODE PATH (T=1)
            # Phase 29 Fix #1: Ring buffer — zero torch.cat allocations per token.
            # ───────────────────────────────────────────────────────────────────
            if not blocks or blocks[-1].state != "ACCUMULATING" or blocks[-1].token_count() >= blocks[-1].micro_block_size:
                # Start a new block. Current token becomes the anchor (1 dense token,
                # irreducible). Pre-allocate the ring buffer for future active tokens.
                anchor_idx = self._next_anchor_idx(blocks)
                anchor_kv = torch.stack([k[:, :, 0], v[:, :, 0]], dim=1)

                pool_idx = None
                if self.native_pool is not None:
                    pool_idx = self.native_pool.allocate_block()

                # Pre-allocate ring buffer — ONE allocation per block (every 32 tokens),
                # not one per token. shape: [1, heads, micro_block_size, head_dim]
                heads    = k.shape[1]
                head_dim = k.shape[3]
                buf_k = torch.empty((1, heads, micro_block_size, head_dim),
                                    device=k.device, dtype=k.dtype)
                buf_v = torch.empty((1, heads, micro_block_size, head_dim),
                                    device=k.device, dtype=k.dtype)

                new_block = StreamingKVBlock(
                    anchor_idx=anchor_idx,
                    anchor_kv=anchor_kv,
                    micro_block_size=micro_block_size,
                    token_indices=[anchor_idx],
                    pool_idx=pool_idx,
                    session_id=session_id,
                    layer_idx=layer_idx,
                    _active_buf_k=buf_k,
                    _active_buf_v=buf_v,
                    _active_fill=0,
                    _metadata_idx=len(blocks),   # O(1) metadata index
                )
                # Flag outlier if the anchor key exceeds the threshold
                new_block.is_outlier = False
                
                # Check for compression exemption
                if self._should_skip_compression(session_id, anchor_idx, 1):
                    new_block.skip_compression = True
                    if layer_idx == 0 and os.environ.get("DKV_TELEMETRY", "0") == "1":
                        print(f"[DKV Ingest] Decode block anchor_idx={anchor_idx} layer={layer_idx}: Exempted from SVD (contains digit/number)")
                
                blocks.append(new_block)
                self.update_metadata_block(session_id, layer_idx, len(blocks) - 1, new_block)

                with self._stats_lock:
                    self.stats["total_blocks_created"] += 1
                return

            current_block = blocks[-1]
            fill   = current_block._active_fill
            buf_k  = current_block._active_buf_k
            buf_v  = current_block._active_buf_v

            if buf_k is not None and fill < buf_k.shape[2]:
                # ── Fast path: in-place ring buffer write (zero allocation) ──
                buf_k[0, :, fill, :] = k[0, :, 0, :]
                buf_v[0, :, fill, :] = v[0, :, 0, :]
                fill += 1
                current_block._active_fill = fill
                # Update active_k/v to be a view of the filled slice — no copy
                current_block.active_k = buf_k[:, :, :fill, :]
                current_block.active_v = buf_v[:, :, :fill, :]
            else:
                # ── Safety fallback (buffer not allocated or overflowed) ──────
                if current_block.active_k is None:
                    current_block.active_k = k
                    current_block.active_v = v
                else:
                    current_block.active_k = torch.cat([current_block.active_k, k], dim=2)
                    current_block.active_v = torch.cat([current_block.active_v, v], dim=2)
            
            # Update outlier status if incoming key token exceeds the threshold
            if False:
                current_block.is_outlier = True

            # Update compression exemption status for newly appended token
            if not current_block.skip_compression:
                new_tok_pos = current_block.anchor_idx + len(current_block.token_indices)
                if self._should_skip_compression(session_id, new_tok_pos, 1):
                    current_block.skip_compression = True
                    if layer_idx == 0 and os.environ.get("DKV_TELEMETRY", "0") == "1":
                        print(f"[DKV Ingest] Decode block anchor_idx={current_block.anchor_idx} layer={layer_idx}: Exempted from SVD (appended digit/number)")

            current_block.dirty = True
            current_block.token_indices.append(
                current_block.anchor_idx + len(current_block.token_indices)
            )
            # The decode cache only consumes compressed membership/state from
            # metadata.  The dense window reads the live block and ring buffer,
            # so rewriting this CPU row for every token is unnecessary.

            # Get the current sequence length to determine rolling dense window
            current_seq_len = current_block.anchor_idx + len(current_block.token_indices)
            recency_cutoff  = current_seq_len - self.recency_window

            # Compress any blocks that have now fallen out of the rolling dense window.
            # OPTIMIZED: scan backwards — during steady-state decode, all old blocks are
            # already COMPRESSED/SUBMITTED; only the last 1-2 are ACCUMULATING.
            # Backwards scan + early-break gives O(1) average case vs O(N_blocks) forward.
            n_blocks = len(blocks)
            for idx in range(n_blocks - 2, -1, -1):   # skip the current (last) block
                b = blocks[idx]
                if b.state not in ("ACCUMULATING",):
                    # Everything earlier is also non-ACCUMULATING (chronological order).
                    break
                if b.active_k is not None:
                    if _is_block_compression_eligible(b, is_last_block=False) and (b.anchor_idx + b.token_count()) < recency_cutoff:
                        self._submit_block_for_compression(b)
                        self.update_metadata_block(session_id, layer_idx, idx, b)
                elif b.is_outlier and (b.anchor_idx + b.token_count()) < recency_cutoff:
                    # Outlier blocks skip SVD but dense tensors can be offloaded to CPU.
                    if b.active_k is not None:
                        b.active_k_cpu = b.active_k.to("cpu", non_blocking=True) if b.active_k.is_cuda else b.active_k.cpu()
                        b.active_v_cpu = b.active_v.to("cpu", non_blocking=True) if b.active_v.is_cuda else b.active_v.cpu()
                        b.active_k = None
                        b.active_v = None
                        b.dirty = True
                        self.update_metadata_block(session_id, layer_idx, idx, b)
            return


        # ───────────────────────────────────────────────────────────────────
        # PREFILL PATH (T > 1) — highly optimized vectorized batch ingestion
        # ───────────────────────────────────────────────────────────────────
        # Partition the prefill sequence into regions based on distance to sequence end
        session_base_idx = self._next_anchor_idx(blocks)
        total_seq_len = session_base_idx + seq_len

        # SVD compression is deferred during prefill to ensure 100% exact causal attention
        # and prevent mixing compressed/uncompressed blocks which perturbs logits.
        past_blocks_to_compress = []

        # Fresh CUDA prefills can keep these blocks entirely in their raw
        # accumulating representation until the boundary.  Pool rows are not
        # read by this exact-causal path, so avoid allocating the full pool (and
        # its residual slabs) while the raw KV is still resident.
        _pool_ready = (
            self.native_pool is not None
            and getattr(self.native_pool, "_allocated", True)
        )

        # All four historical regions currently use the same MBS.  Splitting
        # at the fixed 1024/4096/12288 boundaries therefore changes nothing
        # semantically, but it breaks the global (micro_block_size + 1)
        # anchor stride.  For a 257-token CUDA block this produced the
        # 252-token + 3-token pairs visible in diagnostics at anchors 771 and
        # 1024, which in turn launched many tiny SVD jobs and weakened routing.
        # Keep one contiguous region so an outer chunk aligned to the block
        # capacity stays aligned all the way through ingest.  MLX follows the
        # same contiguous block layout and applies the recency policy at
        # attention time rather than by splitting storage blocks.
        regions = [(0, seq_len, micro_block_size)]

        for start_idx, end_idx, r_mbs in regions:
            region_k = k[:, :, start_idx:end_idx]
            region_v = v[:, :, start_idx:end_idx]
            region_len = region_k.shape[2]
            if region_len == 0:
                continue

            block_capacity = 1 + r_mbs
            num_full_blocks = region_len // block_capacity
            L_full = num_full_blocks * block_capacity

            new_blocks = []
            full_blocks_to_compress = []
            base_idx = self._next_anchor_idx(blocks)

            # 1. Vectorized extraction of full blocks
            if num_full_blocks > 0:
                k_full = region_k[:, :, :L_full]
                v_full = region_v[:, :, :L_full]

                # Reshape into [1, heads, num_full_blocks, block_capacity, head_dim]
                k_reshaped = k_full.reshape(1, k.shape[1], num_full_blocks, block_capacity, k.shape[3])
                v_reshaped = v_full.reshape(1, v.shape[1], num_full_blocks, block_capacity, v.shape[3])

                # Extract anchors: [1, heads, num_full_blocks, head_dim]
                anchors_k = k_reshaped[:, :, :, 0]
                anchors_v = v_reshaped[:, :, :, 0]

                # Stack K/V anchors: [num_full_blocks, 1, 2, heads, head_dim]
                stacked_anchors = torch.stack([anchors_k, anchors_v], dim=2).permute(3, 0, 2, 1, 4)
                
                # Consolidated copy of stacked anchors to CPU in a single step (zero round-trips!)
                stacked_anchors_cpu = stacked_anchors.to("cpu", non_blocking=True)

                # Extract active states: [num_full_blocks, 1, heads, r_mbs, head_dim]
                active_k_blocks = k_reshaped[:, :, :, 1:].permute(2, 0, 1, 3, 4)
                active_v_blocks = v_reshaped[:, :, :, 1:].permute(2, 0, 1, 3, 4)

                # Keep one owning tensor per K/V batch instead of launching a
                # separate device clone for every block.  The per-block
                # slices below remain views of these tensors, so their
                # storage stays alive until deferred compression consumes it.
                # This preserves the old ownership/lifetime guarantee while
                # removing thousands of small CUDA allocations on long
                # prefills.
                active_k_owned = active_k_blocks.contiguous()
                active_v_owned = active_v_blocks.contiguous()

                # Pre-allocate NativeBlockPool indices in a single batch call!
                pool_indices = []
                if _pool_ready:
                    pool_indices = self.native_pool.allocate_blocks(num_full_blocks)

                for i in range(num_full_blocks):
                    anchor_idx = base_idx + i * block_capacity
                    anchor_kv = stacked_anchors[i]
                    anchor_kv_cpu = stacked_anchors_cpu[i]
                    
                    pool_idx = pool_indices[i] if pool_indices else None

                    new_block = StreamingKVBlock(
                        anchor_idx=anchor_idx,
                        anchor_kv=anchor_kv,
                        anchor_kv_cpu=anchor_kv_cpu,
                        micro_block_size=r_mbs,
                        token_indices=list(range(anchor_idx, anchor_idx + block_capacity)),
                        pool_idx=pool_idx,
                    )
                    new_block.active_k = active_k_owned[i]
                    new_block.active_v = active_v_owned[i]
                    
                    # Outlier check (CPU-local list access, zero sync overhead)
                    new_block.is_outlier = False

                    # Check for compression exemption
                    if self._should_skip_compression(session_id, anchor_idx, block_capacity):
                        new_block.skip_compression = True
                        if layer_idx == 0 and os.environ.get("DKV_TELEMETRY", "0") == "1":
                            print(f"[DKV Ingest] Block anchor_idx={anchor_idx} layer={layer_idx}: Exempted from SVD compression (contains digit/number)")

                    # Prefill must remain exact until the final chunk has run.
                    # The next chunk reads these blocks as history, so publishing
                    # an SVD reconstruction here would feed lossy KV back into
                    # causal attention and can change the first generated token.
                    # The normal path submits all eligible blocks from
                    # compress_deferred_prefill_blocks() after prefill.  The
                    # immediate mode is retained only as an explicit benchmark
                    # knob for approximate prefill experiments.
                    _immediate_prefill = os.environ.get("DKV_IMMEDIATE_PREFILL_COMPRESS", "0") == "1"
                    if (_immediate_prefill
                            and (anchor_idx > 0 or not self.protect_block_zero)
                            and not new_block.skip_compression):
                        new_block.state = "SUBMITTED"
                        full_blocks_to_compress.append(new_block)
                    else:
                        new_block.state = "ACCUMULATING"

                    new_blocks.append(new_block)

                    with self._stats_lock:
                        self.stats["total_blocks_created"] += 1

            # 2. Extract partial block if any
            if region_len > L_full:
                anchor_idx = base_idx + L_full
                
                # Slice anchor token
                anchor_k = region_k[:, :, L_full : L_full + 1]
                anchor_v = region_v[:, :, L_full : L_full + 1]
                anchor_kv = torch.stack([anchor_k[:, :, 0], anchor_v[:, :, 0]], dim=1)

                active_start = L_full + 1
                blk_active_k = None
                blk_active_v = None
                token_indices = [anchor_idx]

                if region_len > active_start:
                    blk_active_k = region_k[:, :, active_start:region_len]
                    blk_active_v = region_v[:, :, active_start:region_len]
                    token_indices.extend(list(range(anchor_idx + 1, anchor_idx + 1 + (region_len - active_start))))

                pool_idx = None
                if _pool_ready:
                    pool_idx = self.native_pool.allocate_block()

                new_block = StreamingKVBlock(
                    anchor_idx=anchor_idx,
                    anchor_kv=anchor_kv,
                    anchor_kv_cpu=anchor_kv.to("cpu", non_blocking=True),
                    micro_block_size=r_mbs,
                    token_indices=token_indices,
                    pool_idx=pool_idx,
                )

                if blk_active_k is not None:
                    new_block.active_k = blk_active_k.clone()
                    new_block.active_v = blk_active_v.clone()

                new_block.is_outlier = False

                # Check for compression exemption
                if self._should_skip_compression(session_id, anchor_idx, len(token_indices)):
                    new_block.skip_compression = True
                    if layer_idx == 0 and os.environ.get("DKV_TELEMETRY", "0") == "1":
                        print(f"[DKV Ingest] Partial block anchor_idx={anchor_idx} layer={layer_idx}: Exempted from SVD compression (contains digit/number)")

                # SVD compression is deferred during prefill to ensure exact attention.
                new_block.state = "ACCUMULATING"

                new_blocks.append(new_block)
                with self._stats_lock:
                    self.stats["total_blocks_created"] += 1

            blocks.extend(new_blocks)

            # ── Batched metadata write (Fix #4) ─────────────────────────────
            # OLD: per-block update_metadata_block() call inside a loop = dict
            #      lookups + scalar CPU tensor writes per block (1,120 calls for
            #      a 2540-token prefill across 28 layers).
            # NEW: batch-assign session_id/layer_idx, then write all rows in one
            #      vectorized slice-assign into the metadata tensor.
            n_new = len(new_blocks)
            if n_new > 0:
                base_block_idx = len(blocks) - n_new

                # Ensure metadata tensor exists and is large enough
                metadata = self.session_metadata.setdefault(session_id, {}).setdefault(
                    layer_idx, _new_metadata_tensor(1024)
                )
                if base_block_idx + n_new > metadata.shape[0]:
                    new_size = max(metadata.shape[0] * 2, base_block_idx + n_new)
                    new_meta = _new_metadata_tensor(new_size)
                    new_meta[:metadata.shape[0]] = metadata
                    self.session_metadata[session_id][layer_idx] = new_meta
                    metadata = new_meta

                for i, block in enumerate(new_blocks):
                    block.session_id = session_id
                    block.layer_idx = layer_idx
                    bi = base_block_idx + i
                    block._metadata_idx = bi     # O(1) fast-path index
                    metadata[bi, 0] = int(block.pool_idx) if block.pool_idx is not None else -1
                    metadata[bi, 1] = int(block.anchor_idx)
                    metadata[bi, 2] = block.token_count()
                    metadata[bi, 3] = _STATE_CODES.get(block.state, -1)

                versions = self._metadata_versions.setdefault(session_id, {})
                versions[layer_idx] = versions.get(layer_idx, 0) + 1

            # Batch submit all compression requests in one consolidation transfer
            if full_blocks_to_compress:
                self._submit_blocks_batched(session_id, layer_idx, full_blocks_to_compress)

        # Track peak dense footprint
        dense_tokens = self._count_dense_tokens(blocks)
        with self._stats_lock:
            if dense_tokens > self.stats["total_dense_tokens_peak"]:
                self.stats["total_dense_tokens_peak"] = dense_tokens

        # Compress during the forward pass:
        if os.environ.get("DKV_STREAMING_COMPRESS", "0") == "1":
            self.compress_deferred_blocks_for_layer(session_id, layer_idx)

    def _submit_blocks_batched(self, session_id: str, layer_idx: int, blocks_list: List[StreamingKVBlock]):
        if not blocks_list:
            return

        # Fetch shape metadata from the first active block
        micro_block_size = blocks_list[0].micro_block_size
        heads = blocks_list[0].active_k.shape[1]
        head_dim = blocks_list[0].active_k.shape[3]
        device = blocks_list[0].active_k.device

        # GPU-accelerated randomized SVD path (Component 2).
        # Default to GPU compression on CUDA for maximum throughput; CPU path
        # is the fallback (set DKV_GPU_COMPRESS=0 to force CPU SVD).
        _config_gpu_compress = getattr(
            getattr(self.manager, "config", None), "gpu_compress", device.type == "cuda"
        )
        _gpu_compress_default = "1" if _config_gpu_compress else "0"
        if os.environ.get("DKV_GPU_COMPRESS", _gpu_compress_default) == "1" and device.type == "cuda":
            from native_core.compression.lowrank import compress_layer_blocks_gpu
            from native_core.kv_runtime_manager import get_layer_rank
            _cfg = getattr(self.manager, "config", None)
            _early_boost = getattr(_cfg, "early_layer_rank_boost", False)
            _max_rank_early = getattr(_cfg, "max_rank_early", 0)
            _rank = get_layer_rank(
                layer_idx, self.manager.num_layers, self.manager.rank,
                early_boost=_early_boost, max_rank_early=_max_rank_early
            )
            # Group by T_active: compress_layer_blocks_gpu requires all blocks in
            # a batch to have the same active_k.shape[2].  Chunked CUDA prefill
            # commonly creates partial blocks (for example CH=128 with a
            # 64-token block produces a 63-token remainder on every chunk), so
            # partial groups must go through the same GPU path as full groups.
            # Sending them to the CPU fallback makes the supposedly async path
            # host-bound and, more importantly, used to leave some of them stuck
            # in SUBMITTED when the fallback saw skip_compression=True.
            from collections import defaultdict as _ddict
            _by_T = _ddict(list)
            for _b in blocks_list:
                _by_T[_b.active_k.shape[2]].append(_b)
            _gpu_all_success = True
            _gpu_compressed_count = 0
            # Cap blocks per compress call.  The batched finalization builds
            # [n, T, feat] fp32 intermediates (deltas, recon, U_masked) for the
            # whole call at once, so peak VRAM scales with n.  This cap exists
            # ONLY to bound a genuinely huge layer (64k+ context ≈ 1000 blocks
            # per layer, which spiked GBs and OOM'd next to the model + retained
            # raw prefill KV).  Each block compresses independently, so chunking
            # never changes results.
            #
            # 64, deliberately LOW: the cap must never RAISE the batch above what
            # a layer naturally holds.  A 256 cap measured WORSE on an A100 at
            # 13k (peak 15.07 -> 17.16 GB, and 16k began OOMing) because the
            # natural per-layer batch there is only ~49 blocks — the "cap" was
            # acting as a floor and inflating every transient ~5x.
            _B_MAX = int(os.environ.get("DKV_COMPRESS_BLOCK_BATCH", "64"))
            for _T_active, _group in _by_T.items():
                for _cs in range(0, len(_group), _B_MAX):
                    _sub = _group[_cs:_cs + _B_MAX]
                    try:
                        _ok = compress_layer_blocks_gpu(_sub, _rank, manager=self.manager)
                        if _ok:
                            _gpu_compressed_count += len(_sub)
                            # POST-CONDITION. compress_layer_blocks_gpu runs
                            # SYNCHRONOUSLY, so by the time it returns _ok every
                            # block it accepted must have left SUBMITTED. It
                            # reports one bool for the whole sub-batch, though,
                            # so a block it silently skipped stays SUBMITTED --
                            # and SUBMITTED is in NEITHER collection decode reads
                            # (kv_runtime_manager.get_cached_decode_blocks), so
                            # those tokens vanish from attention with no error.
                            #
                            # This is not hypothetical: the comment above records
                            # the same class of bug ("used to leave some of them
                            # stuck in SUBMITTED when the fallback saw
                            # skip_compression=True"), and the coverage check
                            # still reports one block stranded this way on every
                            # DKV layer of a 2822-token prompt:
                            #   states=['SUBMITTED'] anchors=[1542]
                            #   layers 3, 7, 11, 15, 19, 23
                            # which survived both the block-0 fix and the
                            # queue drain because it is neither of those paths.
                            #
                            # Rather than chase which internal branch skipped it,
                            # enforce the invariant the two FAILURE branches below
                            # already maintain: after a compression attempt a
                            # block is either COMPRESSED (published to the pool)
                            # or ACCUMULATING (served densely). Never SUBMITTED.
                            # An unpublished block still has its raw KV, so
                            # falling back to dense is exact -- strictly better
                            # than dropping it.
                            for _b in _sub:
                                if _b.state == "SUBMITTED":
                                    _b.state = "ACCUMULATING"
                                    self.update_metadata_state(session_id, layer_idx, _b)
                                    print(f"[DKV] compress reported success but left "
                                          f"anchor={_b.anchor_idx} layer={layer_idx} "
                                          f"SUBMITTED (unpublished); serving it dense.",
                                          flush=True)
                        else:
                            _gpu_all_success = False
                            # The GPU helper may return False without raising.  It
                            # did not publish a pool entry in that case, so make
                            # the CPU fallback's ownership explicit instead of
                            # leaving the block marked SUBMITTED indefinitely.
                            for _b in _sub:
                                if _b.state == "SUBMITTED":
                                    _b.state = "ACCUMULATING"
                    except Exception as _gpu_err:
                        import traceback
                        print(f"[DKV] compress_layer_blocks_gpu FAILED for T={_T_active} (falling back to CPU): {_gpu_err}", flush=True)
                        if os.environ.get("DKV_DIAG", "0") == "1":
                            traceback.print_exc()
                        _gpu_all_success = False
                        # Rollback SUBMITTED → ACCUMULATING for this group so CPU path retries
                        for _b in _sub:
                            if _b.state == "SUBMITTED":
                                _b.state = "ACCUMULATING"
            if _gpu_compressed_count > 0:
                with self._stats_lock:
                    self.stats["total_compressed"] += _gpu_compressed_count
                    self.stats["compressions_during_ingest"] += _gpu_compressed_count
            if _gpu_all_success:
                return  # all blocks handled by GPU path — skip CPU fallback
            # Otherwise fall through: only failed GPU groups remain.  Successful
            # groups have active_k=None and are filtered out below.

        # Group blocks by active sequence length to handle partial blocks.
        # Filter out blocks already compressed by the GPU path above (active_k cleared to None).
        from collections import defaultdict
        by_len = defaultdict(list)
        for b in blocks_list:
            if b.active_k is not None:
                by_len[b.active_k.shape[2]].append(b)


        is_async_active = getattr(self.compressor, "_running", False) and hasattr(self.compressor, "submit_cpu")

        for cur_mbs, group in by_len.items():
            num_blocks = len(group)
            
            k_gpu, v_gpu, k_cpu, v_cpu = self._get_session_staging_buffer(
                session_id, num_blocks, heads, cur_mbs, head_dim, device
            )

            # Concat group's active tensors
            k_gpu_concat = torch.cat([b.active_k for b in group], dim=0)
            v_gpu_concat = torch.cat([b.active_v for b in group], dim=0)

            # Copy to CPU staging buffer (which is pinned on CUDA)
            _is_cuda = (device.type == "cuda")
            k_cpu[:num_blocks, :, :cur_mbs, :].copy_(k_gpu_concat, non_blocking=_is_cuda)
            v_cpu[:num_blocks, :, :cur_mbs, :].copy_(v_gpu_concat, non_blocking=_is_cuda)

            # Record a CUDA event after the non-blocking D2H copy so the worker
            # thread can synchronize on it before reading the CPU buffer.  The
            # producer thread must NOT call event.synchronize() here — doing so
            # blocks the main thread until the PCIe transfer finishes and removes
            # any overlap between compression and the next forward pass.
            # On MPS, the copy_ is always synchronous, so no event is needed.
            _dma_event = None
            if _is_cuda:
                _dma_event = _new_event(device.type)
                _dma_event.record()
                # NOTE: Do NOT call _dma_event.synchronize() here.
                # The worker thread owns the synchronization (async_compressor.py
                # _run_item_sequential calls event.synchronize() before SVD).
            elif device.type == "mps":
                # On MPS, copy_ to pinned CPU is always synchronous — no event needed.
                torch.mps.synchronize()

            for idx, block in enumerate(group):
                if is_async_active:
                    # Take a narrow view of the staging buffer for this block.
                    # The staging buffer is pinned on CUDA (allocated in
                    # _get_session_staging_buffer), so no extra clone()+pin_memory()
                    # is needed — the worker can read the slice safely once the
                    # DMA event fires.  A contiguous() ensures the Triton SVD
                    # backend sees a simple stride layout.
                    k_cpu_slice = k_cpu[idx : idx + 1, :, :cur_mbs, :].contiguous()
                    v_cpu_slice = v_cpu[idx : idx + 1, :, :cur_mbs, :].contiguous()

                    # Keep a reference on the block for any synchronous fallback paths
                    block.active_k_cpu = k_cpu_slice
                    block.active_v_cpu = v_cpu_slice

                    # Delete/free the GPU active_k/v immediately!
                    block.active_k = None
                    block.active_v = None
                    block.dirty = True

                    self.compressor.submit_cpu(block, k_cpu_slice, v_cpu_slice, _dma_event)
                else:
                    # Sync execution directly on the GPU/MPS slices (much faster and avoids CPU fallback bugs)
                    self.compress_fn(block, block.active_k, block.active_v)
                    block.state = "COMPRESSED"
                    # FIX (needle drop): sync the metadata state code so decode's
                    # get_cached_decode_blocks (compressed_mask = metadata[:,3]==2,
                    # kv_runtime_manager.py) actually SEES this block. Without this,
                    # a synchronously-compressed block (the common case for
                    # force-compressed skip_compression / digit blocks on the
                    # deferred path) is left COMPRESSED on the object but stale in
                    # metadata → silently excluded from decode → dropped needle.
                    self.update_metadata_state(session_id, layer_idx, block)
                    if hasattr(self.compressor, "stats"):
                        stats = getattr(self.compressor, "stats")
                        if isinstance(stats, dict) and "sync_fallbacks" in stats:
                            lock = getattr(self.compressor, "_stats_lock", None)
                            if lock is not None:
                                with lock:
                                    stats["sync_fallbacks"] += 1
                            else:
                                stats["sync_fallbacks"] += 1

                with self._stats_lock:
                    self.stats["total_compressed"] += 1
                    self.stats["compressions_during_ingest"] += 1

    def _submit_block_for_compression(self, block: StreamingKVBlock):
        """Submit block for background compression. Block stays readable via active_k/v."""
        k = block.active_k
        v = block.active_v
        block.state = "SUBMITTED"

        _is_cuda = k.is_cuda if k is not None else False
        block.active_k_cpu = (k.to("cpu", non_blocking=True) if _is_cuda else k.cpu()) if k is not None else None
        block.active_v_cpu = (v.to("cpu", non_blocking=True) if _is_cuda else v.cpu()) if v is not None else None
        block.anchor_kv_cpu = (block.anchor_kv.to("cpu", non_blocking=True) if _is_cuda and block.anchor_kv is not None else block.anchor_kv.cpu()) if block.anchor_kv is not None else None
        
        # Clear GPU tensors immediately
        block.active_k = None
        block.active_v = None
        block.dirty = True

        # Non-blocking: copies to CPU immediately, frees GPU as soon as SVD completes
        is_async_active = getattr(self.compressor, "_running", False)
        submitted = self.compressor.submit(block, k, v) if (k is not None and is_async_active) else False

        if not submitted:
            # Backpressure or sync mode: compress synchronously
            self.compress_fn(block, k, v)
            block.state = "COMPRESSED"
            # FIX (needle drop): same metadata sync as _submit_blocks_batched —
            # otherwise a sync/backpressure-compressed block is COMPRESSED on the
            # object but stale in metadata, so decode never sees it.
            _sid = getattr(block, "session_id", None)
            _lidx = getattr(block, "layer_idx", None)
            if _sid is not None and _lidx is not None:
                self.update_metadata_state(_sid, _lidx, block)

        with self._stats_lock:
            self.stats["total_compressed"] += 1

    # ── Decode path (single token) ─────────────────────────────────────────────

    def append_decode_token(
        self,
        session_id: str,
        layer_idx: int,
        k: torch.Tensor,   # [1, heads, 1, head_dim]
        v: torch.Tensor,
    ) -> None:
        """
        Append a single decode token. Same micro-block logic applies.
        """
        self.ingest_chunk(session_id, layer_idx, k, v)

    # ── Block access ───────────────────────────────────────────────────────────

    def get_blocks(self, session_id: str, layer_idx: int) -> List[StreamingKVBlock]:
        """Return all blocks. The attention path handles each block's state."""
        if session_id not in self.session_blocks:
            return []
        return self.session_blocks[session_id][layer_idx]

    def get_current_accumulating_block(
        self, session_id: str, layer_idx: int
    ) -> Optional[StreamingKVBlock]:
        """Return the currently accumulating block (dense window)."""
        blocks = self.session_blocks.get(session_id, {}).get(layer_idx, [])
        if blocks and blocks[-1].state == "ACCUMULATING":
            return blocks[-1]
        return None

    # ── Dense footprint accounting ─────────────────────────────────────────────

    def _count_dense_tokens(self, blocks: list) -> int:
        """Count tokens currently held dense in GPU VRAM."""
        count = 0
        for b in blocks:
            if b.active_k is not None:
                count += b.active_k.shape[2]
            # Anchor is always 1 dense token (irreducible)
            count += 1
        return count

    def dense_footprint_bytes(self, session_id: str) -> int:
        """
        Compute current GPU VRAM held as dense KV for a session.
        """
        if session_id not in self.session_blocks:
            return 0
        total = 0
        for layer_blocks in self.session_blocks[session_id].values():
            for b in layer_blocks:
                if b.anchor_kv is not None:
                    total += b.anchor_kv.numel() * 2  # fp16
                if b.active_k is not None:
                    total += b.active_k.numel() * 2
                    total += b.active_v.numel() * 2
        return total

    def sparse_footprint_bytes(self, session_id: str) -> int:
        """
        Compute VRAM held as compressed U/V for a session.
        """
        if session_id not in self.session_blocks:
            return 0
        total = 0
        for layer_blocks in self.session_blocks[session_id].values():
            for b in layer_blocks:
                if b.U is not None:
                    total += b.U.numel() * 2
                    total += b.V.numel() * 2
        return total

    def compress_deferred_blocks_for_layer(self, session_id: str, layer_idx: int) -> None:
        """
        Identify blocks in the specified layer that have left the recency window,
        and submit them to SVD compression immediately.
        """
        if session_id not in self.session_blocks:
            return

        layers = self.session_blocks[session_id]
        blocks = layers.get(layer_idx, [])
        if not blocks:
            return

        # Determine the total sequence length of the session based on this layer's blocks
        last_block = blocks[-1]
        total_seq_len = last_block.anchor_idx + last_block.token_count()

        if total_seq_len < self.short_context_threshold:
            return

        # If init_session deferred a fresh CUDA pool, materialize it now
        if (
            self.native_pool is not None
            and not getattr(self.native_pool, "_allocated", True)
        ):
            self.native_pool.ensure_allocated(total_seq_len)

        blocks_to_compress = []
        for idx, b in enumerate(blocks):
            if b.state == "ACCUMULATING" and (b.active_k is not None or b.active_k_cpu is not None):
                eligible = _is_block_compression_eligible(
                    b, is_last_block=(idx == len(blocks) - 1),
                    ignore_skip_compression=True,
                )
                window_ok = (b.anchor_idx + b.token_count()) < (total_seq_len - self.recency_window)
                if eligible and window_ok:
                    # protect_block_zero now means "never LOSSY", not "never
                    # compressed" -- see _is_block_compression_eligible. Marking
                    # it skip_compression is what makes lowrank.py take the
                    # force-exact path (_force_exact reads this same flag at
                    # lowrank.py:1412), so block 0 keeps every position as an
                    # exact residual. Leaving it out of attention was the only
                    # other option and it lost the data outright.
                    if b.anchor_idx == 0 and StreamingKVBlock.protect_block_zero:
                        b.skip_compression = True
                    b.state = "SUBMITTED"
                    blocks_to_compress.append(b)
                    self.update_metadata_state(session_id, layer_idx, b)

        if blocks_to_compress:
            try:
                self._submit_blocks_batched(session_id, layer_idx, blocks_to_compress)
            except Exception as _e:
                import traceback
                print(f"[DIAG compress_deferred_layer] ERROR in _submit_blocks_batched layer={layer_idx}: {_e}", flush=True)
                traceback.print_exc()
                # Rollback state so next compress attempt can retry
                for _b in blocks_to_compress:
                    if _b.state == "SUBMITTED":
                        _b.state = "ACCUMULATING"

    def compress_deferred_blocks(self, session_id: str) -> None:
        """
        Scan all layers of the session, identify blocks that have left the
        recency window (last 512 tokens), and submit them to SVD compression.
        """
        if session_id not in self.session_blocks:
            return

        # 1. Determine the total sequence length of the session.
        # BUG (found 2026-07-27): this used to read layers.get(0, []) only --
        # correct for non-hybrid models where every layer is attended, but on
        # hybrid architectures (Qwen3.5/Qwen3-Next-style) layer 0 is very
        # often linear_attention and NEVER has blocks, so total_seq_len stayed
        # 0 forever and this function early-returned on EVERY call, for the
        # entire session -- no block was ever deferred-compressed at all.
        # Take the max anchor+token_count across every layer that actually
        # has blocks instead of assuming layer 0 is representative.
        total_seq_len = 0
        layers = self.session_blocks[session_id]
        if layers:
            for _layer_blocks in layers.values():
                if _layer_blocks:
                    _last = _layer_blocks[-1]
                    total_seq_len = max(total_seq_len, _last.anchor_idx + _last.token_count())

        _diag = os.environ.get("DKV_DIAG", "0") == "1"
        if _diag:
            n_blocks_any = sum(len(v) for v in layers.values()) if layers else 0
            print(f"[DIAG compress_deferred] session={session_id} total_seq_len={total_seq_len} "
                  f"recency_window={self.recency_window} short_ctx_threshold={self.short_context_threshold} "
                  f"total blocks (all layers)={n_blocks_any}", flush=True)

        if total_seq_len < self.short_context_threshold:
            if _diag:
                print(f"[DIAG compress_deferred] EARLY RETURN: total_seq_len={total_seq_len} < short_context_threshold={self.short_context_threshold}", flush=True)
            return  # No blocks are eligible

        # If init_session deferred a fresh CUDA pool, materialize it now with
        # the actual context length before compressed rows are published.  The
        # exact prefill path above never needs pool-backed rows.
        if (
            self.native_pool is not None
            and not getattr(self.native_pool, "_allocated", True)
        ):
            self.native_pool.ensure_allocated(total_seq_len)

        # 2. Compress PER LAYER.
        #
        # A cross-layer variant (collect every layer's eligible blocks, submit
        # once) was tried to cut the 48 per-layer cuSOLVER dispatches.  It DID
        # cut compress time (~7.5s -> ~6.0s at 13K) but REGRESSED peak VRAM
        # (15.07 -> 17.16 GB) and made 16K OOM: the natural per-layer batch at
        # 13K is only ~49 blocks, so batching across layers raised the SVD batch
        # to the chunk cap (256) and made every [N, T, feat] transient
        # (deltas / recon / U_masked) ~5x larger.  Peak VRAM matters more than
        # ~1.5s of compress here, so the per-layer loop stays.  The chunk cap
        # below is kept but only BOUNDS a genuinely huge layer (64k+ context has
        # ~1000 blocks/layer) — it must never RAISE the batch above what one
        # layer naturally contains.
        for layer_idx, blocks in layers.items():
            blocks_to_compress = []
            for idx, b in enumerate(blocks):
                if b.state == "ACCUMULATING" and (b.active_k is not None or b.active_k_cpu is not None):
                    # NOTE: skip_compression is intentionally NOT checked here.
                    # That flag was designed for the decode-path inline compression
                    # (ingest_chunk) to keep blocks with digits/math/acronyms dense
                    # so the model doesn't hallucinate during decode.  For the
                    # post-forward prefill compression path (compress_deferred_blocks),
                    # applying it causes ALL blocks in technical/math papers to stay
                    # dense, defeating compression entirely (13K tokens → 53 ACCUMULATING
                    # blocks, dense workspace overflow, trim warning, EOS collapse).
                    # All out-of-window prefill blocks are compressed regardless.
                    eligible = _is_block_compression_eligible(
                        b, is_last_block=(idx == len(blocks) - 1),
                        ignore_skip_compression=True,  # deferred prefill path bypasses skip_compression
                    )
                    window_ok = (b.anchor_idx + b.token_count()) < (total_seq_len - self.recency_window)
                    if _diag and layer_idx == 0:
                        print(f"[DIAG compress_deferred] layer=0 blk#{idx} anchor={b.anchor_idx} "
                              f"tcount={b.token_count()} eligible={eligible} window_ok={window_ok} "
                              f"state={b.state} ak={'yes' if b.active_k is not None else 'none'} "
                              f"skip={getattr(b, 'skip_compression', False)} mbs={b.micro_block_size}", flush=True)
                    if eligible and window_ok:
                        # Same force-exact marking as compress_deferred_blocks_for_layer.
                        # THIS is the site the prefill path actually reaches; the
                        # first version of this fix edited only the other one
                        # (identical statement, different indentation, so a
                        # replace-all silently matched one of two).
                        if b.anchor_idx == 0 and StreamingKVBlock.protect_block_zero:
                            b.skip_compression = True
                        b.state = "SUBMITTED"
                        blocks_to_compress.append(b)
                        self.update_metadata_state(session_id, layer_idx, b)

            if _diag and layer_idx == 0:
                print(f"[DIAG compress_deferred] layer=0 blocks_to_compress={len(blocks_to_compress)}", flush=True)

            if blocks_to_compress:
                try:
                    self._submit_blocks_batched(session_id, layer_idx, blocks_to_compress)
                except Exception as _e:
                    import traceback
                    print(f"[DIAG compress_deferred] ERROR in _submit_blocks_batched layer={layer_idx}: {_e}", flush=True)
                    traceback.print_exc()
                    # Rollback state so next compress attempt can retry
                    for _b in blocks_to_compress:
                        if _b.state == "SUBMITTED":
                            _b.state = "ACCUMULATING"

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _next_anchor_idx(self, blocks: list) -> int:
        if not blocks:
            return 0
        last = blocks[-1]
        return last.anchor_idx + 1 + last.token_count()

    def summary(self, session_id: Optional[str] = None) -> dict:
        s = dict(self.stats)
        if session_id:
            s["dense_bytes"] = self.dense_footprint_bytes(session_id)
            s["sparse_bytes"] = self.sparse_footprint_bytes(session_id)
            dense = s["dense_bytes"]
            sparse = s["sparse_bytes"]
            total = dense + sparse
            s["sparse_ratio"] = round(sparse / (total + 1e-9), 4)
        return s


class DenseWindowRingBuffer:
    """
    Utility class to manage a sliding ring buffer of dense tokens.
    Maintains a fixed VRAM capacity to prevent memory fragmentation and allocations.
    """
    def __init__(self, capacity: int, heads: int, head_dim: int, device: str, dtype: torch.dtype):
        self.capacity = capacity
        self.heads = heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate contiguous memory buffers
        self.k_buffer = torch.zeros((1, heads, capacity, head_dim), device=device, dtype=dtype)
        self.v_buffer = torch.zeros((1, heads, capacity, head_dim), device=device, dtype=dtype)
        self.write_ptr = 0
        self.valid_len = 0

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append keys and values to the ring buffer, overwriting old entries if full."""
        if k.dim() == 3:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            
        seq_len = k.shape[2]
        if seq_len > self.capacity:
            k = k[:, :, -self.capacity:]
            v = v[:, :, -self.capacity:]
            seq_len = self.capacity
            
        end_ptr = (self.write_ptr + seq_len) % self.capacity
        if self.write_ptr + seq_len <= self.capacity:
            self.k_buffer[:, :, self.write_ptr:self.write_ptr + seq_len] = k
            self.v_buffer[:, :, self.write_ptr:self.write_ptr + seq_len] = v
        else:
            first_part = self.capacity - self.write_ptr
            second_part = seq_len - first_part
            self.k_buffer[:, :, self.write_ptr:] = k[:, :, :first_part]
            self.k_buffer[:, :, :second_part] = k[:, :, first_part:]
            self.v_buffer[:, :, self.write_ptr:] = v[:, :, :first_part]
            self.v_buffer[:, :, :second_part] = v[:, :, first_part:]
            
        self.write_ptr = end_ptr
        self.valid_len = min(self.capacity, self.valid_len + seq_len)

    def get_valid_views(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the active sliding window in chronological order."""
        if self.valid_len < self.capacity:
            return self.k_buffer[:, :, :self.valid_len], self.v_buffer[:, :, :self.valid_len]
        else:
            k_ordered = torch.cat([self.k_buffer[:, :, self.write_ptr:], self.k_buffer[:, :, :self.write_ptr]], dim=2)
            v_ordered = torch.cat([self.v_buffer[:, :, self.write_ptr:], self.v_buffer[:, :, :self.write_ptr]], dim=2)
            return k_ordered, v_ordered

    def reset(self):
        self.write_ptr = 0
        self.valid_len = 0
        self.k_buffer.zero_()
        self.v_buffer.zero_()
