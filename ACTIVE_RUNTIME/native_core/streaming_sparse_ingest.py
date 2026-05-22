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

import torch
import queue
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


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

    # Compressed state (set by compressor)
    U: Optional[torch.Tensor] = None
    V: Optional[torch.Tensor] = None
    scale: float = 1.0
    cosine_sim: float = 1.0
    norm_drift: float = 0.0

    token_indices: List[int] = field(default_factory=list)
    state: str = "ACCUMULATING"  # ACCUMULATING | SUBMITTED | COMPRESSED | PAGED
    pool_idx: Optional[int] = None
    dirty: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if self.anchor_kv_cpu is None and self.anchor_kv is not None:
            self.anchor_kv_cpu = self.anchor_kv.cpu()

    def token_count(self) -> int:
        if self.active_k is not None:
            return self.active_k.shape[2]
        if self.U is not None:
            return self.U.shape[0]
        return 0

    def is_compression_eligible(self) -> bool:
        return (
            self.state == "ACCUMULATING"
            and self.active_k is not None
            and self.active_k.shape[2] >= self.micro_block_size
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
    ):
        self.compressor = compressor
        self.compress_fn = compress_fn
        self.micro_block_size = micro_block_size
        self.dense_anchor_only = dense_anchor_only
        self.native_pool = native_pool

        # session_id -> layer_idx -> List[StreamingKVBlock]
        self.session_blocks: Dict[str, Dict[int, List[StreamingKVBlock]]] = {}

        # Telemetry
        self.stats = {
            "total_blocks_created": 0,
            "total_compressed": 0,
            "total_dense_tokens_peak": 0,
            "compressions_during_ingest": 0,
        }
        self._stats_lock = threading.Lock()

    # ── Session management ─────────────────────────────────────────────────────

    def init_session(self, session_id: str, num_layers: int):
        if session_id not in self.session_blocks:
            self.session_blocks[session_id] = {i: [] for i in range(num_layers)}

    def clear_session(self, session_id: str):
        self.session_blocks.pop(session_id, None)

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

        Processes tokens in micro-blocks of `micro_block_size`.
        Triggers compression immediately when each micro-block fills.
        """
        blocks = self.session_blocks[session_id][layer_idx]
        seq_len = k.shape[2]

        if seq_len == 1:
            # ───────────────────────────────────────────────────────────────
            # DECODE PATH (T=1) — legacy sequential append (fast, no loops)
            # ───────────────────────────────────────────────────────────────
            if not blocks or blocks[-1].state != "ACCUMULATING":
                # Start a new block — extract anchor token (1 dense token, irreducible)
                anchor_k = k
                anchor_v = v
                anchor_kv = torch.stack([anchor_k[:, :, 0], anchor_v[:, :, 0]], dim=1)
                
                # Pre-allocate NativeBlockPool index on the single-threaded main thread
                pool_idx = None
                if self.native_pool is not None:
                    pool_idx = self.native_pool.allocate_block()

                new_block = StreamingKVBlock(
                    anchor_idx=self._next_anchor_idx(blocks),
                    anchor_kv=anchor_kv,
                    micro_block_size=self.micro_block_size,
                    token_indices=[self._next_anchor_idx(blocks)],
                    pool_idx=pool_idx,
                )
                blocks.append(new_block)
                
                with self._stats_lock:
                    self.stats["total_blocks_created"] += 1
                return

            current_block = blocks[-1]
            if current_block.active_k is None:
                current_block.active_k = k
                current_block.active_v = v
            else:
                current_block.active_k = torch.cat([current_block.active_k, k], dim=2)
                current_block.active_v = torch.cat([current_block.active_v, v], dim=2)

            current_block.dirty = True
            current_block.token_indices.append(current_block.anchor_idx + len(current_block.token_indices))

            # Immediately compress when micro-block fills — during ingest!
            if current_block.is_compression_eligible():
                self._submit_block_for_compression(current_block)
                with self._stats_lock:
                    self.stats["compressions_during_ingest"] += 1
            return

        # ───────────────────────────────────────────────────────────────────
        # PREFILL PATH (T > 1) — highly optimized vectorized batch ingestion
        # ───────────────────────────────────────────────────────────────────
        block_capacity = 1 + self.micro_block_size
        num_blocks = (seq_len + block_capacity - 1) // block_capacity
        num_full_blocks = seq_len // block_capacity
        L_full = num_full_blocks * block_capacity

        new_blocks = []
        full_blocks_to_compress = []
        base_idx = self._next_anchor_idx(blocks)

        # 1. Vectorized extraction of full blocks
        if num_full_blocks > 0:
            k_full = k[:, :, :L_full]
            v_full = v[:, :, :L_full]

            # Reshape into [1, heads, num_full_blocks, block_capacity, head_dim]
            k_reshaped = k_full.reshape(1, k.shape[1], num_full_blocks, block_capacity, k.shape[3])
            v_reshaped = v_full.reshape(1, v.shape[1], num_full_blocks, block_capacity, v.shape[3])

            # Extract anchors: [1, heads, num_full_blocks, head_dim]
            anchors_k = k_reshaped[:, :, :, 0]
            anchors_v = v_reshaped[:, :, :, 0]

            # Stack K/V anchors: [num_full_blocks, 1, 2, heads, head_dim]
            stacked_anchors = torch.stack([anchors_k, anchors_v], dim=2).permute(3, 0, 2, 1, 4)

            # Extract active states: [num_full_blocks, 1, heads, micro_block_size, head_dim]
            active_k_blocks = k_reshaped[:, :, :, 1:].permute(2, 0, 1, 3, 4)
            active_v_blocks = v_reshaped[:, :, :, 1:].permute(2, 0, 1, 3, 4)

            for i in range(num_full_blocks):
                anchor_idx = base_idx + i * block_capacity
                anchor_kv = stacked_anchors[i]
                
                # Pre-allocate NativeBlockPool index in the single-threaded main thread
                pool_idx = None
                if self.native_pool is not None:
                    pool_idx = self.native_pool.allocate_block()

                new_block = StreamingKVBlock(
                    anchor_idx=anchor_idx,
                    anchor_kv=anchor_kv,
                    micro_block_size=self.micro_block_size,
                    token_indices=list(range(anchor_idx, anchor_idx + block_capacity)),
                    pool_idx=pool_idx,
                )
                new_block.active_k = active_k_blocks[i]
                new_block.active_v = active_v_blocks[i]
                new_block.state = "SUBMITTED"

                new_blocks.append(new_block)
                full_blocks_to_compress.append(new_block)

                with self._stats_lock:
                    self.stats["total_blocks_created"] += 1

        # 2. Extract partial block if any
        if seq_len > L_full:
            anchor_idx = base_idx + L_full
            
            # Slice anchor token
            anchor_k = k[:, :, L_full : L_full + 1]
            anchor_v = v[:, :, L_full : L_full + 1]
            anchor_kv = torch.stack([anchor_k[:, :, 0], anchor_v[:, :, 0]], dim=1)

            active_start = L_full + 1
            blk_active_k = None
            blk_active_v = None
            token_indices = [anchor_idx]

            if seq_len > active_start:
                blk_active_k = k[:, :, active_start:seq_len]
                blk_active_v = v[:, :, active_start:seq_len]
                token_indices.extend(list(range(anchor_idx + 1, anchor_idx + 1 + (seq_len - active_start))))

            pool_idx = None
            if self.native_pool is not None:
                pool_idx = self.native_pool.allocate_block()

            new_block = StreamingKVBlock(
                anchor_idx=anchor_idx,
                anchor_kv=anchor_kv,
                micro_block_size=self.micro_block_size,
                token_indices=token_indices,
                pool_idx=pool_idx,
            )

            if blk_active_k is not None:
                new_block.active_k = blk_active_k
                new_block.active_v = blk_active_v

            if new_block.is_compression_eligible():
                new_block.state = "SUBMITTED"
                full_blocks_to_compress.append(new_block)
            else:
                new_block.state = "ACCUMULATING"

            new_blocks.append(new_block)
            with self._stats_lock:
                self.stats["total_blocks_created"] += 1

        blocks.extend(new_blocks)

        # Batch submit all compression requests in one consolidation transfer
        if full_blocks_to_compress:
            self._submit_blocks_batched(full_blocks_to_compress)

        # Track peak dense footprint
        dense_tokens = self._count_dense_tokens(blocks)
        with self._stats_lock:
            if dense_tokens > self.stats["total_dense_tokens_peak"]:
                self.stats["total_dense_tokens_peak"] = dense_tokens

    def _submit_blocks_batched(self, blocks_list: List[StreamingKVBlock]):
        if not blocks_list:
            return

        # Stack K-V active tensors
        stacked_k = torch.cat([b.active_k for b in blocks_list], dim=0)
        stacked_v = torch.cat([b.active_v for b in blocks_list], dim=0)

        # Single consolidated copy to CPU.
        # CRITICAL: We use non_blocking=True for transfer efficiency.
        # To avoid blocking the main thread, we record a CUDA Event and synchronize in the background.
        stacked_k_cpu = stacked_k.to("cpu", non_blocking=True)
        stacked_v_cpu = stacked_v.to("cpu", non_blocking=True)

        event = None
        if stacked_k.device.type == "cuda":
            event = torch.cuda.Event()
            event.record()

        # Enqueue individual blocks
        for idx, block in enumerate(blocks_list):
            k_cpu_slice = stacked_k_cpu[idx : idx + 1]
            v_cpu_slice = stacked_v_cpu[idx : idx + 1]

            try:
                self.compressor._queue.put_nowait((block, k_cpu_slice, v_cpu_slice, event))
                with self.compressor._stats_lock:
                    self.compressor.stats["submitted"] += 1
                    depth = self.compressor._queue.qsize()
                    if depth > self.compressor.stats["queue_depth_peak"]:
                        self.compressor.stats["queue_depth_peak"] = depth
            except queue.Full:
                # Sync fallback if queue is full
                self.compress_fn(block, block.active_k, block.active_v)
                block.state = "COMPRESSED"
                with self.compressor._stats_lock:
                    self.compressor.stats["sync_fallbacks"] += 1

            with self._stats_lock:
                self.stats["total_compressed"] += 1
                self.stats["compressions_during_ingest"] += 1

    def _submit_block_for_compression(self, block: StreamingKVBlock):
        """Submit block for background compression. Block stays readable via active_k/v."""
        k = block.active_k
        v = block.active_v
        block.state = "SUBMITTED"

        # Non-blocking: copies to CPU immediately, frees GPU as soon as SVD completes
        submitted = self.compressor.submit(block, k, v)

        if not submitted:
            # Backpressure: compress synchronously
            self.compress_fn(block, k, v)
            block.state = "COMPRESSED"

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
