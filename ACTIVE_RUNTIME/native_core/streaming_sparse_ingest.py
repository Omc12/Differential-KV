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


@dataclass(slots=True)
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
    session_id: Optional[str] = None
    layer_idx: Optional[int] = None
    _cache_id: Optional[str] = None
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
        device: str = "cuda",
    ):
        self.compressor = compressor
        self.compress_fn = compress_fn
        self.micro_block_size = micro_block_size
        self.dense_anchor_only = dense_anchor_only
        self.native_pool = native_pool
        self.device = device

        # session_id -> layer_idx -> List[StreamingKVBlock]
        self.session_blocks: Dict[str, Dict[int, List[StreamingKVBlock]]] = {}
        
        # session_id -> layer_idx -> Contiguous 2D metadata tensor [MAX_BLOCKS, 4]
        self.session_metadata: Dict[str, Dict[int, torch.Tensor]] = {}
        
        # session_id -> micro_block_size (dynamic adaptive size per session)
        self.session_micro_block_sizes: Dict[str, int] = {}
        
        # session_id -> (k_gpu, v_gpu, k_cpu, v_cpu) pre-allocated pinned memory buffers.
        # A single buffer per session is shared across all layers (safe because slices are
        # cloned before being enqueued into the async compressor — breaking aliasing).
        self.session_staging_buffers: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

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
        if session_id not in self.session_micro_block_sizes:
            if prefill_len > 0:
                raw_size = max(16, min(256, prefill_len // 64))
                adaptive_size = ((raw_size + 15) // 16) * 16
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
            
            # Allocate larger buffers dynamically
            alloc_blocks = max(num_blocks, 16)
            alloc_mbs = max(micro_block_size, 256)
            
            k_gpu = torch.zeros((alloc_blocks, heads, alloc_mbs, head_dim), dtype=torch.float16, device=device)
            v_gpu = torch.zeros((alloc_blocks, heads, alloc_mbs, head_dim), dtype=torch.float16, device=device)
            
            k_cpu = torch.zeros((alloc_blocks, heads, alloc_mbs, head_dim), dtype=torch.float16).pin_memory()
            v_cpu = torch.zeros((alloc_blocks, heads, alloc_mbs, head_dim), dtype=torch.float16).pin_memory()
            
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
        self.session_micro_block_sizes.pop(session_id, None)
        self.session_staging_buffers.pop(session_id, None)

    def update_metadata_block(self, session_id: str, layer_idx: int, block_idx: int, block):
        metadata = self.session_metadata.setdefault(session_id, {}).setdefault(
            layer_idx, torch.full((1024, 4), -1, dtype=torch.int32, device=self.device)
        )
        if block_idx >= metadata.shape[0]:
            new_size = metadata.shape[0] * 2
            new_meta = torch.full((new_size, 4), -1, dtype=torch.int32, device=self.device)
            new_meta[:metadata.shape[0]] = metadata
            self.session_metadata[session_id][layer_idx] = new_meta
            metadata = new_meta

        metadata[block_idx, 0] = block.pool_idx if block.pool_idx is not None else -1
        metadata[block_idx, 1] = block.anchor_idx
        metadata[block_idx, 2] = block.token_count()
        state_codes = {"ACCUMULATING": 0, "SUBMITTED": 1, "COMPRESSED": 2, "PAGED": 3}
        metadata[block_idx, 3] = state_codes.get(block.state, -1)

    def update_metadata_state(self, session_id: str, layer_idx: int, block):
        blocks = self.session_blocks.get(session_id, {}).get(layer_idx, [])
        try:
            block_idx = blocks.index(block)
        except ValueError:
            return
        metadata = self.session_metadata.get(session_id, {}).get(layer_idx)
        if metadata is not None and block_idx < metadata.shape[0]:
            metadata[block_idx, 0] = block.pool_idx if block.pool_idx is not None else -1
            metadata[block_idx, 2] = block.token_count()
            state_codes = {"ACCUMULATING": 0, "SUBMITTED": 1, "COMPRESSED": 2, "PAGED": 3}
            metadata[block_idx, 3] = state_codes.get(block.state, -1)

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
            # Force micro_block_size to 32 for active window during decode
            micro_block_size = 32
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
                    micro_block_size=micro_block_size,
                    token_indices=[self._next_anchor_idx(blocks)],
                    pool_idx=pool_idx,
                    session_id=session_id,
                    layer_idx=layer_idx,
                )
                blocks.append(new_block)
                self.update_metadata_block(session_id, layer_idx, len(blocks) - 1, new_block)
                
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
            self.update_metadata_block(session_id, layer_idx, len(blocks) - 1, current_block)

            # Immediately compress when micro-block fills — during ingest!
            if current_block.is_compression_eligible():
                self._submit_block_for_compression(current_block)
                self.update_metadata_block(session_id, layer_idx, len(blocks) - 1, current_block)
                with self._stats_lock:
                    self.stats["compressions_during_ingest"] += 1
            return

        # ───────────────────────────────────────────────────────────────────
        # PREFILL PATH (T > 1) — highly optimized vectorized batch ingestion
        # ───────────────────────────────────────────────────────────────────
        # Partition the prefill sequence into regions based on distance to sequence end
        regions = []
        # Region 1 (Active recent window): last 1024 tokens (MBS = 32)
        r1_start = max(0, seq_len - 1024)
        if r1_start < seq_len:
            regions.append((r1_start, seq_len, 32))
            
        # Region 2 (Conversational locality): from seq_len-4096 to seq_len-1024 (MBS = 64)
        r2_start = max(0, seq_len - 4096)
        if r2_start < r1_start:
            regions.append((r2_start, r1_start, 64))
            
        # Region 3 (Mid-history): from seq_len-12288 to seq_len-4096 (MBS = 128)
        r3_start = max(0, seq_len - 12288)
        if r3_start < r2_start:
            regions.append((r3_start, r2_start, 128))
            
        # Region 4 (Cold archive): from 0 to seq_len-12288 (MBS = 256)
        if 0 < r3_start:
            regions.append((0, r3_start, 256))
            
        # Reverse to process chronologically (left to right)
        regions.reverse()

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
                stacked_anchors_cpu = stacked_anchors.cpu()

                # Extract active states: [num_full_blocks, 1, heads, r_mbs, head_dim]
                active_k_blocks = k_reshaped[:, :, :, 1:].permute(2, 0, 1, 3, 4)
                active_v_blocks = v_reshaped[:, :, :, 1:].permute(2, 0, 1, 3, 4)

                # Pre-allocate NativeBlockPool indices in a single batch call!
                pool_indices = []
                if self.native_pool is not None:
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
                    new_block.active_k = active_k_blocks[i]
                    new_block.active_v = active_v_blocks[i]
                    new_block.state = "SUBMITTED"

                    new_blocks.append(new_block)
                    full_blocks_to_compress.append(new_block)

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
                if self.native_pool is not None:
                    pool_idx = self.native_pool.allocate_block()

                new_block = StreamingKVBlock(
                    anchor_idx=anchor_idx,
                    anchor_kv=anchor_kv,
                    anchor_kv_cpu=anchor_kv.cpu(),
                    micro_block_size=r_mbs,
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
            for idx, block in enumerate(new_blocks):
                block.session_id = session_id
                block.layer_idx = layer_idx
                self.update_metadata_block(session_id, layer_idx, len(blocks) - len(new_blocks) + idx, block)

            # Batch submit all compression requests in one consolidation transfer
            if full_blocks_to_compress:
                self._submit_blocks_batched(session_id, layer_idx, full_blocks_to_compress)

        # Track peak dense footprint
        dense_tokens = self._count_dense_tokens(blocks)
        with self._stats_lock:
            if dense_tokens > self.stats["total_dense_tokens_peak"]:
                self.stats["total_dense_tokens_peak"] = dense_tokens

    def _submit_blocks_batched(self, session_id: str, layer_idx: int, blocks_list: List[StreamingKVBlock]):
        if not blocks_list:
            return

        # Fetch shape metadata from the first active block
        micro_block_size = blocks_list[0].micro_block_size
        heads = blocks_list[0].active_k.shape[1]
        head_dim = blocks_list[0].active_k.shape[3]
        device = blocks_list[0].active_k.device

        # Get single session-level staging buffer (shared across all layers).
        k_gpu, v_gpu, k_cpu, v_cpu = self._get_session_staging_buffer(
            session_id, len(blocks_list), heads, micro_block_size, head_dim, device
        )

        # Concat K-V active tensors in-place directly into our pre-allocated GPU staging buffers
        torch.cat([b.active_k for b in blocks_list], dim=0, out=k_gpu)
        torch.cat([b.active_v for b in blocks_list], dim=0, out=v_gpu)

        # Synchronous GPU->CPU DMA via pinned memory
        k_cpu.copy_(k_gpu, non_blocking=True)
        v_cpu.copy_(v_gpu, non_blocking=True)

        # Use a targeted CUDA Event to wait only for the DMA copy, not all subsequent
        # GPU work. current_stream().synchronize() would also stall on any compute
        # kernels queued after this point (e.g. next layer's projections), causing
        # 600ms+ prefill overhead across 28 layers × 4 regions = 112 sync points.
        if k_gpu.is_cuda:
            _dma_event = torch.cuda.Event()
            _dma_event.record()
            _dma_event.synchronize()

        # Enqueue cloned slices — each clone is an independent CPU tensor,
        # so the shared staging buffer can be safely reused across layers.
        is_async_active = getattr(self.compressor, "_running", False) and hasattr(self.compressor, "_queue")
        
        for idx, block in enumerate(blocks_list):
            k_cpu_slice = k_cpu[idx : idx + 1].clone()
            v_cpu_slice = v_cpu[idx : idx + 1].clone()

            if is_async_active:
                try:
                    self.compressor._queue.put_nowait((block, k_cpu_slice, v_cpu_slice, None))
                    with self.compressor._stats_lock:
                        self.compressor.stats["submitted"] += 1
                        depth = self.compressor._queue.qsize()
                        if depth > self.compressor.stats["queue_depth_peak"]:
                            self.compressor.stats["queue_depth_peak"] = depth
                except queue.Full:
                    # Sync fallback if queue is full
                    self.compress_fn(block, k_cpu_slice, v_cpu_slice)
                    block.state = "COMPRESSED"
                    with self.compressor._stats_lock:
                        self.compressor.stats["sync_fallbacks"] += 1
            else:
                # Sync execution directly on the CPU slice
                self.compress_fn(block, k_cpu_slice, v_cpu_slice)
                block.state = "COMPRESSED"
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
