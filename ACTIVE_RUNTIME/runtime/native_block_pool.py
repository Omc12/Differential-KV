"""
runtime/native_block_pool.py

Phase 10: Native Block Pool (vLLM style block tables)

Pre-allocates large contiguous GPU/MPS memory pools for all sparse block components
(U, V_K, V_V, anchors, scales, seq_lens). When blocks are compressed, their data
is copied into an assigned slot in this pool.

During inference, we completely bypass `torch.stack`. We simply pass a 1D tensor
of `block_indices` to the Triton kernel, which does the gather natively in SRAM.

Mac/MPS: all `torch.cuda.*` calls are routed through native_core.mac_utils.

Phase Optimization: max_seq_len is now passed as the actual micro_block_size
(16-32 tokens) rather than a static 256, reducing U-tensor VRAM by 8-16x per block.
MPS gets a smaller initial footprint (128 blocks) and finer growth increments (128).
Pre-realloc gc.collect() prevents momentary 2x VRAM spike during pool growth.
"""

import torch

# SRL descriptor dimension — must match native_core/srl/chunk_descriptor.py
_SRL_DESC_DIM = 64
import gc
try:
    from native_core.mac_utils import empty_cache as _empty_cache
except ImportError:
    def _empty_cache(device=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class NativeBlockPool:
    def __init__(
        self,
        max_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        rank: int,
        max_seq_len: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        initial_blocks: int = 512,
        num_layers: int = 28,
        lazy: bool = False,
    ):
        # ── Phase 1: Record config — NO GPU tensors allocated yet if lazy ────────────
        # Allocation is deferred to ensure_allocated(n_tokens), called by
        # KVRuntimeManager.create_session() once the actual context length
        # is known. This prevents the pool from consuming 185 MB at startup
        # for a 512-token session where there is nothing to compress.
        self.max_blocks      = max_blocks
        self.num_kv_heads    = num_kv_heads
        self.head_dim        = head_dim
        self.rank            = rank
        self.max_seq_len     = max_seq_len
        self.device          = device
        self.dtype           = dtype
        self.initial_blocks  = initial_blocks
        self.num_layers      = num_layers

        _is_mps = (str(device) == "mps" or
                   (isinstance(device, torch.device) and device.type == "mps"))
        self._grow_increment = 128 if _is_mps else 512
        self._is_mps         = _is_mps

        # Bytes per block — used for n_blocks computation in ensure_allocated
        self._bytes_per_block = (
            max_seq_len * rank * 1 +              # U  (int8)
            rank * num_kv_heads * head_dim * 2 * 2 +  # V_K + V_V (fp16)
            num_kv_heads * head_dim * 2 * 2 +     # anchors K + V (fp16)
            6 + 2                                 # scales (2B) + seq_lens (4B) + U_scale (2B)
        )

        # Default token hint used as fallback when ensure_allocated is called
        # without an explicit context length (e.g. from _grow_pool before first session).
        _startup_target_bytes = 64 * 1024 * 1024 if _is_mps else 256 * 1024 * 1024
        self._default_token_hint = _startup_target_bytes // max(self._bytes_per_block, 1) * max_seq_len

        # Allocation state — nothing on GPU until ensure_allocated() is called
        self._allocated      = False
        self.current_blocks  = 0

        # Allocator state (populated by ensure_allocated or eager allocation)
        self._free_indices     = []
        self._free_indices_set = set()
        self._ref_counts       = []
        self._last_used        = []
        self.version           = []

        # Random projection matrix — set by KVRuntimeManager after construction
        self.W_proj: torch.Tensor = None  # type: ignore[assignment]

        self.lazy = lazy
        if not lazy:
            self._allocate_tensors(initial_blocks)
            self._allocated = True

    # ── Phase 2: Actual GPU allocation ───────────────────────────────────────
    def ensure_allocated(self, n_tokens: int = None) -> None:
        """
        Allocate pool tensors sized to *n_tokens* of context.

        Called by KVRuntimeManager.create_session() with the actual prefill
        length so the pool is sized to what the session will actually need,
        not a fixed worst-case. Safe to call multiple times — no-op after
        first allocation (use _grow_pool to expand).

        n_tokens: estimated total tokens this session will produce.
                  None → use the default startup hint.
        """
        if self._allocated:
            return  # Already allocated — nothing to do

        if n_tokens is None or n_tokens <= 0:
            n_tokens = self._default_token_hint

        # n_blocks = blocks needed across all layers with 1.5x headroom, clamped to [64, max_blocks]
        n_raw = max(1, int(n_tokens / self.max_seq_len) * self.num_layers)
        n_desired = int(n_raw * 1.5)
        
        # CRITICAL FIX: Respect the max_blocks budget that was set based on MPS memory constraints
        # Start with a smaller initial allocation and let it grow on demand via _grow_pool
        # Initial allocation: min(n_desired, max_blocks // 2, 512)
        # This prevents upfront allocation of the full budget when a large prompt comes in
        n_blocks = max(64, min(n_desired, self.max_blocks // 2, 512))

        self._allocate_tensors(n_blocks)
        self._allocated = True
        print(f"[Pool] Lazy-allocated {n_blocks} slots for ~{n_tokens} tokens "
              f"= {self._pool_mb():.1f} MB (device={self.device})")

    def _allocate_tensors(self, n_blocks: int) -> None:
        """Allocate (or re-allocate) all pool tensors at *n_blocks* size."""
        self.current_blocks = n_blocks
        self.U          = torch.zeros((n_blocks, self.max_seq_len, self.rank), device=self.device, dtype=torch.int8)
        self.U_scale    = torch.zeros((n_blocks,), device=self.device, dtype=self.dtype)
        self.V_KV       = torch.zeros((n_blocks, 2, self.rank, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
        self.anchors_KV = torch.zeros((n_blocks, 2, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
        self.scales     = torch.zeros((n_blocks,), device=self.device, dtype=self.dtype)
        self.seq_lens   = torch.zeros((n_blocks,), device=self.device, dtype=torch.int32)
        self.desc       = torch.zeros((n_blocks, _SRL_DESC_DIM), device=self.device, dtype=torch.float16)

        self._free_indices     = list(range(n_blocks - 1, -1, -1))
        self._free_indices_set = set(self._free_indices)
        self._ref_counts       = [0] * n_blocks
        self._last_used        = [0.0] * n_blocks
        self.version           = [0] * n_blocks

        # Re-attach W_proj at the new size if it was already set
        if self.W_proj is not None and self.W_proj.device != torch.device("cpu"):
            pass  # W_proj is a [DESC_DIM, head_dim] matrix — shape is independent of n_blocks

    def _pool_mb(self) -> float:
        """Current pool VRAM usage in megabytes."""
        total = 0
        for attr in ("U", "U_scale", "V_KV", "anchors_KV", "scales", "seq_lens", "desc"):
            t = getattr(self, attr, None)
            if t is not None:
                total += t.numel() * t.element_size()
        return total / 1024 ** 2

    def _ensure(self) -> None:
        """Guard used by all access methods to trigger lazy allocation if needed."""
        if not self._allocated:
            self.ensure_allocated()



    def _grow_pool(self, new_blocks: int = None):
        self._ensure()  # Trigger lazy allocation if pool not yet created
        old_blocks = self.current_blocks
        if old_blocks >= self.max_blocks:
            raise RuntimeError(f"NativeBlockPool is out of memory and has reached its absolute maximum limit of {self.max_blocks} blocks!")
        
        if new_blocks is None:
            new_blocks = min(self.max_blocks, old_blocks + self._grow_increment)
        else:
            new_blocks = min(self.max_blocks, max(new_blocks, old_blocks + self._grow_increment))
            
        added = new_blocks - old_blocks
        if added <= 0:
            return
            
        num_kv_heads = self.num_kv_heads
        head_dim     = self.head_dim
        rank         = self.rank
        max_seq_len  = self.max_seq_len

        # ── Release old memory BEFORE allocating new tensors ──────────────
        # Running gc+empty_cache here helps the GPU allocator reclaim the
        # old pool pages before the new (larger) tensors are created, cutting
        # the momentary peak from ~2x down to ~1.x of the new pool size.
        gc.collect()
        _empty_cache(self.device)
        
        new_U = torch.zeros((new_blocks, max_seq_len, rank), device=self.device, dtype=torch.int8)
        new_U_scale = torch.zeros((new_blocks,), device=self.device, dtype=self.dtype)
        new_V_KV = torch.zeros((new_blocks, 2, rank, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_anchors_KV = torch.zeros((new_blocks, 2, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_scales = torch.zeros((new_blocks,), device=self.device, dtype=self.dtype)
        new_seq_lens = torch.zeros((new_blocks,), device=self.device, dtype=torch.int32)
        new_desc = torch.zeros((new_blocks, _SRL_DESC_DIM), device=self.device, dtype=torch.float16)

        new_U[:old_blocks] = self.U
        new_U_scale[:old_blocks] = self.U_scale
        new_V_KV[:old_blocks] = self.V_KV
        new_anchors_KV[:old_blocks] = self.anchors_KV
        new_scales[:old_blocks] = self.scales
        new_seq_lens[:old_blocks] = self.seq_lens
        new_desc[:old_blocks] = self.desc

        # Explicitly delete old tensors so the allocator can reclaim them
        del self.U, self.U_scale, self.V_KV, self.anchors_KV, self.scales, self.seq_lens, self.desc

        self.U = new_U
        self.U_scale = new_U_scale
        self.V_KV = new_V_KV
        self.anchors_KV = new_anchors_KV
        self.scales = new_scales
        self.seq_lens = new_seq_lens
        self.desc = new_desc
        
        self._ref_counts.extend([0] * added)
        self._last_used.extend([0.0] * added)
        self.version.extend([0] * added)
        added_range = range(new_blocks - 1, old_blocks - 1, -1)
        self._free_indices.extend(added_range)
        self._free_indices_set.update(added_range)
        self.current_blocks = new_blocks
        
        gc.collect()
        _empty_cache(self.device)
        
    def allocate_block(self) -> int:
        import time as _time
        self._ensure()  # Trigger lazy allocation if pool not yet created
        if not self._free_indices:
            self._grow_pool()
            if not self._free_indices:
                raise RuntimeError("NativeBlockPool is completely full and no free blocks are available!")
        
        idx = self._free_indices.pop()
        self._free_indices_set.discard(idx)
        self._ref_counts[idx] = 1
        self._last_used[idx] = _time.time()
        return idx

    def allocate_blocks(self, count: int) -> list:
        allocated = []
        for _ in range(count):
            allocated.append(self.allocate_block())
        return allocated
        
    def increment_ref(self, pool_idx: int):
        if pool_idx is not None and 0 <= pool_idx < self.current_blocks:
            self._ref_counts[pool_idx] += 1
            if pool_idx in self._free_indices_set:
                self._free_indices_set.discard(pool_idx)
                try:
                    self._free_indices.remove(pool_idx)
                except ValueError:
                    pass


    def free_block(self, pool_idx: int):
        import time as _time
        if pool_idx is not None and 0 <= pool_idx < self.current_blocks:
            self._ref_counts[pool_idx] -= 1
            if self._ref_counts[pool_idx] <= 0:
                self._ref_counts[pool_idx] = 0
                self._last_used[pool_idx] = _time.time() # Mark freed time as last used
                if pool_idx not in self._free_indices_set:
                    self._free_indices.append(pool_idx)
                    self._free_indices_set.add(pool_idx)

    def touch_block(self, pool_idx: int):
        import time as _time
        if pool_idx is not None and 0 <= pool_idx < self.current_blocks:
            self._last_used[pool_idx] = _time.time()
        
    def write_block(
        self, 
        pool_idx: int, 
        U: torch.Tensor, 
        V: torch.Tensor, 
        anchor_K: torch.Tensor, 
        anchor_V: torch.Tensor, 
        scale: float, 
        seq_len: int
    ):
        """
        Copies compressed data directly into the contiguous pool.
        This happens in the background (AsyncCompressor) or once per block,
        NEVER during the decode hot-path.

        U may be unpadded shape (seq_len, dynamic_rank) — we write only as
        many rank columns as U actually has, leaving trailing zeros intact.

        V is a joint [dynamic_rank, 2 * num_kv_heads * head_dim] tensor.
        The first half of columns is V_K; the second half is V_V.
        A ValueError is raised (rather than silently corrupted) if V's rank
        exceeds the pool's allocated rank dimension.
        """
        self._ensure()  # Trigger lazy allocation if pool not yet created
        import math as _math
        
        # Sanitize inputs to prevent NaN/Inf propagation
        if not torch.isfinite(U).all():
            U = torch.nan_to_num(U, nan=0.0, posinf=65504.0, neginf=-65504.0)
        if not torch.isfinite(V).all():
            V = torch.nan_to_num(V, nan=0.0, posinf=65504.0, neginf=-65504.0)
        if not torch.isfinite(anchor_K).all():
            anchor_K = torch.nan_to_num(anchor_K, nan=0.0, posinf=65504.0, neginf=-65504.0)
        if not torch.isfinite(anchor_V).all():
            anchor_V = torch.nan_to_num(anchor_V, nan=0.0, posinf=65504.0, neginf=-65504.0)
        if not _math.isfinite(scale):
            scale = 1.0

        pool_max_seq = self.U.shape[1]
        pool_rank    = self.U.shape[2]
        num_kv       = self.V_KV.shape[3]
        h_dim        = self.V_KV.shape[4]
        
        write_seq    = min(seq_len, pool_max_seq)
        write_rank   = min(U.shape[1], pool_rank)
        
        if V.shape[0] > pool_rank:
            raise ValueError(f"V rank {V.shape[0]} exceeds pool rank capacity {pool_rank}")

        self.U[pool_idx] = 0
        self.V_KV[pool_idx] = 0
        
        # Quantize U to int8
        U_sliced = U[:write_seq, :write_rank].float()
        max_abs = U_sliced.abs().max()
        scale_u = torch.clamp(max_abs / 127.0, min=1e-5).to(self.dtype)
        self.U[pool_idx, :write_seq, :write_rank] = torch.clamp(torch.round(U_sliced / scale_u), -127, 127).to(torch.int8)
        self.U_scale[pool_idx] = scale_u
        
        # Split V_K and V_V
        vk = V[:write_rank, :num_kv * h_dim].view(write_rank, num_kv, h_dim)
        vv = V[:write_rank, num_kv * h_dim:].view(write_rank, num_kv, h_dim)
        
        self.V_KV[pool_idx, 0, :write_rank] = vk.to(self.dtype)
        self.V_KV[pool_idx, 1, :write_rank] = vv.to(self.dtype)
        self.anchors_KV[pool_idx, 0] = anchor_K.to(self.dtype)
        self.anchors_KV[pool_idx, 1] = anchor_V.to(self.dtype)
        self.scales[pool_idx] = scale
        self.seq_lens[pool_idx] = seq_len
        self.version[pool_idx] += 1

        # ── SRL: compute and store semantic descriptor ─────────────────────
        # Runs only when W_proj is initialized (set by KVRuntimeManager).
        # Cost: ~3R+2D multiplications — negligible vs. SVD compression cost.
        if self.W_proj is not None:
            try:
                from native_core.srl.chunk_descriptor import compute_descriptor
                self.desc[pool_idx] = compute_descriptor(
                    anchor_K = self.anchors_KV[pool_idx, 0],              # [kv_heads, D] fp16
                    U_int8   = self.U[pool_idx, :write_seq, :write_rank], # [S, R] int8
                    U_scale  = self.U_scale[pool_idx],                    # scalar fp16
                    V_K      = self.V_KV[pool_idx, 0, :write_rank],       # [R, kv_heads, D] fp16
                    W_proj   = self.W_proj,                               # [DESC_DIM, D] fp32
                )
            except Exception:
                pass  # Descriptor failure is non-fatal — SRL routing degrades gracefully

    def reset(self):
        """Completely reset the pool to its initial lightweight state, releasing all grown VRAM."""
        # Free old tensors before re-allocating
        for attr in ("U", "U_scale", "V_KV", "anchors_KV", "scales", "seq_lens", "desc"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        _empty_cache(self.device)

        self._allocated = False
        self.current_blocks = 0
        self._free_indices     = []
        self._free_indices_set = set()
        self._ref_counts       = []
        self._last_used        = []
        self.version           = []

        if not self.lazy:
            self._allocate_tensors(self.initial_blocks)
            self._allocated = True

        gc.collect()
        _empty_cache(self.device)


    # Contiguous property views for backward-compatibility with callers/kernels
    @property
    def V_K(self):
        return self.V_KV[:, 0]

    @property
    def V_V(self):
        return self.V_KV[:, 1]

    @property
    def anchors_K(self):
        return self.anchors_KV[:, 0]

    @property
    def anchors_V(self):
        return self.anchors_KV[:, 1]
