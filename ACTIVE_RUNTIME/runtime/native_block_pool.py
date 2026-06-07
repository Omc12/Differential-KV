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
    ):
        # ── Compute a lean startup footprint ─────────────────────────────────
        # Actual bytes per block in the pool.
        bytes_per_block = (
            max_seq_len * rank * 1 +           # U  (int8)
            rank * num_kv_heads * head_dim * 2 * 2 +  # V_K + V_V (fp16)
            num_kv_heads * head_dim * 2 * 2 +  # anchors K + V (fp16)
            6 + 2                              # scales (2B) + seq_lens (4B) + U_scale (2B)
        )

        # Base rank schedule and config setup
        _is_mps = (str(device) == "mps" or
                   (isinstance(device, torch.device) and device.type == "mps"))
        _startup_target_bytes = 64 * 1024 * 1024 if _is_mps else 256 * 1024 * 1024
        # Smaller default initial_blocks on MPS
        if _is_mps and initial_blocks > 128:
            initial_blocks = 128

        # Target startup footprint, but never more than max_blocks and
        # never less than initial_blocks. The pool grows lazily via _grow_pool().
        # For unit tests (where max_blocks is very small), we honor initial_blocks exactly.
        if max_blocks <= 64:
            computed_initial = initial_blocks
        else:
            computed_initial = max(
                initial_blocks,
                min(max_blocks, _startup_target_bytes // max(bytes_per_block, 1))
            )

        self.max_blocks     = max_blocks
        self.initial_blocks = computed_initial
        self.current_blocks = self.initial_blocks
        self.device = device
        self.dtype  = dtype
        # Growth increment: coarser on CUDA (512), finer on MPS (128) to
        # keep peak spikes small on unified-memory devices.
        self._grow_increment = 128 if _is_mps else 512

        # Allocate pools (fused contiguous allocation layout)
        self.U          = torch.zeros((self.current_blocks, max_seq_len, rank), device=device, dtype=torch.int8)
        self.U_scale    = torch.zeros((self.current_blocks,), device=device, dtype=self.dtype)
        self.V_KV       = torch.zeros((self.current_blocks, 2, rank, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.anchors_KV = torch.zeros((self.current_blocks, 2, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.scales     = torch.zeros((self.current_blocks,), device=device, dtype=self.dtype)
        self.seq_lens   = torch.zeros((self.current_blocks,), device=device, dtype=torch.int32)

        # ── SRL descriptor tensor ─────────────────────────────────────────────
        # desc[i] is a 64-dim semantic fingerprint for pool slot i.
        # Written by write_block() after each SVD compression.
        # Used by SemanticIndex for ANN search during decode routing.
        self.desc = torch.zeros(
            (self.current_blocks, _SRL_DESC_DIM), device=device, dtype=torch.float16
        )

        # Random projection matrix [DESC_DIM, head_dim] — set by KVRuntimeManager
        # after construction (needs head_dim which is known at pool init).
        # Initialized here as None; KVRuntimeManager sets it immediately after.
        self.W_proj: torch.Tensor = None  # type: ignore[assignment]

        # Block allocator state
        self._free_indices = list(range(self.current_blocks - 1, -1, -1))
        self._free_indices_set = set(self._free_indices)
        self._ref_counts = [0] * self.current_blocks
        self._last_used = [0.0] * self.current_blocks
        self.version = [0] * self.current_blocks

    def _grow_pool(self, new_blocks: int = None):
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
            
        num_kv_heads = self.V_KV.shape[3]
        head_dim = self.V_KV.shape[4]
        rank = self.V_KV.shape[2]
        max_seq_len = self.U.shape[1]
        
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
        self.current_blocks = self.initial_blocks
        self._free_indices = list(range(self.current_blocks - 1, -1, -1))
        self._free_indices_set = set(self._free_indices)
        self._ref_counts = [0] * self.current_blocks
        self._last_used = [0.0] * self.current_blocks
        self.version = [0] * self.current_blocks
        
        num_kv_heads = self.V_KV.shape[3]
        head_dim = self.V_KV.shape[4]
        rank = self.V_KV.shape[2]
        max_seq_len = self.U.shape[1]
        
        del self.U, self.U_scale, self.V_KV, self.anchors_KV, self.scales, self.seq_lens, self.desc
        gc.collect()
        _empty_cache(self.device)

        self.U          = torch.zeros((self.current_blocks, max_seq_len, rank), device=self.device, dtype=torch.int8)
        self.U_scale    = torch.zeros((self.current_blocks,), device=self.device, dtype=self.dtype)
        self.V_KV       = torch.zeros((self.current_blocks, 2, rank, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        self.anchors_KV = torch.zeros((self.current_blocks, 2, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        self.scales     = torch.zeros((self.current_blocks,), device=self.device, dtype=self.dtype)
        self.seq_lens   = torch.zeros((self.current_blocks,), device=self.device, dtype=torch.int32)
        self.desc       = torch.zeros((self.current_blocks, _SRL_DESC_DIM), device=self.device, dtype=torch.float16)

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
