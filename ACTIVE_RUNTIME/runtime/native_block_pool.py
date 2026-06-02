"""
runtime/native_block_pool.py

Phase 10: Native Block Pool (vLLM style block tables)

Pre-allocates large contiguous GPU/MPS memory pools for all sparse block components
(U, V_K, V_V, anchors, scales, seq_lens). When blocks are compressed, their data
is copied into an assigned slot in this pool.

During inference, we completely bypass `torch.stack`. We simply pass a 1D tensor
of `block_indices` to the Triton kernel, which does the gather natively in SRAM.

Mac/MPS: all `torch.cuda.*` calls are routed through native_core.mac_utils.
"""

import torch
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
            max_seq_len * rank * 2 +           # U  (fp16)
            rank * num_kv_heads * head_dim * 2 * 2 +  # V_K + V_V (fp16)
            num_kv_heads * head_dim * 2 * 2 +  # anchors K + V (fp16)
            6                                  # scales (2B) + seq_lens (4B)
        )

        # Target startup footprint: 256 MB, but never more than max_blocks and
        # never less than initial_blocks.  The pool grows lazily via _grow_pool().
        # For unit tests (where max_blocks is very small), we honor initial_blocks exactly.
        if max_blocks <= 64:
            computed_initial = initial_blocks
        else:
            target_startup_bytes = 256 * 1024 * 1024   # 256 MB
            computed_initial = max(
                initial_blocks,
                min(max_blocks, target_startup_bytes // max(bytes_per_block, 1))
            )

        self.max_blocks     = max_blocks
        self.initial_blocks = computed_initial
        self.current_blocks = self.initial_blocks
        self.device = device
        self.dtype  = dtype

        # Allocate pools
        self.U          = torch.zeros((self.current_blocks, max_seq_len, rank), device=device, dtype=self.dtype)
        self.V_K        = torch.zeros((self.current_blocks, rank, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.V_V        = torch.zeros((self.current_blocks, rank, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.anchors_K  = torch.zeros((self.current_blocks, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.anchors_V  = torch.zeros((self.current_blocks, num_kv_heads, head_dim), device=device, dtype=self.dtype)
        self.scales     = torch.zeros((self.current_blocks,), device=device, dtype=self.dtype)
        self.seq_lens   = torch.zeros((self.current_blocks,), device=device, dtype=torch.int32)
        
        # Block allocator state
        self._free_indices = list(range(self.current_blocks - 1, -1, -1))
        self._ref_counts = [0] * self.current_blocks

    def _grow_pool(self, increment: int = 512):
        old_blocks = self.current_blocks
        if old_blocks >= self.max_blocks:
            raise RuntimeError(f"NativeBlockPool is out of memory and has reached its absolute maximum limit of {self.max_blocks} blocks!")
        
        new_blocks = min(self.max_blocks, old_blocks + increment)
        added = new_blocks - old_blocks
        if added <= 0:
            return
            
        num_kv_heads = self.V_K.shape[2]
        head_dim = self.V_K.shape[3]
        rank = self.U.shape[2]
        max_seq_len = self.U.shape[1]
        
        new_U = torch.zeros((new_blocks, max_seq_len, rank), device=self.device, dtype=self.dtype)
        new_V_K = torch.zeros((new_blocks, rank, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_V_V = torch.zeros((new_blocks, rank, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_anchors_K = torch.zeros((new_blocks, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_anchors_V = torch.zeros((new_blocks, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_scales = torch.zeros((new_blocks,), device=self.device, dtype=self.dtype)
        new_seq_lens = torch.zeros((new_blocks,), device=self.device, dtype=torch.int32)
        
        new_U[:old_blocks] = self.U
        new_V_K[:old_blocks] = self.V_K
        new_V_V[:old_blocks] = self.V_V
        new_anchors_K[:old_blocks] = self.anchors_K
        new_anchors_V[:old_blocks] = self.anchors_V
        new_scales[:old_blocks] = self.scales
        new_seq_lens[:old_blocks] = self.seq_lens
        
        self.U = new_U
        self.V_K = new_V_K
        self.V_V = new_V_V
        self.anchors_K = new_anchors_K
        self.anchors_V = new_anchors_V
        self.scales = new_scales
        self.seq_lens = new_seq_lens
        
        self._ref_counts.extend([0] * added)
        self._free_indices.extend(range(new_blocks - 1, old_blocks - 1, -1))
        self.current_blocks = new_blocks
        
        import gc
        gc.collect()
        _empty_cache(self.device)
        
    def allocate_block(self) -> int:
        if not self._free_indices:
            self._grow_pool()
        idx = self._free_indices.pop()
        self._ref_counts[idx] = 1
        return idx

    def allocate_blocks(self, count: int) -> list:
        while len(self._free_indices) < count:
            self._grow_pool()
        allocated = self._free_indices[-count:]
        del self._free_indices[-count:]
        for idx in allocated:
            self._ref_counts[idx] = 1
        return allocated
        
    def increment_ref(self, pool_idx: int):
        if pool_idx is not None and 0 <= pool_idx < self.current_blocks:
            self._ref_counts[pool_idx] += 1

    def free_block(self, pool_idx: int):
        if pool_idx is not None and 0 <= pool_idx < self.current_blocks:
            self._ref_counts[pool_idx] -= 1
            if self._ref_counts[pool_idx] <= 0:
                self._ref_counts[pool_idx] = 0
                self._free_indices.append(pool_idx)
        
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
        """
        self.U[pool_idx, :seq_len, :U.shape[1]] = U.to(self.dtype)
        
        rank = V.shape[0]
        num_kv = self.V_K.shape[2]
        h_dim = self.V_K.shape[3]
        
        vk = V[:, :num_kv * h_dim].view(rank, num_kv, h_dim)
        vv = V[:, num_kv * h_dim:].view(rank, num_kv, h_dim)
        
        self.V_K[pool_idx, :rank] = vk.to(self.dtype)
        self.V_V[pool_idx, :rank] = vv.to(self.dtype)
        self.anchors_K[pool_idx] = anchor_K.to(self.dtype)
        self.anchors_V[pool_idx] = anchor_V.to(self.dtype)
        self.scales[pool_idx] = scale
        self.seq_lens[pool_idx] = seq_len

    def reset(self):
        """Completely reset the pool to its initial lightweight state, releasing all grown VRAM."""
        self.current_blocks = self.initial_blocks
        self._free_indices = list(range(self.current_blocks - 1, -1, -1))
        self._ref_counts = [0] * self.current_blocks
        
        num_kv_heads = self.V_K.shape[2]
        head_dim = self.V_K.shape[3]
        rank = self.U.shape[2]
        max_seq_len = self.U.shape[1]
        
        self.U          = torch.zeros((self.current_blocks, max_seq_len, rank), device=self.device, dtype=self.dtype)
        self.V_K        = torch.zeros((self.current_blocks, rank, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        self.V_V        = torch.zeros((self.current_blocks, rank, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        self.anchors_K  = torch.zeros((self.current_blocks, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        self.anchors_V  = torch.zeros((self.current_blocks, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        self.scales     = torch.zeros((self.current_blocks,), device=self.device, dtype=self.dtype)
        self.seq_lens   = torch.zeros((self.current_blocks,), device=self.device, dtype=torch.int32)
        
        import gc
        gc.collect()
        _empty_cache(self.device)

