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

import os
import torch
from typing import Optional, List, Union, Tuple

# SRL descriptor dimension — must match native_core/srl/chunk_descriptor.py
_SRL_DESC_DIM = 64

_BLOCK_TRUNCATION_WARNED = False


def _warn_block_truncation(seq_len: int, pool_max_seq: int) -> None:
    """Block content dropped on write because it exceeded pool capacity.

    Both write paths clamp to `min(seq_len, pool_max_seq)` and zero the slot
    first, so tokens past capacity are not merely lost -- they reconstruct as
    delta=0, i.e. exact copies of the anchor, and the decoders admit them to the
    softmax as genuine tokens. This used to happen with NO signal at all because
    seq_lens recorded the untruncated length. It is a data-loss event, so it says
    so once, loudly, rather than being inferred later from bad recall.
    """
    global _BLOCK_TRUNCATION_WARNED
    if seq_len <= pool_max_seq or _BLOCK_TRUNCATION_WARNED:
        return
    _BLOCK_TRUNCATION_WARNED = True
    print(f"[DKV WARNING] block truncated on write: seq_len={seq_len} > pool "
          f"capacity={pool_max_seq}; {seq_len - pool_max_seq} token(s) DROPPED. "
          f"seq_lens now records {pool_max_seq} so the decoder does not attend "
          f"phantom anchor copies, but that content is gone -- raise "
          f"micro_block_size/max_seq_len to hold a full block.")
import gc
try:
    from native_core.mac_utils import empty_cache as _empty_cache
except ImportError:
    def _empty_cache(device=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _JointVAdapter:
    """Presents ``V_KV`` [n, 2, r, H, D] as the [n, r, 2*H*D] joint form the
    shared-basis registry works in.

    The registry needs a basis as one [r, F] matrix — K and V halves
    concatenated along the feature axis, which is the layout ``compress``
    produces and the layout the projection math is written for.  The pool
    stores the two halves split, and that split is load-bearing: ``V_K`` and
    ``V_V`` are contiguous slices of it and every decode kernel reads them that
    way.  Reshaping between the two needs a permute, so it cannot be a view.

    Rather than change the pool's layout (which every kernel, the paging store
    and three metadata readers depend on) this adapter pays for one small copy
    at the boundary.  It is affordable because it only runs at COMPRESS time and
    only over the handful of candidate groups a block is scored against —
    never per decode step, never over the whole store.
    """

    __slots__ = ("_t", "_F")

    def __init__(self, V_KV: torch.Tensor):
        self._t = V_KV
        self._F = int(V_KV.shape[3]) * int(V_KV.shape[4]) * 2

    @property
    def shape(self):
        return (int(self._t.shape[0]), int(self._t.shape[2]), self._F)

    @property
    def device(self):
        return self._t.device

    @property
    def dtype(self):
        return self._t.dtype

    def numel(self):
        return self._t.numel()

    def new_zeros(self, shape, **kw):
        return self._t.new_zeros(shape, **kw)

    def __getitem__(self, rows):
        """[r, F] for a scalar row, [m, r, F] for a tensor/list of rows."""
        scalar = not (torch.is_tensor(rows) or isinstance(rows, (list, tuple, slice)))
        sel = self._t[rows]                          # [.., 2, r, H, D]
        if scalar:
            # [2, r, H, D] -> [r, 2*H*D]
            return sel.permute(1, 0, 2, 3).reshape(int(sel.shape[1]), self._F)
        return sel.permute(0, 2, 1, 3, 4).reshape(
            int(sel.shape[0]), int(sel.shape[2]), self._F)

    def __setitem__(self, row, value):
        """value: [r, F] (or [m, r, F] for a batch of rows)."""
        half = self._F // 2
        H, D = int(self._t.shape[3]), int(self._t.shape[4])
        v = value.to(self._t.dtype)
        if v.dim() == 2:
            r = int(v.shape[0])
            self._t[row, 0, :r] = v[:, :half].reshape(r, H, D)
            self._t[row, 1, :r] = v[:, half:].reshape(r, H, D)
        else:
            m, r = int(v.shape[0]), int(v.shape[1])
            self._t[row, 0, :r] = v[:, :, :half].reshape(m, r, H, D)
            self._t[row, 1, :r] = v[:, :, half:].reshape(m, r, H, D)

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
        max_residual_tokens: Optional[int] = None,
        shared_basis: bool = False,
        shared_basis_frac: float = 0.50,
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
        import os as _os
        if max_residual_tokens is not None:
            self.max_residual_tokens = max_residual_tokens
        else:
            _env_max_res = _os.environ.get("DKV_MAX_RESIDUAL_TOKENS")
            self.max_residual_tokens = int(_env_max_res) if _env_max_res and _env_max_res.isdigit() else 8

        _is_mps = (str(device) == "mps" or
                   (isinstance(device, torch.device) and device.type == "mps"))
        self._grow_increment = 128 if _is_mps else 512
        self._is_mps         = _is_mps

        # ── Legacy slots: stratified-U (U_sem/U_sem_scale/U_fact/n_semantic) and
        # fact anchors.  ONLY the CPU compress path writes these (via
        # write_block's optional kwargs, from finalize_compressed_blocks).  The
        # CUDA GPU-compress path (compress_layer_blocks_gpu, the default on CUDA)
        # never passes them, so they stay all-zero/-1 while still costing
        # ~42 KB per slot (≈11% of a slot) AND still being handed to the decode
        # kernel every token — HAS_FACT=True makes it loop 3 dead fact slots per
        # block, per layer, per token.  MLX's session store has no equivalent.
        # Skip allocating them when the GPU path owns compression; every reader
        # already treats "missing" as absent (getattr(...) is None →
        # _build_stratified_U_for_triton early-returns, has_fact=False).
        # ENABLED 2026-08-13, after guarding all fourteen readers (seven paired
        # block-property getters in kv_runtime_manager.py and their duplicates in
        # streaming_sparse_ingest.py) plus _build_stratified_U_for_triton, which
        # tested `hasattr(pool, "n_semantic")` — true even when the attribute is
        # None. Every one now returns None/0 for "absent", which every caller
        # already handled.
        #
        # This was previously deferred as "<1% of the ~15 GB peak", and that
        # arithmetic was against the wrong denominator — the same denominator
        # trap ACTIVE_RUNTIME/docs/cuda_port_record.md warns about (report the KV-side ratio). Measured against the
        # POOL, which is the only line KV compression can move: on Qwen3.5-2B at
        # 32k the legacy slots are 52 MB of a 166 MB pool, i.e. 31% of it.
        #
        # Writes are safe on the fallback path: write_block lazily builds the
        # slots via _ensure_legacy_slots() if CPU compress output ever arrives.
        # DKV_LEGACY_SLOTS forces the old always-allocate behaviour, which is
        # what the A/B control that gated this change ran against.
        _is_cuda_dev = (str(device) == "cuda" or
                        (isinstance(device, torch.device) and device.type == "cuda"))
        _gpu_compress = _os.environ.get("DKV_GPU_COMPRESS", "1") == "1"
        _force_legacy = _os.environ.get("DKV_LEGACY_SLOTS")
        self._needs_legacy_slots = (
            _force_legacy == "1" if _force_legacy is not None
            else not (_is_cuda_dev and _gpu_compress))

        # ── Shared low-rank bases (DKV_SHARED_BASIS, default off) ────────────
        # When on, V_KV holds ceil(frac * n_blocks) BASIS rows instead of one
        # per block, and `basis_of` maps a slot to the row it reads.  See
        # native_core/compression/basis_group.py for the projection math and
        # the capacity contract.
        # `shared_basis` / `shared_basis_frac` come from the preset (DKVConfig
        # turns it on for `low`, the memory-priority rung) and are overridden by
        # DKV_SHARED_BASIS / DKV_SHARED_BASIS_FRAC. The env check has to be for
        # an EXPLICIT setting, not a truthy one: shared_basis_enabled() returns
        # False when the variable is absent, so consulting it unconditionally
        # would let "unset" silently override a preset that asked for it on.
        try:
            from native_core.compression.basis_group import (
                shared_basis_enabled as _sb_on,
                shared_basis_fraction as _sb_frac,
            )
            if _os.environ.get("DKV_SHARED_BASIS") is not None:
                self._shared_basis = bool(_sb_on())
            else:
                self._shared_basis = bool(shared_basis)
            if _os.environ.get("DKV_SHARED_BASIS_FRAC") is not None:
                self._basis_frac = float(_sb_frac())
            else:
                self._basis_frac = min(1.0, max(1e-3, float(shared_basis_frac)))
        except Exception:                                        # noqa: BLE001
            self._shared_basis = False
            self._basis_frac = 1.0
        # Residuals in CORRECTION form are a delta against the low-rank
        # reconstruction, so re-expressing a block in a shared basis would
        # invalidate every one of them.  In EXACT form (the default on every
        # device) they hold the anchor-relative true K/V and do not depend on
        # the basis at all, which is what makes sharing safe.  Refuse rather
        # than silently corrupt.
        if self._shared_basis:
            try:
                from native_core.compression.lowrank import _exact_keys_enabled
                if not _exact_keys_enabled(device):
                    print("[DKV] DKV_SHARED_BASIS ignored: residuals are in "
                          "CORRECTION form (DKV_RESIDUAL_EXACT_KEYS=0), which "
                          "is defined against a block's OWN low-rank "
                          "reconstruction. Re-expressing the block in a shared "
                          "basis would invalidate every stored residual.",
                          flush=True)
                    self._shared_basis = False
            except Exception:                                    # noqa: BLE001
                pass
        # 4-bit KV quantisation and shared bases are ANTAGONISTIC, measured.
        # The quantisation noise dominates the delta subspaces, so no two blocks
        # clear the join threshold: on `low` (q4_0) the store fills with
        # founders and every later block is FORCE-joined --
        #
        #     preset  kv_quant   groups     joined  forced  mean_kept
        #     mid     f16        293/462       463       0      0.969
        #     low     q4_0       462/462 FULL    0     294      0.685
        #
        # -- while the pool MB is IDENTICAL either way, because the saving comes
        # from allocating fewer basis rows and not from successful grouping. So
        # this degrades silently if you only watch memory. Say so at
        # construction rather than leaving it to be found in basis_stats().
        if self._shared_basis:
            _q = str(getattr(getattr(self, "config", None), "kv_quant", "") or "")
            if not _q:
                _q = str(_os.environ.get("DKV_KV_QUANT", "") or "")
            if _q.lower().startswith(("q4", "int4", "nf4")):
                print(f"[DKV WARNING] shared bases with kv_quant={_q}: 4-bit KV "
                      f"quantisation destroys the subspace agreement this "
                      f"depends on. Expect ZERO voluntary joins and forced "
                      f"lossy sharing (measured mean_kept 0.685 vs 0.969 at "
                      f"f16). The VRAM saving is unchanged, which is why this "
                      f"is easy to miss -- check pool.basis_stats()['joined'].",
                      flush=True)
        # ── SHARED BASES REQUIRE AN UNROTATED POOL ──────────────────────────
        # Shared low-rank bases compare SUBSPACES. RoPE rotates every key by its
        # ABSOLUTE POSITION, so two blocks holding the same text at different
        # offsets have subspaces rotated apart and the grouping collapses.
        # Measured on the MLX port, same document, same block size, frac=0.50,
        # with only DKV_ROTATED_POOL differing:
        #
        #   pool        best-partner retained energy   founded  joined  forced
        #   rotated     mean 0.486,  0/27 clear 0.90       560      10     186
        #   unrotated   mean 0.972, 26/27 clear 0.90       236     520       0
        #
        # The unrotated row reproduces this pool's own published numbers
        # (joined 463, forced 0, mean_kept 0.969), which is what identifies
        # rotation as the whole mechanism.
        #
        # CUDA has never hit this because it got lucky: every preset that turns
        # sharing on -- mid, high, ultra -- already sets rotated_pool=False. But
        # `low` is the one ROTATED preset, so DKV_SHARED_BASIS=1 on `low`
        # degenerates exactly as above.
        #
        # REFUSE, don't warn. The kv_quant note above warns because a 4-bit pool
        # still produces a usable (if badly grouped) result; this one produces no
        # error, no shape change, and the FULL memory win -- pool MB is identical
        # either way, because the saving comes from allocating fewer basis rows
        # rather than from grouping succeeding. A rotated run therefore reports
        # the expected saving with its fidelity quietly bought by forced lossy
        # joins. These are two INDEPENDENT reasons sharing fails on `low`;
        # excusing one does not cover the other.
        #
        # DKV_SHARED_BASIS_ALLOW_ROTATED=1 keeps the bad configuration
        # measurable, because a guard that cannot be turned off cannot be
        # A/B'd against.
        if self._shared_basis:
            try:
                from native_core.sparse_decode.triton_fused_decode import (
                    pool_stores_rotated_k as _psr,
                )
                _rot = bool(_psr())
            except Exception:                                    # noqa: BLE001
                _rot = str(_os.environ.get("DKV_ROTATED_POOL", "1")).strip().lower()                     not in ("0", "false", "off", "no")
            if _rot:
                if str(_os.environ.get("DKV_SHARED_BASIS_ALLOW_ROTATED", "0")
                       ).strip().lower() in ("1", "true", "on", "yes"):
                    print("[DKV WARNING] shared bases on a ROTATED pool "
                          "(DKV_SHARED_BASIS_ALLOW_ROTATED=1). Grouping will "
                          "degenerate to forced lossy joins while the memory "
                          "number looks correct -- check "
                          "pool.basis_stats()['forced'].", flush=True)
                else:
                    print("[DKV] DKV_SHARED_BASIS ignored: the pool stores "
                          "ROTATED keys (DKV_ROTATED_POOL=1, which the `low` "
                          "preset sets). RoPE rotates each key by its absolute "
                          "position, so blocks holding the same text at "
                          "different offsets have subspaces rotated apart and "
                          "no two blocks clear the join threshold -- the store "
                          "fills with founders and everything later is "
                          "FORCE-joined, at the full advertised memory saving. "
                          "Set DKV_ROTATED_POOL=0 to use sharing, or "
                          "DKV_SHARED_BASIS_ALLOW_ROTATED=1 to measure it "
                          "anyway.", flush=True)
                    self._shared_basis = False

        self.basis_of = None          # [n_blocks] int32 device tensor, or None
        self.basis_registry = None
        self.basis_store = None       # _JointVAdapter over V_KV, or None

        # Bytes per block — used for n_blocks computation in ensure_allocated.
        # Under shared bases V is amortised across `1/frac` slots, so the same
        # VRAM budget derives proportionally MORE blocks — that is where the
        # saving is actually spent.
        _v_share = self._basis_frac if self._shared_basis else 1.0
        self._bytes_per_block = (
            max_seq_len * rank * 1 +              # U  (int8)
            int(rank * num_kv_heads * head_dim * 2 * 2 * _v_share) +  # V_K + V_V (fp16)
            num_kv_heads * head_dim * 2 * 2 +     # anchors K + V (fp16)
            6 + 2 +                               # scales (2B) + seq_lens (4B) + U_scale (2B)
            self.max_residual_tokens * 2 +        # residual_K_positions (2B, int16)
            self.max_residual_tokens * 2 +        # residual_V_positions (2B, int16)
            self.max_residual_tokens * num_kv_heads * head_dim * 2 +  # residual_K_values (fp16)
            self.max_residual_tokens * num_kv_heads * head_dim * 2    # residual_V_values (fp16)
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

        # OPT-D: Generation counter for the stratified U proxy cache in
        # triton_fused_decode.py.  Incremented on every write_block call so
        # the decode-side cache knows when stratified U data has changed.
        self._stratified_generation: int = 0

        # Widest routing block actually written, in tokens. 0 until the first
        # write; the router falls back to routing_topk_default until then.
        self.observed_block_span: int = 0

        self.lazy = lazy
        if not lazy:
            self._allocate_tensors(initial_blocks)
            self._allocated = True

    # ── Phase 2: Actual GPU allocation ───────────────────────────────────────
    def _required_blocks(self, n_tokens: int = None) -> int:
        """Return the slot count needed for prefill plus bounded decode headroom."""
        if n_tokens is None or n_tokens <= 0:
            n_tokens = self._default_token_hint

        blocks_per_layer = max(
            1, (int(n_tokens) + self.max_seq_len - 1) // self.max_seq_len
        )

        # One guard block per layer absorbs the first decode block after a
        # partially filled prefill tail.  Additional headroom is opt-in so
        # long-generation serving can trade memory for fewer pool grows.
        import os as _os_pool
        try:
            reserve_tokens = int(
                _os_pool.environ.get("DKV_DECODE_RESERVE_TOKENS", "0")
            )
        except ValueError:
            reserve_tokens = 0
        reserve_tokens = max(0, reserve_tokens)
        reserve_blocks_per_layer = 1 + (
            reserve_tokens + self.max_seq_len - 1
        ) // self.max_seq_len
        # Size by the layers DKV actually COMPRESSES, not by the model's total
        # layer count.  On a hybrid model most layers are linear-attention and
        # never hold a block: Qwen3.5-2B has 24 layers but only 6 attended
        # (3, 7, 11, 15, ...), so sizing by 24 allocated 4x the slots the session
        # can ever use -- 1246 MB of pool for ~312 MB of blocks at 32k, dead VRAM
        # that is the reason DKV measured ABOVE dense on real device memory.
        #
        # Safe to under-estimate: slots come from one global free list
        # (allocate_block pops _free_indices, they are not partitioned per layer)
        # and _grow_pool() covers a shortfall.  Falls back to num_layers when the
        # attended count was never published, i.e. non-hybrid models where the
        # two are equal anyway.
        _sizing_layers = getattr(self, "sizing_layers", None) or self.num_layers
        return (blocks_per_layer + reserve_blocks_per_layer) * _sizing_layers

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

        n_desired = self._required_blocks(n_tokens)

        if self._is_mps:
            n_blocks = max(64, min(n_desired, self.max_blocks // 2, 512))
        else:
            # CUDA: pre-allocate exact prefill capacity plus the small guard
            # above.  _grow_pool handles unusually long generations.
            n_blocks = max(64, min(n_desired, self.max_blocks))

        self._allocate_tensors(n_blocks)
        self._allocated = True
        print(f"[Pool] Lazy-allocated {n_blocks} slots for ~{n_tokens} tokens "
              f"= {self._pool_mb():.1f} MB (device={self.device})")

    def _ensure_legacy_slots(self) -> None:
        """Allocate the stratified-U / fact-anchor slots late.

        Only the CPU compress path writes these. It is not dead code -- the GPU
        path falls back to it on failure -- but it is not the path that normally
        runs, so the slots are not allocated up front. Sized at current_blocks,
        and _grow_pool carries them forward once they exist because it keys off
        _needs_legacy_slots, which this flips.
        """
        if self.n_semantic is not None or not self._allocated:
            return
        n, r, S = self.current_blocks, self.rank, self.max_seq_len
        H, D = self.num_kv_heads, self.head_dim
        self.U_sem = torch.zeros((n, S // 2, r), device=self.device, dtype=torch.int8)
        self.U_sem_scale = torch.zeros((n, r), device=self.device, dtype=self.dtype)
        self.U_fact = torch.zeros((n, S, r), device=self.device, dtype=self.dtype)
        self.n_semantic = torch.zeros((n,), device=self.device, dtype=torch.int16)
        self.fact_anchors_K = torch.zeros((n, 3, H, D), device=self.device, dtype=self.dtype)
        self.fact_anchors_V = torch.zeros((n, 3, H, D), device=self.device, dtype=self.dtype)
        self.fact_anchor_positions = torch.full((n, 3), -1, device=self.device,
                                                dtype=torch.int16)
        self._needs_legacy_slots = True
        print("[DKV] CPU compress path active — allocated stratified/fact slots "
              f"for {n} blocks", flush=True)

    # ── Shared-basis plumbing ────────────────────────────────────────────────
    #
    # Every one of these is a no-op returning the identity when
    # DKV_SHARED_BASIS is off, which is the default: `basis_of` stays None,
    # `_n_basis_rows` returns n_blocks, and `basis_index` hands back the slot
    # indices it was given.  The pool then allocates and behaves exactly as it
    # did before this existed.

    @property
    def shared_basis_active(self) -> bool:
        return bool(self._shared_basis) and self.basis_of is not None

    def _n_basis_rows(self, n_blocks: int) -> int:
        if not self._shared_basis:
            return n_blocks
        import math as _math
        return max(1, min(n_blocks, int(_math.ceil(n_blocks * self._basis_frac))))

    def _init_basis_map(self, n_blocks: int) -> None:
        """(Re)build the slot -> basis-row map and its registry."""
        if not self._shared_basis:
            self.basis_of = None
            self.basis_registry = None
            self.basis_store = None
            return
        from native_core.compression.basis_group import SharedBasisRegistry
        n_basis = int(self.V_KV.shape[0])
        # Row 0, not -1: an UNWRITTEN slot must still resolve to a valid row or
        # any gather over it indexes out of bounds.  Row 0 is safe as a landing
        # place because an unwritten slot's U is all zeros, so it reconstructs
        # to exactly the anchor whatever basis it points at.
        self.basis_of = torch.zeros((n_blocks,), device=self.device, dtype=torch.int32)
        # ...but "points at row 0" is then indistinguishable from "holds a
        # refcount on row 0", and releasing a claim that was never made
        # corrupts the registry: the founding block's own group gets
        # decremented by the next slot to be written, the row returns to the
        # free list while blocks still read it, and a later block RE-FOUNDS it
        # with a different basis -- silently changing what those earlier blocks
        # decompress to.  This flag is the difference.
        self._basis_claimed = bytearray(n_blocks)
        self.basis_store = _JointVAdapter(self.V_KV)
        self.basis_registry = SharedBasisRegistry(
            capacity=n_basis, device=self.V_KV.device, dtype=self.dtype)

    def basis_index(self, indices):
        """Map pool slot ids to V-store rows.  Identity when sharing is off."""
        if self.basis_of is None:
            return indices
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, device=self.basis_of.device)
        idx = indices.to(device=self.basis_of.device, dtype=torch.long)
        idx = idx.clamp(0, self.basis_of.shape[0] - 1)
        return self.basis_of[idx].long()

    def basis_row(self, pool_idx: int) -> int:
        """Scalar form of basis_index, for the metadata readers."""
        if self.basis_of is None:
            return int(pool_idx)
        if not (0 <= pool_idx < self.basis_of.shape[0]):
            return 0
        return int(self.basis_of[pool_idx].item())

    def release_basis(self, pool_idx: int) -> None:
        """Drop this slot's claim on its basis row so the row can be reclaimed.

        No-op unless the slot actually holds a claim — see `_basis_claimed`.
        """
        if self.basis_registry is None or self.basis_of is None:
            return
        if not (0 <= pool_idx < self.basis_of.shape[0]):
            return
        if not self._basis_claimed[pool_idx]:
            return
        self._basis_claimed[pool_idx] = 0
        try:
            self.basis_registry.release_row(int(self.basis_of[pool_idx].item()))
        except Exception:                                        # noqa: BLE001
            pass

    def _claim_basis(self, pool_idx: int, row: int) -> None:
        self.basis_of[pool_idx] = int(row)
        self._basis_claimed[pool_idx] = 1

    def assign_basis(self, U, V, layer_idx: int, pool_indices):
        """Pick a shared basis for each block and return (rows, bases, kept).

        Exposed so the COMPRESS path can assign before it measures
        reconstruction error.  It has to: residual selection scores
        ``delta - recon``, and under a shared basis the stored recon comes from
        the GROUP basis, not the block's own.  Assigning inside write_block
        would leave every residual chosen against a reconstruction that is not
        the one decode rebuilds.

        Callers that pre-assign pass the returned rows back to
        write_blocks_batched(basis_rows=...), which then only records the map.

        U: [N, T, r]  V: [N, r, F]  ->  rows [N] long, bases [N, r, F], kept [N]
        """
        from native_core.compression.basis_group import reproject_U  # noqa: F401
        slots = [int(x) for x in (pool_indices.tolist()
                                  if torch.is_tensor(pool_indices) else pool_indices)]
        for s in slots:
            self.release_basis(s)
        asg, gathered = self.basis_registry.assign_batch(
            U, V, layer=int(layer_idx), basis_store=self.basis_store)
        for s, a in zip(slots, asg):
            self._claim_basis(s, a.row)
        rows = torch.tensor([a.row for a in asg], device=self.V_KV.device, dtype=torch.long)
        kept = torch.tensor([a.kept for a in asg], device=self.V_KV.device, dtype=torch.float32)
        return rows, gathered, kept

    def basis_stats(self) -> dict:
        if self.basis_registry is None:
            return {"enabled": False}
        s = dict(self.basis_registry.stats())
        s["enabled"] = True
        s["frac"] = self._basis_frac
        n_slots = int(self.basis_of.shape[0]) if self.basis_of is not None else 0
        s["slots"] = n_slots
        # Sharing factor over slots that ACTUALLY hold a basis claim, not over
        # pool capacity. Dividing capacity by group count reports 1/frac
        # whenever the store is full regardless of how many blocks were
        # written -- i.e. it reports the CONFIG back, not the outcome.
        n_claimed = int(sum(self._basis_claimed)) if self.basis_of is not None else 0
        s["claimed"] = n_claimed
        s["sharing_factor"] = (n_claimed / s["groups"]) if s["groups"] else 0.0
        return s

    def _allocate_tensors(self, n_blocks: int) -> None:
        """Allocate (or re-allocate) all pool tensors at *n_blocks* size."""
        self.current_blocks = n_blocks
        self.U          = torch.zeros((n_blocks, self.max_seq_len, self.rank), device=self.device, dtype=torch.int8)
        self.U_scale    = torch.zeros((n_blocks,), device=self.device, dtype=self.dtype)
        # Stratified-U slots — only the CPU compress path fills these (see
        # _needs_legacy_slots).  None on the CUDA GPU-compress path.
        if self._needs_legacy_slots:
            self.U_sem      = torch.zeros((n_blocks, self.max_seq_len // 2, self.rank), device=self.device, dtype=torch.int8)
            self.U_sem_scale = torch.zeros((n_blocks, self.rank), device=self.device, dtype=self.dtype)
            self.U_fact     = torch.zeros((n_blocks, self.max_seq_len, self.rank), device=self.device, dtype=self.dtype)
            self.n_semantic = torch.zeros((n_blocks,), device=self.device, dtype=torch.int16)
        else:
            self.U_sem = self.U_sem_scale = self.U_fact = self.n_semantic = None
        n_basis = self._n_basis_rows(n_blocks)
        self.V_KV       = torch.zeros((n_basis, 2, self.rank, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
        self._init_basis_map(n_blocks)
        self.anchors_KV = torch.zeros((n_blocks, 2, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
        self.scales     = torch.zeros((n_blocks,), device=self.device, dtype=self.dtype)
        self.seq_lens   = torch.zeros((n_blocks,), device=self.device, dtype=torch.int32)
        self.desc       = torch.zeros((n_blocks, _SRL_DESC_DIM), device=self.device, dtype=torch.float16)

        self.residual_K_positions = torch.full((n_blocks, self.max_residual_tokens), -1, device=self.device, dtype=torch.int16)
        self.residual_K_values = torch.zeros((n_blocks, self.max_residual_tokens, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
        self.residual_V_positions = torch.full((n_blocks, self.max_residual_tokens), -1, device=self.device, dtype=torch.int16)
        self.residual_V_values = torch.zeros((n_blocks, self.max_residual_tokens, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)

        # Fact Anchors (Solution 3) — CPU-compress path only; None on the CUDA
        # GPU path so the decode kernel gets HAS_FACT=False instead of looping
        # 3 all-(-1) slots per block per layer per token.
        if self._needs_legacy_slots:
            self.fact_anchors_K = torch.zeros((n_blocks, 3, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
            self.fact_anchors_V = torch.zeros((n_blocks, 3, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
            self.fact_anchor_positions = torch.full((n_blocks, 3), -1, device=self.device, dtype=torch.int16)
        else:
            self.fact_anchors_K = self.fact_anchors_V = self.fact_anchor_positions = None

        self._free_indices     = list(range(n_blocks - 1, -1, -1))
        self._free_indices_set = set(self._free_indices)
        self._ref_counts       = [0] * n_blocks
        self._last_used        = [0.0] * n_blocks
        self.version           = [0] * n_blocks

        # B1: Cached residual presence flag — updated at write_block time so that the
        # decode loop can read a plain Python bool instead of calling .item() on a
        # device tensor every layer every step (~4096 device→host syncs per generation).
        self.has_any_residual: bool = False

        # Re-attach W_proj at the new size if it was already set
        if self.W_proj is not None and self.W_proj.device != torch.device("cpu"):
            pass  # W_proj is a [DESC_DIM, head_dim] matrix — shape is independent of n_blocks

    def _pool_mb(self) -> float:
        """Current pool VRAM usage in megabytes."""
        total = 0
        attrs = ("U", "U_scale", "V_KV", "anchors_KV", "scales", "seq_lens", "desc",
                 "residual_K_positions", "residual_K_values", "residual_V_positions", "residual_V_values",
                 "U_sem", "U_sem_scale", "U_fact", "n_semantic",
                 "fact_anchors_K", "fact_anchors_V", "fact_anchor_positions")
        for attr in attrs:
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
        # empty_cache here lets the allocator reclaim the old pool pages before
        # the new (larger) tensors are created, cutting the momentary peak from
        # ~2x to ~1.x of the new pool size.
        #
        # No gc.collect(). The pool GROWS DURING DECODE as generated tokens form
        # blocks, and a decode-only profile put gc.collect at 3.59 ms/token at
        # 32k -- ~140 ms for a single call, on a path where the whole token costs
        # ~90 ms. It also frees nothing: the same isolation done for the prefill
        # trim measured reserved memory identical with and without the collect
        # (8.82 GB either way), because CPython refcounts the old pool tensors
        # away as soon as they are rebound. empty_cache is what reclaims.
        _empty_cache(self.device)
        
        _legacy = self._needs_legacy_slots
        new_U = torch.zeros((new_blocks, max_seq_len, rank), device=self.device, dtype=torch.int8)
        new_U_scale = torch.zeros((new_blocks,), device=self.device, dtype=self.dtype)
        new_U_sem = torch.zeros((new_blocks, max_seq_len // 2, rank), device=self.device, dtype=torch.int8) if _legacy else None
        new_U_sem_scale = torch.zeros((new_blocks, rank), device=self.device, dtype=self.dtype) if _legacy else None
        new_U_fact = torch.zeros((new_blocks, max_seq_len, rank), device=self.device, dtype=self.dtype) if _legacy else None
        new_n_semantic = torch.zeros((new_blocks,), device=self.device, dtype=torch.int16) if _legacy else None
        new_n_basis = self._n_basis_rows(new_blocks)
        new_V_KV = torch.zeros((new_n_basis, 2, rank, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_anchors_KV = torch.zeros((new_blocks, 2, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_scales = torch.zeros((new_blocks,), device=self.device, dtype=self.dtype)
        new_seq_lens = torch.zeros((new_blocks,), device=self.device, dtype=torch.int32)
        new_desc = torch.zeros((new_blocks, _SRL_DESC_DIM), device=self.device, dtype=torch.float16)

        new_res_K_pos = torch.full((new_blocks, self.max_residual_tokens), -1, device=self.device, dtype=torch.int16)
        new_res_K_val = torch.zeros((new_blocks, self.max_residual_tokens, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)
        new_res_V_pos = torch.full((new_blocks, self.max_residual_tokens), -1, device=self.device, dtype=torch.int16)
        new_res_V_val = torch.zeros((new_blocks, self.max_residual_tokens, num_kv_heads, head_dim), device=self.device, dtype=self.dtype)

        new_fact_anc_K = torch.zeros((new_blocks, 3, num_kv_heads, head_dim), device=self.device, dtype=self.dtype) if _legacy else None
        new_fact_anc_V = torch.zeros((new_blocks, 3, num_kv_heads, head_dim), device=self.device, dtype=self.dtype) if _legacy else None
        new_fact_anc_pos = torch.full((new_blocks, 3), -1, device=self.device, dtype=torch.int16) if _legacy else None

        new_U[:old_blocks] = self.U
        new_U_scale[:old_blocks] = self.U_scale
        if _legacy:
            new_U_sem[:old_blocks] = self.U_sem
            new_U_sem_scale[:old_blocks] = self.U_sem_scale
            new_U_fact[:old_blocks] = self.U_fact
            new_n_semantic[:old_blocks] = self.n_semantic
        new_V_KV[:self.V_KV.shape[0]] = self.V_KV
        new_anchors_KV[:old_blocks] = self.anchors_KV
        new_scales[:old_blocks] = self.scales
        new_seq_lens[:old_blocks] = self.seq_lens
        new_desc[:old_blocks] = self.desc

        new_res_K_pos[:old_blocks] = self.residual_K_positions
        new_res_K_val[:old_blocks] = self.residual_K_values
        new_res_V_pos[:old_blocks] = self.residual_V_positions
        new_res_V_val[:old_blocks] = self.residual_V_values

        if _legacy:
            new_fact_anc_K[:old_blocks] = self.fact_anchors_K
            new_fact_anc_V[:old_blocks] = self.fact_anchors_V
            new_fact_anc_pos[:old_blocks] = self.fact_anchor_positions

        # Explicitly delete old tensors so the allocator can reclaim them
        del (self.U, self.U_scale, self.V_KV, self.anchors_KV, self.scales, self.seq_lens, self.desc,
             self.residual_K_positions, self.residual_K_values, self.residual_V_positions, self.residual_V_values,
             self.U_sem, self.U_sem_scale, self.U_fact, self.n_semantic,
             self.fact_anchors_K, self.fact_anchors_V, self.fact_anchor_positions)

        self.U = new_U
        self.U_scale = new_U_scale
        self.U_sem = new_U_sem
        self.U_sem_scale = new_U_sem_scale
        self.U_fact = new_U_fact
        self.n_semantic = new_n_semantic
        self.V_KV = new_V_KV
        # Grow the slot -> basis map and hand the registry its extra rows.  The
        # existing map entries stay valid because growth only APPENDS rows: a
        # basis row keeps its index, so no already-written slot has to be
        # re-pointed or re-projected.
        if self._shared_basis and self.basis_of is not None:
            _old_map = self.basis_of
            self.basis_of = torch.zeros((new_blocks,), device=self.device, dtype=torch.int32)
            self.basis_of[:_old_map.shape[0]] = _old_map
            self._basis_claimed = self._basis_claimed + bytearray(new_blocks - len(self._basis_claimed))
            self.basis_store = _JointVAdapter(self.V_KV)
            if self.basis_registry is not None:
                _added_rows = new_n_basis - self.basis_registry.capacity
                if _added_rows > 0:
                    self.basis_registry.capacity = new_n_basis
                    self.basis_registry._free_rows.extend(
                        range(new_n_basis - 1, new_n_basis - 1 - _added_rows, -1))
        self.anchors_KV = new_anchors_KV
        self.scales = new_scales
        self.seq_lens = new_seq_lens
        self.desc = new_desc

        self.residual_K_positions = new_res_K_pos
        self.residual_K_values = new_res_K_val
        self.residual_V_positions = new_res_V_pos
        self.residual_V_values = new_res_V_val

        self.fact_anchors_K = new_fact_anc_K
        self.fact_anchors_V = new_fact_anc_V
        self.fact_anchor_positions = new_fact_anc_pos
        
        self._ref_counts.extend([0] * added)
        self._last_used.extend([0.0] * added)
        self.version.extend([0] * added)
        added_range = range(new_blocks - 1, old_blocks - 1, -1)
        self._free_indices.extend(added_range)
        self._free_indices_set.update(added_range)
        self.current_blocks = new_blocks

        # Same reasoning as the pre-realloc trim above: empty_cache reclaims, the
        # collect only costs. This one runs after the new tensors are live, so
        # there is even less for a collector to find.
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
        # DIAGNOSTIC ONLY (DKV_NO_SLOT_REUSE=1): leak the slot instead of
        # returning it to the free list, so no slot is ever handed to a second
        # block. If a bug disappears under this, the mechanism is a stale
        # slot-index mapping surviving recycling; if it survives, recycling is
        # not involved and that whole class is excluded. Leaks the pool, so it
        # is only usable for short repros -- never a fix.
        if os.environ.get("DKV_NO_SLOT_REUSE") == "1":
            if pool_idx is not None and 0 <= pool_idx < self.current_blocks:
                self._ref_counts[pool_idx] = 0
                self.seq_lens[pool_idx] = 0
            return
        if pool_idx is not None and 0 <= pool_idx < self.current_blocks:
            self._ref_counts[pool_idx] -= 1
            if self._ref_counts[pool_idx] <= 0:
                self._ref_counts[pool_idx] = 0
                self._last_used[pool_idx] = _time.time() # Mark freed time as last used
                # seq_lens is the pool's OCCUPANCY SIGNAL, not just kernel
                # metadata. Freeing a slot without clearing it made occupancy a
                # HIGH-WATER MARK that never came back down:
                #
                #   _occupied_slots() (triton_fused_decode.py:125) is literally
                #       (seq_lens[:current_blocks] > 0).nonzero()
                #   and its result is what TieredBlockStore.maybe_evict divides
                #   by max_blocks to decide whether the pool is past
                #   DKV_TIER_EVICT_THRESH (0.80).
                #
                # Every clear_session frees its slots, so a process that runs
                # several prompts (a benchmark sweep, a validator suite, any
                # multi-turn server) accumulates dead-but-"occupied" slots until
                # occupancy crosses the threshold permanently -- at which point
                # eviction starts firing on LIVE blocks mid-decode, forever,
                # driven entirely by slots that no longer hold anything.
                #
                # Zeroing here is also what makes reuse safe: write_block sets
                # seq_lens on the next write, so a slot is only ever "occupied"
                # between its write and its free, which is what every consumer
                # already assumes.
                #
                # .zero_() and not `= 0`. Assigning a Python scalar into a CUDA
                # tensor stages it through host memory and synchronises; a sync
                # probe over one 32k prefill caught 868 of them here alone (one
                # per block per layer), on a path that is only bookkeeping.
                # zero_() is a device-side fill and never touches the host.
                self.seq_lens[pool_idx].zero_()
                # Give up this slot's claim on its shared basis row. Without
                # this the registry's refcount only ever rises, so a session
                # that cycles topics exhausts basis capacity and every later
                # block is FORCE-JOINED to a group it does not belong in --
                # a fidelity cliff with no error anywhere.
                self.release_basis(pool_idx)
                if pool_idx not in self._free_indices_set:
                    self._free_indices.append(pool_idx)
                    self._free_indices_set.add(pool_idx)
                # A freed slot's bytes are dead. If the tiering layer still
                # believes this slot lives on CPU, its restore path will copy a
                # DIFFERENT block's U/V/anchors -- and seq_len -- over whatever
                # gets written here next (tiered_block_store.restore_slot does
                # an unconditional copy_ keyed only on slot id). Nothing else
                # resets that bookkeeping: allocate_block and write_block do not
                # know the store exists, so without this a recycled slot stays
                # marked 'CPU' with a stale _cpu_store entry attached to it.
                _store = getattr(getattr(self, "_manager", None),
                                 "_kt_tiered_store", None)
                if _store is not None:
                    try:
                        _store.invalidate_slot(pool_idx)
                    except Exception:                            # noqa: BLE001
                        pass

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
        seq_len: int,
        residual_K_positions: Optional[torch.Tensor] = None,
        residual_K_values: Optional[torch.Tensor] = None,
        residual_V_positions: Optional[torch.Tensor] = None,
        residual_V_values: Optional[torch.Tensor] = None,
        U_sem_int4: Optional[torch.Tensor] = None,
        U_sem_scale: Optional[torch.Tensor] = None,
        U_fact_fp16: Optional[torch.Tensor] = None,
        n_semantic: int = 0,
        fact_anchors_K: Optional[torch.Tensor] = None,
        fact_anchors_V: Optional[torch.Tensor] = None,
        fact_anchor_positions: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
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
        
        # Sanitize inputs to prevent NaN/Inf propagation.
        # NOTE: The full torch.isfinite().all() check is a GPU D2H sync on every
        # write_block call (called from background compressor thread).
        # Replace with lightweight Python-level scale check only.
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

        # ── Shared basis (see write_blocks_batched for the ordering note) ────
        _v_row = pool_idx
        _shared = False
        if self._shared_basis and self.basis_of is not None:
            from native_core.compression.basis_group import reproject_U
            U_in = U[:write_seq, :write_rank].float().unsqueeze(0)
            V_in = V[:write_rank, :].float().unsqueeze(0)
            self.release_basis(pool_idx)      # before assign — see the batched path
            asg, gathered = self.basis_registry.assign_batch(
                U_in, V_in, layer=int(layer_idx), basis_store=self.basis_store)
            _v_row = asg[0].row
            self._claim_basis(pool_idx, _v_row)
            U = U.clone()
            U[:write_seq, :write_rank] = reproject_U(
                U_in, V_in, gathered[:, :write_rank, :].float())[0].to(U.dtype)
            _shared = True
        else:
            self.V_KV[pool_idx] = 0

        # Quantize U to int8
        U_sliced = U[:write_seq, :write_rank].float()
        max_abs = U_sliced.abs().max()
        scale_u = torch.clamp(max_abs / 127.0, min=1e-5).to(self.dtype)
        self.U[pool_idx, :write_seq, :write_rank] = torch.clamp(torch.round(U_sliced / scale_u), -127, 127).to(torch.int8)
        self.U_scale[pool_idx] = scale_u

        if not _shared:
            # Split V_K and V_V
            vk = V[:write_rank, :num_kv * h_dim].view(write_rank, num_kv, h_dim)
            vv = V[:write_rank, num_kv * h_dim:].view(write_rank, num_kv, h_dim)

            self.V_KV[pool_idx, 0, :write_rank] = vk.to(self.dtype)
            self.V_KV[pool_idx, 1, :write_rank] = vv.to(self.dtype)
        self.anchors_KV[pool_idx, 0] = anchor_K.to(self.dtype)
        self.anchors_KV[pool_idx, 1] = anchor_V.to(self.dtype)
        self.scales[pool_idx] = scale
        # seq_lens must describe what was ACTUALLY STORED, not what was offered.
        # U above is written only up to write_seq and the slot was zeroed first, so
        # any slot past it reconstructs as delta=0 -> exactly the anchor. Recording
        # the untruncated seq_len told every decoder those slots were real tokens,
        # so overflow tokens silently vanished and were replaced by PHANTOM COPIES
        # OF THE ANCHOR inside the softmax. MLX cannot hit this (fixed
        # block_size=256); this side blocks adaptively, so it can.
        _warn_block_truncation(seq_len, pool_max_seq)
        self.seq_lens[pool_idx] = write_seq
        self.version[pool_idx] += 1

        # Copy residuals
        self.residual_K_positions[pool_idx] = -1
        self.residual_K_values[pool_idx] = 0.0
        self.residual_V_positions[pool_idx] = -1
        self.residual_V_values[pool_idx] = 0.0

        if residual_K_positions is not None and residual_K_positions.numel() > 0:
            n_res_k = min(residual_K_positions.numel(), self.max_residual_tokens)
            self.residual_K_positions[pool_idx, :n_res_k] = residual_K_positions[:n_res_k].to(torch.int16)
            self.residual_K_values[pool_idx, :n_res_k] = residual_K_values[:n_res_k].view(n_res_k, num_kv, h_dim).to(self.dtype)
            # B1: update the cached flag — any valid residual position means True.
            # This is a cheap CPU bool check (residual_K_positions is a small int16 tensor).
            if not self.has_any_residual:
                self.has_any_residual = bool((residual_K_positions >= 0).any().item())

        if residual_V_positions is not None and residual_V_positions.numel() > 0:
            n_res_v = min(residual_V_positions.numel(), self.max_residual_tokens)
            self.residual_V_positions[pool_idx, :n_res_v] = residual_V_positions[:n_res_v].to(torch.int16)
            self.residual_V_values[pool_idx, :n_res_v] = residual_V_values[:n_res_v].view(n_res_v, num_kv, h_dim).to(self.dtype)

        # Copy stratified SVD components (Solution 2).  Skipped entirely when the
        # slots were never allocated (CUDA GPU-compress path — see
        # _needs_legacy_slots); that path never supplies U_sem_int4/n_semantic.
        if self.n_semantic is None and n_semantic:
            # The CPU compress path ran after all -- either DKV_GPU_COMPRESS=0 or
            # the GPU helper raised and streaming_sparse_ingest fell back. Build
            # the slots now rather than refusing the write: the fallback is a
            # runtime event, so a pool built for the GPU path must still be able
            # to accept CPU output.
            self._ensure_legacy_slots()
        if self.n_semantic is None:
            pass
        else:
            self.U_sem[pool_idx] = 0
            self.U_sem_scale[pool_idx] = 0.0
            self.U_fact[pool_idx] = 0.0
            self.n_semantic[pool_idx] = n_semantic
        # Track whether any block has ever received non-zero n_semantic.
        # This is a cheap Python int comparison — no GPU sync.
        # Used by _build_stratified_U_for_triton to skip the GPU .all().item()
        # check when stratified quantization has never been activated.
        if n_semantic > 0 and not getattr(self, "_n_semantic_ever_nonzero", False):
            self._n_semantic_ever_nonzero = True

        if U_sem_int4 is not None and U_sem_int4.numel() > 0:
            write_sem_seq = min(U_sem_int4.shape[0], self.U_sem.shape[1])
            write_sem_rank = min(U_sem_int4.shape[1], self.U_sem.shape[2])
            self.U_sem[pool_idx, :write_sem_seq, :write_sem_rank] = U_sem_int4[:write_sem_seq, :write_sem_rank]
            self.U_sem_scale[pool_idx, :write_sem_rank] = U_sem_scale[:write_sem_rank].to(self.dtype)

        if U_fact_fp16 is not None and U_fact_fp16.numel() > 0:
            write_fact_seq = min(U_fact_fp16.shape[0], self.U_fact.shape[1])
            write_fact_rank = min(U_fact_fp16.shape[1], self.U_fact.shape[2])
            self.U_fact[pool_idx, :write_fact_seq, :write_fact_rank] = U_fact_fp16[:write_fact_seq, :write_fact_rank].to(self.dtype)

        # Copy fact anchors (Solution 3).  Skipped when the slots were never
        # allocated (CUDA GPU-compress path never supplies fact anchors).  Must
        # NOT early-return here — the SRL descriptor and the generation bump
        # below still have to run.
        if self.fact_anchor_positions is not None:
            self.fact_anchors_K[pool_idx] = 0.0
            self.fact_anchors_V[pool_idx] = 0.0
            self.fact_anchor_positions[pool_idx] = -1

            if fact_anchors_K is not None and fact_anchors_K.numel() > 0:
                self.fact_anchors_K[pool_idx] = fact_anchors_K.to(self.dtype)
            if fact_anchors_V is not None and fact_anchors_V.numel() > 0:
                self.fact_anchors_V[pool_idx] = fact_anchors_V.to(self.dtype)
            if fact_anchor_positions is not None and fact_anchor_positions.numel() > 0:
                self.fact_anchor_positions[pool_idx] = fact_anchor_positions.to(torch.int16)

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
                    V_K      = self.V_KV[_v_row, 0, :write_rank],         # [R, kv_heads, D] fp16
                    W_proj   = self.W_proj,                               # [DESC_DIM, D] fp32
                )
            except Exception:
                pass  # Descriptor failure is non-fatal — SRL routing degrades gracefully

        # OPT-D: Notify the stratified U proxy cache that pool data has changed.
        # Incrementing this counter causes _build_stratified_U_for_triton to
        # skip the cached proxy and reconstruct fresh fp16 U for the new data.
        self._stratified_generation += 1

    def write_blocks_batched(
        self,
        pool_indices,            # [N] long — pre-allocated slots (allocate_blocks)
        U,                       # [N, S, R] — per-block U, cols ≥ rank[i] already zeroed
        V,                       # [N, R, 2*kv*hd] — joint V (first half V_K, second V_V)
        anchor_K,                # [N, kv, hd]
        anchor_V,                # [N, kv, hd]
        scales,                  # [N]
        seq_len,                 # int — SHARED across the batch (grouped by T_active)
        res_K_positions=None,    # [N, max_res] int16 (-1 padded) or None
        res_K_values=None,       # [N, max_res, kv, hd] or None
        res_V_positions=None,
        res_V_values=None,
        layer_idx=-1,            # int — which layer these blocks belong to
        basis_rows=None,         # [N] long — rows from a prior assign_basis()
    ):
        """Vectorized equivalent of N write_block() calls that share seq_len.

        Mirrors write_block() field-for-field (int8 U quant with a per-block
        scale, V_K/V_V split, anchors, residuals, SRL descriptor, generation
        bump) but scatters every block in one set of ops instead of one
        write_block() call each — the CUDA compress path issued ~2,352 of those
        per 13K prefill (48 layers × ~49 blocks), each launching ~15 kernels.
        Verified bit-identical to the per-block path by
        test_write_blocks_batched_parity.  Zeroed rank columns (blocks with
        dynamic_rank < R) are exactly what write_block leaves after `self.U=0`,
        so padding to a uniform R changes nothing stored.
        """
        self._ensure()
        N = int(pool_indices.shape[0])
        if N == 0:
            return
        dev = self.U.device
        pidx = pool_indices.to(device=dev, dtype=torch.long)
        pool_max_seq = self.U.shape[1]
        pool_rank    = self.U.shape[2]
        num_kv       = self.V_KV.shape[3]
        h_dim        = self.V_KV.shape[4]
        write_seq  = min(int(seq_len), pool_max_seq)
        write_rank = min(int(U.shape[2]), pool_rank)
        if V.shape[1] > pool_rank:
            raise ValueError(f"V rank {V.shape[1]} exceeds pool rank capacity {pool_rank}")

        # ── Shared basis: assign a group BEFORE quantising U ─────────────────
        # Order matters. Joining a group re-expresses U in that group's basis
        # (U' = U (V Vg^T)), which changes U's magnitudes, so the int8 scale
        # below has to be derived from the RE-PROJECTED U or the quantisation
        # range is set from a vector that is no longer being stored.
        U_eff = U.to(dev)
        if basis_rows is not None:
            # The caller already assigned (compress does, so that its residual
            # selection scores the reconstruction that will actually be stored)
            # and has already re-projected U. Nothing to do but honour the map.
            basis_rows = basis_rows.to(device=dev, dtype=torch.long)
        elif self._shared_basis and self.basis_of is not None:
            from native_core.compression.basis_group import reproject_U
            U_in = U_eff[:, :write_seq, :write_rank].float()
            V_in = V.to(dev)[:, :write_rank, :].float()
            # Release BEFORE assigning. A slot being overwritten must give up
            # its previous claim first, so that if it was a group's last member
            # the row is back on the free list and this write can reuse it.
            # Releasing afterwards would decrement the group this write just
            # joined.
            _slots = pidx.tolist()
            for _p in _slots:
                self.release_basis(_p)
            asg, gathered = self.basis_registry.assign_batch(
                U_in, V_in, layer=int(layer_idx), basis_store=self.basis_store)
            basis_rows = torch.tensor([a.row for a in asg], device=dev, dtype=torch.long)
            for _p, _a in zip(_slots, asg):
                self._claim_basis(_p, _a.row)
            U_new = U_eff.clone()
            U_new[:, :write_seq, :write_rank] = reproject_U(
                U_in, V_in, gathered[:, :write_rank, :].float()).to(U_eff.dtype)
            U_eff = U_new

        # ── int8 U quant with a per-block scale (matches write_block) ──
        self.U[pidx] = 0
        U_sliced = U_eff[:, :write_seq, :write_rank].float()              # [N, s, r]
        max_abs = U_sliced.abs().amax(dim=(1, 2))                         # [N]
        scale_u = torch.clamp(max_abs / 127.0, min=1e-5).to(self.dtype)   # [N]
        U_q = torch.clamp(
            torch.round(U_sliced / scale_u.float().view(N, 1, 1)), -127, 127
        ).to(torch.int8)
        self.U[pidx, :write_seq, :write_rank] = U_q
        self.U_scale[pidx] = scale_u

        # ── V_K / V_V split ──
        # Skipped entirely under shared bases: the registry already wrote the
        # (orthonormalised, possibly pre-existing) basis into its row, and
        # writing this block's own V there would clobber the basis every other
        # member of the group reads.
        if basis_rows is None:
            Vd = V.to(dev)
            vk = Vd[:, :write_rank, :num_kv * h_dim].reshape(N, write_rank, num_kv, h_dim)
            vv = Vd[:, :write_rank, num_kv * h_dim:].reshape(N, write_rank, num_kv, h_dim)
            self.V_KV[pidx] = 0
            self.V_KV[pidx, 0, :write_rank] = vk.to(self.dtype)
            self.V_KV[pidx, 1, :write_rank] = vv.to(self.dtype)

        self.anchors_KV[pidx, 0] = anchor_K.to(device=dev, dtype=self.dtype)
        self.anchors_KV[pidx, 1] = anchor_V.to(device=dev, dtype=self.dtype)

        # Sanitize non-finite scales exactly like write_block's per-block guard.
        sc = scales.to(device=dev).float()
        sc = torch.where(torch.isfinite(sc), sc, torch.ones_like(sc))
        self.scales[pidx] = sc.to(self.dtype)
        # Same contract as write_block: record what was stored, not what was
        # offered, or slots past capacity become phantom anchor copies.
        _warn_block_truncation(int(seq_len), pool_max_seq)
        self.seq_lens[pidx] = write_seq
        # Record the ACTUAL span of a routing block, host-side and free (seq_len is
        # already a Python int here, so no device read). The router needs it: K is
        # meaningless on its own, the quantity that has to match MLX is the routed
        # TOKEN budget K * span. MLX has a fixed block_size=256 so K=16 always
        # means 4096 tokens; this side blocks ADAPTIVELY (~32-64 tokens for short
        # contexts, ~256 for long — see the avg_block_sz note in
        # kv_runtime_manager), so a K derived from an assumed 257 routes 4x too
        # few tokens whenever the real blocks are 64 wide.
        _span = write_seq
        if _span > 0:
            self.observed_block_span = max(getattr(self, "observed_block_span", 0), _span)
        # One transfer, not one per block: int(pidx[i]) on a device tensor is a
        # device->host sync, and this loop ran it once per block in the batched
        # write -- 1,596 hits in the sync recorder during prefill.
        for _p in pidx.tolist():
            self.version[_p] += 1

        # ── residuals (padded [N, max_res]; slots zeroed first) ──
        self.residual_K_positions[pidx] = -1
        self.residual_K_values[pidx] = 0.0
        self.residual_V_positions[pidx] = -1
        self.residual_V_values[pidx] = 0.0
        if res_K_positions is not None and res_K_positions.numel() > 0:
            mr = min(res_K_positions.shape[1], self.max_residual_tokens)
            self.residual_K_positions[pidx, :mr] = res_K_positions[:, :mr].to(device=dev, dtype=torch.int16)
            self.residual_K_values[pidx, :mr] = res_K_values[:, :mr].to(device=dev, dtype=self.dtype)
            if not self.has_any_residual:
                self.has_any_residual = bool((res_K_positions >= 0).any().item())
        if res_V_positions is not None and res_V_positions.numel() > 0:
            mr = min(res_V_positions.shape[1], self.max_residual_tokens)
            self.residual_V_positions[pidx, :mr] = res_V_positions[:, :mr].to(device=dev, dtype=torch.int16)
            self.residual_V_values[pidx, :mr] = res_V_values[:, :mr].to(device=dev, dtype=self.dtype)

        # ── stratified slots: CLEAR, exactly as write_block does ──────────────
        # write_block zeroes U_sem/U_sem_scale/U_fact and SETS n_semantic on every
        # write; this path did neither, so the docstring's "mirrors write_block
        # field-for-field" was false for these four fields.
        #
        # It is not inert on CUDA. _needs_legacy_slots is HARDCODED True (:121,
        # with the `not (_is_cuda_dev and _gpu_compress)` form commented out
        # directly above it), so these tensors ARE allocated here, and this
        # batched writer is the live path whenever gpu_compress is on.
        #
        # The consequence is only visible on a RECYCLED slot: a fresh slot is
        # zeros from torch.zeros, so the first prompt in a process is clean, and
        # every later one inherits whatever the slot's previous occupant left --
        # a stale n_semantic makes the reconstruction split the block at the
        # wrong point and read U_sem/U_fact bytes belonging to another block.
        # That is exactly the observed signature: first prompt correct, later
        # prompts garbage, and the whole thing disappears under
        # DKV_NO_SLOT_REUSE=1.
        if self.n_semantic is not None:
            self.U_sem[pidx] = 0
            self.U_sem_scale[pidx] = 0.0
            self.U_fact[pidx] = 0.0
            self.n_semantic[pidx] = 0
        if self.fact_anchor_positions is not None:
            self.fact_anchors_K[pidx] = 0.0
            self.fact_anchors_V[pidx] = 0.0
            self.fact_anchor_positions[pidx] = -1

        # ── SRL descriptor (batched compute_descriptor over the written slots) ──
        if self.W_proj is not None:
            try:
                anchor_mean = self.anchors_KV[pidx, 0].float().mean(dim=1)              # [N, hd]
                U_f32 = self.U[pidx, :write_seq, :write_rank].float() \
                    * self.U_scale[pidx].float().view(N, 1, 1)                          # [N, s, r]
                mean_u = U_f32.mean(dim=1)                                              # [N, r]
                _vidx = basis_rows if basis_rows is not None else pidx
                vk_mean = self.V_KV[_vidx, 0, :write_rank].float().mean(dim=2)          # [N, r, hd]
                delta_centroid = torch.bmm(mean_u.unsqueeze(1), vk_mean).squeeze(1)     # [N, hd]
                centroid = anchor_mean + delta_centroid                                 # [N, hd]
                desc = centroid @ self.W_proj.float().t()                               # [N, DESC]
                desc = desc / (desc.norm(dim=1, keepdim=True) + 1e-8)
                self.desc[pidx] = desc.to(torch.float16)
            except Exception:
                pass  # descriptor failure is non-fatal (SRL routing degrades gracefully)

        self._stratified_generation += N

    def reset(self):
        """Completely reset the pool to its initial lightweight state, releasing all grown VRAM."""
        # Free old tensors before re-allocating
        attrs = ("U", "U_scale", "V_KV", "anchors_KV", "scales", "seq_lens", "desc",
                 "residual_K_positions", "residual_K_values", "residual_V_positions", "residual_V_values",
                 "U_sem", "U_sem_scale", "U_fact", "n_semantic",
                 "fact_anchors_K", "fact_anchors_V", "fact_anchor_positions")
        for attr in attrs:
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
        # BASIS GROUPS ARE POOL STATE AND MUST DIE WITH THE POOL.
        #
        # The attrs loop above deletes V_KV, but `basis_store` is a
        # _JointVAdapter holding a reference to it, so the old tensor stays
        # alive behind the adapter while the next allocation builds a NEW V_KV.
        # `basis_registry` and `basis_of` were not cleared either. On the LAZY
        # path -- which is CUDA's default -- reset() does not call
        # _allocate_tensors, so nothing rebuilt them: the registry kept every
        # group it had, its rows indexing a store no reader uses any more, and
        # its capacity already spent. The next document's blocks then found
        # nothing free and were FORCE-JOINED to the previous document's bases.
        #
        # This is the same defect the MLX port hit with the registry on the
        # manager, and it is invisible to any single-session test: prefill state
        # is byte-identical between a passing and a failing configuration, and
        # only a several-requests-in-one-process run reaches it. That is what
        # colab/needle_suite_cuda.py is for.
        self.basis_of = None
        self.basis_registry = None
        self.basis_store = None
        self._basis_claimed = None
        # OPT-D: Bump generation so any proxy cached against the pre-reset data
        # is automatically evicted on the next decode call.
        self._stratified_generation = getattr(self, "_stratified_generation", 0) + 1

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
