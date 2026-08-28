"""
native_core/graph_runtime/static_decode_graph.py

CUDA Graph wrapper for the DKV decode step.

Eliminates Python dispatch overhead, kernel launch scheduling, and driver API
overhead from the decode hot-path. After one-time capture, each decode step
replays the static graph with ~microsecond overhead instead of ~millisecond.

Usage:
    runner = CUDAGraphDecodeRunner()
    # Inside decode loop:
    out = runner.run(fn, static_inputs)

Architecture:
- Warmup runs: 3 forward passes to let torch.compile and Triton JIT warm up.
- Capture: records the full forward into a CUDA graph with static input buffers.
- Replay: copies new inputs into static buffers, calls graph.replay().
- MPS path: CUDA graphs are not available on MPS — transparently falls through
  to normal eager execution (no capture overhead).
"""

import torch
import os
from typing import Dict, Optional


def _is_cuda_available():
    return torch.cuda.is_available()


class CUDAGraphDecodeRunner:
    """
    CUDA Graph wrapper for the DKV batch decode step.

    API (used by batch_engine.py):
        runner = CUDAGraphDecodeRunner()
        runner.capture(model, input_ids, position_ids)   # once per new prefill
        out    = runner.run(input_ids, position_ids)     # every decode step (~2 µs)
        runner.invalidate()                              # on new prefill / shape change
        runner.is_captured() -> bool

    Lifecycle:
        1. After prefill completes, batch_engine calls capture() on the first decode step.
        2. capture() does 3 warmup passes (Triton JIT / compile) then records the graph.
        3. Subsequent steps call run() — inputs are copied into static buffers, graph replays.
        4. invalidate() is called when the pool layout changes (new prefill / session change).

    Shape constraint: input_ids and position_ids must have the same shape every call.
    If shape changes, call invalidate() first.

    MPS / CPU: capture_enabled=False, all calls fall through to None so batch_engine
    uses the eager path.
    """

    def __init__(self):
        self._graph                  = None
        self._static_input_ids       = None   # [B, 1] long — static buffer for graph
        self._static_position_ids    = None   # [B, 1] long — static buffer
        self._static_output_logits   = None   # [B, V] float — static output
        self._captured_shape_sig     = None   # (batch, seq) shape for invalidation
        self._captured_past_kv       = None   # cache object the graph mutates in place
        self._static_cache_position  = None   # [q] long — KV slot to write this step
        self._model_ref              = None   # weak ref to the model (non-owning)
        # CUDA graph capture is DISABLED by default.
        #
        # The DKV attention patch mutates Python/session state on every decode
        # forward (routing slots, dense-window layout, SRL state, session IDs).
        # A captured graph replays without executing any of this Python — the
        # graph becomes stale after the first routing change and silently produces
        # incorrect outputs.  The graph ABI must be redesigned around static,
        # device-resident state buffers before it can be safely re-enabled.
        #
        # `DKV_DISABLE_CUDA_GRAPH=0` remains accepted for compatibility, but
        # it is not sufficient to make a DKV model capturable.  The model
        # must explicitly advertise a static-state ABI (see capture()).
        _disable_graph = os.environ.get("DKV_DISABLE_CUDA_GRAPH", "1")
        self._capture_enabled = _is_cuda_available() and _disable_graph != "1"
        if _is_cuda_available() and not self._capture_enabled:
            import sys as _sys
            print(
                "[DKV] CUDA graph capture DISABLED (mutable routing state "
                "prevents correct replay; the compatibility opt-in is still ABI-guarded).",
                file=_sys.stderr,
            )
        self._num_warmup             = 3
        self._unsafe_capture_warned  = False

    @property
    def capture_enabled(self) -> bool:
        """Whether the environment/device allow capture before model checks."""
        return self._capture_enabled

    def is_captured(self) -> bool:
        """Returns True if a CUDA graph has been captured and is ready for replay."""
        return self._graph is not None and self._capture_enabled

    def _fill_mask(self, position_ids):
        """Causal mask for a decode step, written in place on-device.

        Entry j is 0 where key j is visible to this query and -inf otherwise.
        For a decode step every key up to the current absolute position is
        visible, so the row is a prefix of zeros. Done with a device-side
        comparison -- no host round-trip, so it stays valid to call between
        graph replays.
        """
        _min = torch.finfo(self._static_attn_mask.dtype).min
        cols = torch.arange(self._mask_kv_len, device=position_ids.device)
        # position_ids is [B, q]; allow keys <= that query's absolute position.
        allowed = cols.view(1, 1, 1, -1) <= position_ids.view(
            position_ids.shape[0], 1, position_ids.shape[1], 1)
        self._static_attn_mask.copy_(
            torch.where(allowed,
                        torch.zeros((), device=position_ids.device,
                                    dtype=self._static_attn_mask.dtype),
                        torch.full((), _min, device=position_ids.device,
                                   dtype=self._static_attn_mask.dtype)))

    # ── Decode-state snapshot/restore ─────────────────────────────────────
    # Warmup and capture each run a REAL forward, so between them they advance
    # the decode state by 4 steps for the single token being captured. The
    # wrapper then calls run() for that same token, so the KV write index ends up
    # ahead of position_ids and the position-derived attention mask points at the
    # wrong keys -- the model degenerates into repeating one token while still
    # returning well-formed logits. Capture must therefore be a no-op on state:
    # snapshot before, restore after, so run() performs exactly one write.
    #
    # Two kinds of state matter on a hybrid model like Qwen3.5:
    #   StaticLayer          -- `cumulative_length`, a 0-dim DEVICE tensor that is
    #                           the K/V write index (slot contents past it are
    #                           masked off, so only the counter needs restoring).
    #   LinearAttentionLayer -- `conv_states` / `recurrent_states`, dicts of
    #                           tensors carrying the recurrence. These are not
    #                           reconstructible from a counter; they must be cloned.
    _STATE_DICTS = ("conv_states", "recurrent_states")

    @staticmethod
    def _snapshot_cache(cache):
        snap = []
        for layer in getattr(cache, "layers", []) or []:
            ent = {}
            cl = getattr(layer, "cumulative_length", None)
            if isinstance(cl, torch.Tensor):
                ent["cumulative_length"] = cl.clone()
            elif isinstance(cl, int):
                ent["cumulative_length"] = cl
            for name in CUDAGraphDecodeRunner._STATE_DICTS:
                d = getattr(layer, name, None)
                if isinstance(d, dict):
                    ent[name] = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                 for k, v in d.items()}
            snap.append(ent)
        return snap

    @staticmethod
    def _restore_cache(cache, snap):
        # Restores IN PLACE. The captured graph holds raw pointers to these exact
        # tensors, so rebinding the attribute to a fresh tensor would leave the
        # graph writing into an orphaned buffer.
        for layer, ent in zip(getattr(cache, "layers", []) or [], snap):
            cl = ent.get("cumulative_length")
            if cl is not None:
                cur = getattr(layer, "cumulative_length", None)
                if isinstance(cur, torch.Tensor) and isinstance(cl, torch.Tensor):
                    cur.copy_(cl)
                else:
                    setattr(layer, "cumulative_length", cl)
            for name in CUDAGraphDecodeRunner._STATE_DICTS:
                d = ent.get(name)
                if not isinstance(d, dict):
                    continue
                cur = getattr(layer, name, None)
                if not isinstance(cur, dict):
                    continue
                for k, v in d.items():
                    tgt = cur.get(k)
                    if isinstance(v, torch.Tensor) and isinstance(tgt, torch.Tensor)                             and tgt.shape == v.shape:
                        tgt.copy_(v)
                    else:
                        cur[k] = v

    def capture(self, model, input_ids: torch.Tensor, position_ids: torch.Tensor,
                cache=None):
        """
        Capture a CUDA graph of model(input_ids, position_ids, use_cache=True).

        Runs 3 warmup passes first so Triton JIT and torch.compile have already
        compiled. Then records the graph into a static input/output buffer pair.

        Args:
            model:        The causal LM model (wrapper.model).
            input_ids:    [B, 1] int64 on CUDA — current token ids.
            position_ids: [B, 1] int64 on CUDA — current absolute positions.
            cache:        The decode cache the forward mutates (the DKV bypass
                          StaticCache). REQUIRED — without it capture cannot undo
                          its own warmup writes, so it refuses rather than leave
                          the cache misaligned and produce silently wrong text.
        """
        import time as _t
        _t_cap0 = _t.perf_counter()
        if not self._capture_enabled:
            return

        # The ROUTED path has no dense_cache to snapshot: its decode state is the
        # block pool and the dense window, not a StaticCache. What makes warmup
        # safe there is that the forward DOES NOT MUTATE -- mutation-out moves
        # ingest and window assembly to the wrapper, between forwards -- so the
        # three warmup passes and the capture run write nothing to roll back.
        # Correctness is checked by comparing generated text against the eager
        # path, not assumed.
        #
        # This used to read DKV_GRAPH_SAFE_ROUTING with a default of "0" while
        # dkv_attention reads the SAME variable with a default of "1", and the
        # variable is not in BEST_DECODE_DEFAULTS so it is normally unset. The
        # two modules therefore disagreed: fixed-shape routing was on, and the
        # capture that needs it refused anyway. That mismatch, not any missing
        # machinery, is why the routed graph never captured in the wrapper path.
        #
        # Gating on mutation-out instead is also the CORRECT condition rather
        # than merely a working one. Fixed-shape routing keeps shapes static,
        # which capture needs, but says nothing about whether warmup writes
        # state; mutation-out is exactly the property that makes a cache
        # unnecessary. With it off the forward still ingests, and three warmup
        # ingests of the same token are three appends -- which is what the
        # rollback was protecting against.
        _mutation_out = False
        try:
            import runtime.dkv_attention as _da_mod          # local: avoids a cycle
            _mutation_out = bool(getattr(_da_mod, "_MUTATION_OUT_ACTIVE", False))
        except Exception:                                    # noqa: BLE001
            _mutation_out = False
        _routed_ok = _mutation_out or os.environ.get(
            "DKV_GRAPH_SAFE_ROUTING_CAPTURE", "0") == "1"
        if cache is None and not _routed_ok:
            if not self._unsafe_capture_warned:
                import sys as _sys
                print(
                    "[DKV] CUDA graph capture skipped: no decode cache supplied "
                    "and mutation-out is not active, so capture's warmup writes "
                    "could not be rolled back. On the routed path this means "
                    "--fastdc / DKV_GRAPH_MUTATION_OUT is off, or the gate "
                    "disabled it for this session.",
                    file=_sys.stderr,
                )
                self._unsafe_capture_warned = True
            return

        # ── HOW TO MAKE THE ROUTED PATH CAPTURABLE (scoped 2026-08-16) ───────
        # Read this before attempting it. The remaining work is ONE thing --
        # device-resident routing -- and the two cheaper designs are already
        # eliminated by measurement, so do not re-derive them.
        #
        # RULED OUT, deferred ingest: ingest_streaming() runs BEFORE the attention
        # dispatch, so the dense window supplies the SELF-attention term and
        # deferring drops it silently. Fixable. What is not fixable is the
        # arithmetic: routing is recomputed in Python every step, so replay
        # freezes it and the graph must be re-captured whenever it changes.
        # Capture costs 288 ms (measured); replay saves ~14.7 ms/token; break-even
        # is ~20 tokens of replay per capture, and DKV_REMAT_INTERVAL freezes
        # routing for 4. Re-capture on routing change is a 5x NET LOSS.
        #
        # THE ROUTER IS ALREADY DEVICE-RESIDENT -- scoped and confirmed, so do
        # not spend time rebuilding it. query_router.route_blocks_relevance ends
        # in `sel = torch.topk(relevance, k=k_eff).indices` and returns
        # `block_indices[sel].to(torch.int32)`: device tensors throughout, no
        # .cpu() and no .tolist() on that path. The decode step also has ZERO
        # syncs inside model.forward (colab/decode_sync_probe.py), which is why
        # capture succeeds at all.
        #
        # SO WHAT ACTUALLY BLOCKS REPLAY IS MUTATION, NOT SYNCS OR ROUTING.
        # Replay executes no Python, and four things the forward mutates per
        # token are Python:
        #   1. ingest_streaming()          -- appends K/V to the tail block
        #   2. assemble_dense_window_kv()  -- rebuilds the recent-token window
        #   3. block_indices               -- built from CPU-resident metadata
        #                                     ("Phase 29: metadata is now
        #                                     CPU-resident, zero CUDA syncs on
        #                                     write") and fed INTO the device router
        #   4. block finalise + compression trigger
        #
        # None of these synchronise, which is exactly why the sync probe reports
        # a clean forward while replay is still wrong. Sync-freedom was necessary
        # and is not sufficient.
        #
        # THE DESIGN THAT WORKS, given the router is already device-side:
        #   A. give attend_with_remat an explicit curr_k/curr_v row, so the
        #      current token is attended from IN-GRAPH tensors and no longer
        #      depends on having been ingested into the window first;
        #   B. move ingest_streaming OUT of the forward -- the wrapper ingests
        #      token t after the forward returns, reading K/V from the graph's
        #      fixed-address buffers (references stashed at capture time);
        #   C. move assemble_dense_window_kv out the same way, writing into the
        #      already-preallocated per-layer workspace (fixed addresses);
        #   D. make block_indices a fixed-address buffer the wrapper refreshes
        #      between replays, so the in-graph router reads fresh candidates.
        # Then the graph holds only tensor math and every mutation happens
        # between replays, where Python is allowed to run.
        #
        # A-D must land TOGETHER: with A alone the current token is attended
        # twice (window + curr row); with B alone it is attended zero times.
        #
        # STATUS 2026-08-16: A, B and C are IMPLEMENTED and VERIFIED behind
        # DKV_GRAPH_MUTATION_OUT (default off). With the flag on and graphs off,
        # generated text equals eager EXACTLY (md5 1c58b822b4983e8d), so the
        # forward is genuinely read-only with respect to KV state now: ingest and
        # window assembly happen in the wrapper between forwards, and the current
        # token is attended from an in-graph curr_kv row.
        #
        # ROUTED REPLAY IS STILL WRONG, and the cause is now isolated. With the
        # flag on, capture succeeds but replay gives 026be52c49f9f6c0 against
        # eager's 1c58b822b4983e8d. Turning the remat cache OFF does not fix it
        # (replay 4f31d5fb3ed271aa vs eager 3c5bdad12dd9e50d), which RULES OUT the
        # remat materialisation as the cause and leaves exactly one thing:
        #
        #   block_indices / anchor_indices are produced by
        #   get_cached_decode_blocks INSIDE the forward, in Python. Replay freezes
        #   them at their capture-time values, so routing never updates.
        #
        # THE FIX, and it is smaller than it looks. Routing needs the QUERY to
        # score blocks, and the query is computed inside the forward -- so the
        # wrapper cannot route before the forward. It CAN route after: stash the
        # query the same way K/V are stashed, and between forwards compute the
        # routing for the NEXT step into fixed-address buffers the forward reads.
        # Routing is then ONE STEP STALE, which is strictly less staleness than
        # the DKV_REMAT_INTERVAL=4 freeze the system already ships and accepts.
        #
        # WHAT THE WRAPPER WILL NEED, enumerated from the call site so the next
        # attempt is mechanical rather than exploratory. route_blocks_relevance
        # (dkv_attention ~2357) is called with:
        #     Q              q_for_routing  -- stash per layer, as K/V already are
        #     pool           kv_manager.native_pool
        #     block_indices  from get_cached_decode_blocks(sid, layer, device)
        #     anchor_indices same call
        #     scale          the attention scale for this layer
        #     cos, sin       session_dict["rope_cos"/"rope_sin"], already cached
        #                    per session and grown to max(seq_len+1, pool.U.shape[1])
        #     srl_state      kv_manager.get_srl_state(sid)
        #     layer_idx      the layer
        # Everything except q_for_routing is already reachable from the manager,
        # so the only new plumbing is the query stash.
        #
        # THE TRAP TO AVOID: the forward wraps that call in conditions (router
        # mode, the k_eff engage test, the legacy srl threshold). Calling
        # route_blocks_relevance directly from the wrapper without them will
        # diverge on exactly the configurations those guards exist for. Either
        # replicate the guards or -- better -- extract the whole block into one
        # manager method and have BOTH the forward and the wrapper call it, so
        # they cannot drift apart.
        #
        # Then the same md5-vs-eager gate decides it, followed by the accuracy
        # suite. Do not judge it on speed: replay is already 1.41x and wrong.
        #
        # (Superseded note: A is IMPLEMENTED and inert -- attend_with_remat now accepts
        # curr_kv and dense_mask, both defaulting to None.)
        #
        # EDIT THE RIGHT FORWARD. There are two in dkv_attention.py and only one
        # is live: `dkv_forward` defined at ~1342 INSIDE apply_dkv_attention_patch
        # is what gets bound onto layer.self_attn.forward, and it is the one that
        # calls _remat_attend. `_dkv_decode_forward_impl` (~4742) is a separate
        # integration and is NOT the HF wrapper path -- an earlier revision of
        # this note cited its line numbers by mistake. In the LIVE path the order
        # is ingest (~1957) -> assemble_dense_window_kv (~2573) -> _remat_attend
        # (~3948/3994), which is what makes the window supply the self term.
        #
        # WHY IT IS NEVERTHELESS SAFE TO ATTEMPT: routing output is DISCRETE.
        # A device router can be built alongside the CPU one and checked for
        # EXACT INDEX EQUALITY, which is a far stronger gate than "the text still
        # looks fine" -- and it is the gate that catches the failure mode that
        # matters here, silently attending the wrong blocks.
        #
        # ORDER, and do not reorder it:
        #   1. Build the device router beside the CPU one. Do not wire it in.
        #   2. Assert exact equality of the selected indices, per layer per token,
        #      across the full needle sweep and linkbench 48 seeds. Any mismatch
        #      is a bug in the device router, not a tolerance to widen.
        #   3. Only then switch the read path over, and re-verify with
        #      colab/graph_verify.py -- md5 of generated text must equal eager.
        #   4. Only then allow capture on the routed path.
        #
        # Expected payoff, measured on the current build: 1.41x
        # (16.59 -> 23.38 tok/s wall at 16k). Decode is ~39% GPU-idle, so this is
        # launch overhead, not compute, and the ceiling is real.
        #
        # NEVER measure replay with DKV_TIME_ATTN. Under replay the DKV Python
        # does not run, so it reports a fictional ~1449 tok/s. Use
        # colab/decode_wall_vs_timer.py.
        #
        # A full DKV model is not a static CUDA-graph workload yet.  Its
        # attention interception updates Python/session state (KV blocks,
        # routing slots, dense-window membership and SRL state) on every
        # forward.  CUDA Graph replay records kernels only, so replaying this
        # forward would reuse stale state and also fail to append the token.
        # MLX can compile the analogous path because its @mx.compile functions
        # are pure array functions with all state passed explicitly.  Refuse
        # capture until the CUDA path has the same ABI instead of exposing a
        # switch that silently changes model semantics.
        if not getattr(model, "_dkv_cuda_graph_safe", False):
            if not self._unsafe_capture_warned:
                import sys as _sys
                print(
                    "[DKV] CUDA graph capture skipped: the current stateful "
                    "DKV forward has no static-state graph ABI; using eager "
                    "decode for correctness.",
                    file=_sys.stderr,
                )
                self._unsafe_capture_warned = True
            return

        sig = (tuple(input_ids.shape), tuple(position_ids.shape))
        if sig == self._captured_shape_sig and self._graph is not None:
            return   # already captured for this shape

        self._model_ref = model

        # ── Allocate static input buffers (pinned to a fixed GPU address) ──
        # Allocated BEFORE warmup on purpose: warmup must run through the exact
        # same tensors as the capture. Warming up on the caller's tensors without
        # a cache_position let every warmup pass append another token to the dense
        # KV cache at a drifting slot, so capture recorded a forward over a cache
        # already polluted with 3 junk tokens.
        self._static_input_ids    = input_ids.clone()
        self._static_position_ids = position_ids.clone()

        # ── Static 4D attention mask ──────────────────────────────────────
        # Without this, transformers builds the causal mask itself every forward
        # and masking_utils.eager_mask does
        #     torch.where(mask, tensor(0.0, device=...), min_dtype)
        # where min_dtype is a bare PYTHON FLOAT. Torch wraps that scalar in a
        # CPU tensor and copies it to the GPU, which raises during capture:
        # "Cannot copy between CPU and CUDA tensors during CUDA graph capture
        # unless the CPU tensor is pinned".
        #
        # _preprocess_mask_arguments returns an already-4D mask AS-IS and skips
        # creation entirely, so supplying one avoids that copy. It also fixes the
        # separate "(*bias): last dimension must be contiguous" SDPA raises on
        # StaticCache's sliced mask, because this buffer is contiguous by
        # construction.
        #
        # Contents are rebuilt per step in run(), OUTSIDE the graph -- only the
        # buffer's address has to stay fixed, exactly like input_ids.
        _dtype = next(model.parameters()).dtype
        _kv_len = int(os.environ.get("DKV_GRAPH_MAX_CTX", "8192"))
        _q = int(input_ids.shape[1])
        self._static_attn_mask = torch.zeros(
            (int(input_ids.shape[0]), 1, _q, _kv_len),
            device=input_ids.device, dtype=_dtype)
        self._mask_kv_len = _kv_len
        self._fill_mask(position_ids)

        # ── Static cache_position ─────────────────────────────────────────
        # THE correctness blocker for replay. If cache_position is not passed,
        # model.forward builds it as
        #     torch.arange(past_key_values.get_seq_length(), ...)
        # and get_seq_length() is a HOST int, so capture bakes the write index in
        # as a constant. Every replay then index_copy_'s the new token's K/V into
        # the SAME slot and attends over a prefix that never grows -- the model
        # degenerates into repeating a token ("susersusers...") while still
        # returning well-formed logits, so nothing raises.
        #
        # Passing it makes the write index a device tensor read at replay time.
        # On the bypass path the dense StaticCache holds the prefix densely, so
        # slot index == absolute position and position_ids can drive it directly.
        self._static_cache_position = position_ids.reshape(-1).clone()

        # ── Hybrid (linear-attention) models cannot be captured ───────────
        # Qwen3.5 interleaves LinearAttentionLayer with full attention. Those
        # layers REBIND their conv/recurrent state every forward
        # (recurrent_states[i] = new_tensor) instead of writing in place --
        # measured: 0/36 state tensors kept their address across a decode step.
        # A replay executes no Python, so the dict never gets reassigned and the
        # graph keeps reading and writing the addresses that were live at
        # capture: the recurrence freezes while the full-attention layers keep
        # advancing, and the model emits fluent-looking garbage with nothing
        # raised. Refusing is the only correct option until those layers update
        # in place, which is transformers' modeling code, not ours.
        _recurrent = [type(L).__name__ for L in (getattr(cache, "layers", None) or [])
                      if any(isinstance(getattr(L, _n, None), dict)
                             for _n in self._STATE_DICTS)]
        if _recurrent:
            if not self._unsafe_capture_warned:
                import sys as _sys
                print(
                    f"[DKV] CUDA graph capture skipped: hybrid model has "
                    f"{len(_recurrent)} recurrent layers ({_recurrent[0]}) whose "
                    f"state rebinds each step and cannot be replayed; using eager "
                    f"decode for correctness.",
                    file=_sys.stderr,
                )
                self._unsafe_capture_warned = True
            return

        _snap = self._snapshot_cache(cache) if cache is not None else None

        # ── Warmup ────────────────────────────────────────────────────────
        # Same tensors, same kwargs as the capture below. Because every pass
        # writes K/V at the identical cache_position, warmup is idempotent: the
        # cache ends holding exactly one token at that slot, which is precisely
        # the state one real decode step would have produced. The first replay
        # then overwrites it correctly.
        with torch.no_grad():
            for _ in range(self._num_warmup):
                model(input_ids=self._static_input_ids,
                      position_ids=self._static_position_ids,
                      attention_mask=self._static_attn_mask,
                      cache_position=self._static_cache_position,
                      use_cache=True)
        torch.cuda.synchronize()

        # ── Capture ────────────────────────────────────────────────────────
        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(self._graph):
                _captured_out = model(
                    input_ids=self._static_input_ids,
                    position_ids=self._static_position_ids,
                    attention_mask=self._static_attn_mask,
                    cache_position=self._static_cache_position,
                    use_cache=True,
                )
                # Capture only the last-token logits slice — that's all we need
                self._static_output_logits = _captured_out.logits[:, -1:, :]
                # The graph writes K/V into THIS cache object every replay, so the
                # caller's `past_kv = outputs.past_key_values` must keep pointing at
                # it. Without this the decode loop sets past_kv=None on the first
                # replayed step and the next eager fallback would re-prefill.
                self._captured_past_kv = _captured_out.past_key_values

        # Roll the decode state back to exactly what it was on entry, so the
        # caller's following run() is this token's one and only step.
        if _snap is not None:
            self._restore_cache(cache, _snap)

        self._captured_shape_sig = sig
        self._last_capture_seconds = _t.perf_counter() - _t_cap0
        # POSITIVE confirmation. Capture is wrapped in `except Exception: pass`
        # by the caller, so a failure is silent and indistinguishable from eager
        # decode -- which means a benchmark can appear to measure graph replay
        # while measuring nothing of the sort. Say so explicitly.
        import sys as _sys
        print(f"[DKV] CUDA graph CAPTURED for shape {sig} in "
              f"{self._last_capture_seconds * 1000:.0f} ms — replay active",
              file=_sys.stderr, flush=True)

    def run(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        """
        Replay the captured CUDA graph with new inputs.

        Copies input_ids and position_ids into the static buffers, replays,
        then returns an object with .logits[:, -1, :] accessible.

        Raises RuntimeError if not yet captured — caller should check is_captured() first.
        """
        if not self.is_captured():
            raise RuntimeError("CUDAGraphDecodeRunner.run() called before capture()")

        # In-place copy new tokens into static buffers (no allocation)
        self._static_input_ids.copy_(input_ids, non_blocking=True)
        self._static_position_ids.copy_(position_ids, non_blocking=True)
        # Rebuild the causal row for THIS step. Outside the graph, so it may use
        # ordinary ops; only the buffer address is what replay depends on.
        self._fill_mask(position_ids)
        self._static_cache_position.copy_(position_ids.reshape(-1), non_blocking=True)

        self._graph.replay()

        # Return a lightweight wrapper so batch_engine can do out.logits[:, -1, :]
        return _GraphOutput(self._static_output_logits, self._captured_past_kv)

    def invalidate(self):
        """Reset — next run() will re-capture (called when pool layout changes)."""
        self._graph                 = None
        self._static_input_ids      = None
        self._static_position_ids   = None
        self._static_attn_mask      = None
        self._static_output_logits  = None
        self._captured_shape_sig    = None
        self._captured_past_kv      = None
        self._static_cache_position = None


class _GraphOutput:
    """
    Thin wrapper returned by CUDAGraphDecodeRunner.run() so that
    batch_engine can access out.logits[:, -1, :] without modification.

    _static_output_logits already has shape [B, 1, V] (last token only),
    so out.logits[:, -1, :] correctly gives [B, V].
    """
    __slots__ = ("logits", "past_key_values")

    def __init__(self, static_logits: torch.Tensor, past_key_values=None):
        # static_logits: [B, 1, V] — the captured output slice, so [:, -1, :]
        # gives [B, V] exactly as it would on a real CausalLMOutput.
        self.logits = static_logits
        # Same cache object the capture ran against; the graph mutates it in place.
        self.past_key_values = past_key_values



class StaticSparseDecodeGraph:
    """
    Legacy class — kept for backward compatibility.
    New code should use CUDAGraphDecodeRunner.
    """
    def __init__(self, decode_fn, max_batch_size: int, head_dim: int, device: str = "cuda"):
        self.decode_fn = decode_fn
        self.device = device

        # Static input buffers for graph replay
        self.static_q = torch.zeros((max_batch_size, 32, head_dim), dtype=torch.float16, device=device)
        self.static_session_ids = torch.zeros((max_batch_size,), dtype=torch.int32, device=device)
        self.static_out = torch.zeros((max_batch_size, 32, head_dim), dtype=torch.float16, device=device)

        self.graph = None
        self.is_captured = False

    def capture(self, q: torch.Tensor, session_ids: torch.Tensor):
        """Captures the Triton Sparse Decode kernel execution."""
        self.static_q[:q.size(0)].copy_(q)
        self.static_session_ids[:session_ids.size(0)].copy_(session_ids)

        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            out = self.decode_fn(self.static_q, self.static_session_ids)
            self.static_out.copy_(out)

        self.is_captured = True

    def replay(self, q: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
        """Replays the static graph with new queries."""
        if not self.is_captured:
            self.capture(q, session_ids)

        bsz = q.size(0)
        self.static_q[:bsz].copy_(q)
        self.static_session_ids[:bsz].copy_(session_ids)

        self.graph.replay()

        return self.static_out[:bsz].clone()
