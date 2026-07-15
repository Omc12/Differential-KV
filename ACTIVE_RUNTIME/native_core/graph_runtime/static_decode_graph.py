"""
native_core/graph_runtime/static_decode_graph.py

CUDA Graph wrapper for the DiffKV decode step.

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
    CUDA Graph wrapper for the DiffKV batch decode step.

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
        self._model_ref              = None   # weak ref to the model (non-owning)
        # CUDA graph capture is DISABLED by default.
        #
        # The DiffKV attention patch mutates Python/session state on every decode
        # forward (routing slots, dense-window layout, SRL state, session IDs).
        # A captured graph replays without executing any of this Python — the
        # graph becomes stale after the first routing change and silently produces
        # incorrect outputs.  The graph ABI must be redesigned around static,
        # device-resident state buffers before it can be safely re-enabled.
        #
        # To opt-in for testing: DIFFKV_DISABLE_CUDA_GRAPH=0
        # Production code should leave this at the default until the graph ABI
        # redesign is complete and validated (see CUDA_VS_MLX_PERFORMANCE_AUDIT).
        _disable_graph = os.environ.get("DIFFKV_DISABLE_CUDA_GRAPH", "1")
        self._capture_enabled = _is_cuda_available() and _disable_graph != "1"
        if _is_cuda_available() and not self._capture_enabled:
            import sys as _sys
            print(
                "[DiffKV] CUDA graph capture DISABLED (default — mutable routing state "
                "prevents correct replay). Set DIFFKV_DISABLE_CUDA_GRAPH=0 to opt-in.",
                file=_sys.stderr,
            )
        self._num_warmup             = 3

    def is_captured(self) -> bool:
        """Returns True if a CUDA graph has been captured and is ready for replay."""
        return self._graph is not None and self._capture_enabled

    def capture(self, model, input_ids: torch.Tensor, position_ids: torch.Tensor):
        """
        Capture a CUDA graph of model(input_ids, position_ids, use_cache=True).

        Runs 3 warmup passes first so Triton JIT and torch.compile have already
        compiled. Then records the graph into a static input/output buffer pair.

        Args:
            model:        The causal LM model (wrapper.model).
            input_ids:    [B, 1] int64 on CUDA — current token ids.
            position_ids: [B, 1] int64 on CUDA — current absolute positions.
        """
        if not self._capture_enabled:
            return

        sig = (tuple(input_ids.shape), tuple(position_ids.shape))
        if sig == self._captured_shape_sig and self._graph is not None:
            return   # already captured for this shape

        self._model_ref = model

        # ── Warmup ────────────────────────────────────────────────────────
        with torch.no_grad():
            for _ in range(self._num_warmup):
                _out = model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
        torch.cuda.synchronize()

        # ── Allocate static input buffers (pinned to a fixed GPU address) ──
        self._static_input_ids    = input_ids.clone()
        self._static_position_ids = position_ids.clone()

        # ── Capture ────────────────────────────────────────────────────────
        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(self._graph):
                _captured_out = model(
                    input_ids=self._static_input_ids,
                    position_ids=self._static_position_ids,
                    use_cache=True,
                )
                # Capture only the last-token logits slice — that's all we need
                self._static_output_logits = _captured_out.logits[:, -1:, :]

        self._captured_shape_sig = sig

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

        self._graph.replay()

        # Return a lightweight wrapper so batch_engine can do out.logits[:, -1, :]
        return _GraphOutput(self._static_output_logits)

    def invalidate(self):
        """Reset — next run() will re-capture (called when pool layout changes)."""
        self._graph                 = None
        self._static_input_ids      = None
        self._static_position_ids   = None
        self._static_output_logits  = None
        self._captured_shape_sig    = None


class _GraphOutput:
    """
    Thin wrapper returned by CUDAGraphDecodeRunner.run() so that
    batch_engine can access out.logits[:, -1, :] without modification.

    _static_output_logits already has shape [B, 1, V] (last token only),
    so out.logits[:, -1, :] correctly gives [B, V].
    """
    __slots__ = ("logits",)

    def __init__(self, static_logits: torch.Tensor):
        # static_logits: [B, 1, V] — the captured output slice
        # Expand to [B, 2, V] so [:, -1, :] gives the right slice
        # (the static_logits tensor IS the last-token slice)
        self.logits = static_logits  # [B, 1, V] — index [:, -1, :] gives [B, V]



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
