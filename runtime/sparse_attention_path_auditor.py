"""
STAGE 2 — SAT: Sparse Attention Path Auditor
Phase 38.9 — Sparse Attention Transition

Hooks into the actual transformer attention computation to record:
  - sparse attention invocations (real sparse path taken)
  - dense attention invocations (full-context dense path taken)
  - sparse bypass successes / failures
  - fallback frequency per layer
  - per-layer execution mode distribution

All counters are derived from real execution events.
NO guessed percentages or synthetic values.
"""

import time
import json
import threading
import os
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any


TRACE_PATH = "traces/stage2/phase_38_9_sat/sparse_attention_trace.jsonl"


@dataclass
class LayerAttentionRecord:
    layer_idx: int
    sparse_hits: int = 0
    dense_fallbacks: int = 0
    bypass_successes: int = 0
    bypass_failures: int = 0
    last_mode: str = "unknown"
    last_ts: float = field(default_factory=time.time)
    total_duration_ms: float = 0.0
    invocation_count: int = 0


class SparseAttentionPathAuditor:
    """
    STAGE 2 SAT: Sparse Attention Path Auditor.

    Instruments transformer attention paths at the layer granularity.
    Must be wired into the actual forward pass — not called synthetically.

    Usage:
        auditor = SparseAttentionPathAuditor()
        with auditor.audit_attention(layer_idx=3) as ctx:
            ctx.set_mode("sparse")   # or "dense" / "hybrid"
            # ... real attention computation here ...

    Or as a lightweight callback:
        auditor.record_attention_event(layer_idx, mode, duration_ms, bypass_ok)
    """

    def __init__(self, trace_path: str = TRACE_PATH):
        self.trace_path = trace_path
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)

        self._lock = threading.Lock()
        self._layers: Dict[int, LayerAttentionRecord] = defaultdict(
            lambda: LayerAttentionRecord(layer_idx=-1)
        )

        # Global counters (derived from per-layer records — never set directly)
        self._global_sparse = 0
        self._global_dense = 0
        self._global_bypass_ok = 0
        self._global_bypass_fail = 0

        self._session_start = time.time()
        self._trace_buf: List[Dict] = []
        self._flush_every = 32  # flush trace buffer every N events

    # ------------------------------------------------------------------
    # Primary recording API
    # ------------------------------------------------------------------

    def record_attention_event(
        self,
        layer_idx: int,
        mode: str,          # "sparse" | "dense" | "hybrid"
        duration_ms: float,
        bypass_ok: bool,
        token_count: int = 0,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """
        Record a single attention event from real execution.

        Args:
            layer_idx:       transformer layer index (0-based)
            mode:            execution mode chosen for this invocation
            duration_ms:     wall-clock duration of the attention step
            bypass_ok:       True if sparse bypass was attempted AND succeeded
            token_count:     number of tokens processed in this call
            fallback_reason: if mode=="dense", optional reason string
        """
        ts = time.time()
        with self._lock:
            rec = self._layers[layer_idx]
            rec.layer_idx = layer_idx
            rec.last_mode = mode
            rec.last_ts = ts
            rec.invocation_count += 1
            rec.total_duration_ms += duration_ms

            if mode == "sparse":
                rec.sparse_hits += 1
                self._global_sparse += 1
            else:
                rec.dense_fallbacks += 1
                self._global_dense += 1

            if bypass_ok:
                rec.bypass_successes += 1
                self._global_bypass_ok += 1
            else:
                rec.bypass_failures += 1
                self._global_bypass_fail += 1

            event = {
                "ts": ts,
                "layer": layer_idx,
                "mode": mode,
                "duration_ms": round(duration_ms, 4),
                "bypass_ok": bypass_ok,
                "token_count": token_count,
                "fallback_reason": fallback_reason,
            }
            self._trace_buf.append(event)

            if len(self._trace_buf) >= self._flush_every:
                self._flush()

    # ------------------------------------------------------------------
    # Context-manager API for wrapping real attention blocks
    # ------------------------------------------------------------------

    def audit_attention(self, layer_idx: int):
        return _AttentionAuditCtx(self, layer_idx)

    # ------------------------------------------------------------------
    # Metrics queries
    # ------------------------------------------------------------------

    def get_global_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._global_sparse + self._global_dense
            sparse_rate = self._global_sparse / max(total, 1)
            bypass_total = self._global_bypass_ok + self._global_bypass_fail
            bypass_rate = self._global_bypass_ok / max(bypass_total, 1)
            return {
                "sparse_invocations": self._global_sparse,
                "dense_invocations": self._global_dense,
                "total_invocations": total,
                "sparse_rate": round(sparse_rate, 4),
                "bypass_success_rate": round(bypass_rate, 4),
                "bypass_successes": self._global_bypass_ok,
                "bypass_failures": self._global_bypass_fail,
                "elapsed_sec": round(time.time() - self._session_start, 2),
            }

    def get_layer_distribution(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for idx in sorted(self._layers.keys()):
                rec = self._layers[idx]
                total = rec.sparse_hits + rec.dense_fallbacks
                rows.append({
                    "layer": idx,
                    "sparse_hits": rec.sparse_hits,
                    "dense_fallbacks": rec.dense_fallbacks,
                    "sparse_rate": round(rec.sparse_hits / max(total, 1), 4),
                    "bypass_rate": round(
                        rec.bypass_successes / max(rec.bypass_successes + rec.bypass_failures, 1), 4
                    ),
                    "avg_duration_ms": round(
                        rec.total_duration_ms / max(rec.invocation_count, 1), 4
                    ),
                    "last_mode": rec.last_mode,
                    "invocations": rec.invocation_count,
                })
            return rows

    def get_fallback_frequency(self) -> float:
        """Dense fallbacks as a fraction of total invocations."""
        with self._lock:
            total = self._global_sparse + self._global_dense
            return self._global_dense / max(total, 1)

    def flush_and_close(self) -> None:
        with self._lock:
            self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Write buffered trace events to disk (must be called under self._lock)."""
        if not self._trace_buf:
            return
        with open(self.trace_path, "a") as f:
            for ev in self._trace_buf:
                f.write(json.dumps(ev) + "\n")
        self._trace_buf.clear()


class _AttentionAuditCtx:
    """Context manager returned by SparseAttentionPathAuditor.audit_attention()."""

    def __init__(self, auditor: SparseAttentionPathAuditor, layer_idx: int):
        self._auditor = auditor
        self._layer = layer_idx
        self._mode = "dense"  # default; caller must call set_mode()
        self._bypass_ok = False
        self._fallback_reason: Optional[str] = None
        self._token_count = 0
        self._t0: float = 0.0

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_bypass(self, ok: bool) -> None:
        self._bypass_ok = ok

    def set_token_count(self, n: int) -> None:
        self._token_count = n

    def set_fallback_reason(self, reason: str) -> None:
        self._fallback_reason = reason

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._t0) * 1000.0
        self._auditor.record_attention_event(
            layer_idx=self._layer,
            mode=self._mode,
            duration_ms=duration_ms,
            bypass_ok=self._bypass_ok,
            token_count=self._token_count,
            fallback_reason=self._fallback_reason,
        )
        return False  # do not suppress exceptions
