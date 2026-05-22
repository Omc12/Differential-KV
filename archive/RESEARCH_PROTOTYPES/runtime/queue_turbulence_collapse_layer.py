"""
RCO-N Phase 41.1: Queue Turbulence Collapse Layer.

Collapses:
- queue rebuild storms
- reconnect fragmentation
- cancellation fragmentation
- stream synchronization turbulence

Uses persistent queue structures and reduced churn
to eliminate scheduler contention overhead.
"""

import time
import json
import threading
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StreamToken:
    """A single token delivery event in the stable stream."""
    session_id: str
    token_text: str
    token_idx: int
    ts: float = field(default_factory=time.perf_counter)
    is_final: bool = False


class StableStreamBuffer:
    """
    A persistent, non-rebuilding token stream buffer per session.
    Tokens accumulate here and are drained by the consumer at its own pace,
    preventing synchronization turbulence from stream rebuilds.
    """
    def __init__(self, session_id: str, max_backlog: int = 512):
        self.session_id = session_id
        self._buffer: Deque[StreamToken] = deque(maxlen=max_backlog)
        self._lock = threading.Lock()
        self._consumer_event = asyncio.Event() if asyncio.get_event_loop().is_running() else None
        self._total_produced = 0
        self._total_consumed = 0
        self._max_backlog_hit = 0

    def produce(self, token: StreamToken):
        with self._lock:
            old_len = len(self._buffer)
            self._buffer.append(token)
            if len(self._buffer) == old_len and old_len == self._buffer.maxlen:
                self._max_backlog_hit += 1
            self._total_produced += 1

    def consume_all(self) -> List[StreamToken]:
        with self._lock:
            if not self._buffer:
                return []
            items = list(self._buffer)
            self._buffer.clear()
            self._total_consumed += len(items)
            return items

    def peek_count(self) -> int:
        return len(self._buffer)

    @property
    def backpressure_ratio(self) -> float:
        if self._total_produced == 0:
            return 0.0
        return min(1.0, len(self._buffer) / max(self._buffer.maxlen, 1))


class QueueTurbulenceCollapseLayer:
    """
    RCO-N Phase 41.1: Collapses queue turbulence through persistent structures
    and coalesced reconnect/cancellation handling.

    Key behaviors:
    1. Queue structures are NEVER fully rebuilt — only incrementally updated
    2. Reconnects are deduplicated within a coalesce window
    3. Cancellations are batched and processed together
    4. Stream sync events are merged per-session
    5. Persistent stream buffers eliminate per-token synchronization cost
    """

    RECONNECT_COALESCE_MS  = 50    # Deduplicate reconnects within 50ms
    CANCEL_BATCH_SIZE      = 8     # Process cancellations in batches of 8
    SYNC_COALESCE_MS       = 10    # Coalesce stream syncs within 10ms
    MAX_STREAM_BACKLOG     = 256   # Max tokens buffered per session

    def __init__(self, trace_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._logger = logging.getLogger("RCO_QueueTurbulenceCollapse")

        # Persistent stream buffers (one per session, never rebuilt)
        self._stream_buffers: Dict[str, StableStreamBuffer] = {}

        # Reconnect coalescing
        self._pending_reconnects: Dict[str, float] = {}   # session_id -> ts of first pending
        self._reconnect_count = 0
        self._reconnect_coalesced = 0

        # Cancellation batching
        self._pending_cancellations: Deque[str] = deque()  # request_ids
        self._cancellation_batches_processed = 0
        self._cancellations_batched = 0

        # Queue health tracking
        self._queue_depth_history: Deque[int] = deque(maxlen=200)
        self._queue_rebuilds_prevented = 0
        self._turbulence_events: Deque[float] = deque(maxlen=100)  # timestamps

        # Stream sync coalescing
        self._sync_pending: Dict[str, float] = {}   # session_id -> last sync request ts
        self._sync_coalesced = 0
        self._sync_executed = 0

        self._trace_path = (
            Path(trace_dir) / "queue_turbulence_collapse_trace.jsonl"
            if trace_dir else None
        )
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger.info(
            "QueueTurbulenceCollapseLayer initialized | "
            "reconnect_coalesce=%.0fms | cancel_batch=%d",
            self.RECONNECT_COALESCE_MS, self.CANCEL_BATCH_SIZE
        )

    # -----------------------------------------------------------------------
    # Persistent stream buffer management
    # -----------------------------------------------------------------------

    def get_stream_buffer(self, session_id: str) -> StableStreamBuffer:
        """Get or create a persistent stream buffer for a session."""
        with self._lock:
            if session_id not in self._stream_buffers:
                self._stream_buffers[session_id] = StableStreamBuffer(
                    session_id, self.MAX_STREAM_BACKLOG
                )
            return self._stream_buffers[session_id]

    def push_token(self, session_id: str, token_text: str, token_idx: int, is_final: bool = False):
        """Push a token to the session's persistent stream buffer."""
        buf = self.get_stream_buffer(session_id)
        buf.produce(StreamToken(
            session_id=session_id,
            token_text=token_text,
            token_idx=token_idx,
            is_final=is_final,
        ))

    def drain_session_tokens(self, session_id: str) -> List[StreamToken]:
        """Drain all buffered tokens for a session."""
        with self._lock:
            buf = self._stream_buffers.get(session_id)
        if buf:
            return buf.consume_all()
        return []

    def remove_session(self, session_id: str):
        """Remove a session's persistent buffer on completion."""
        with self._lock:
            self._stream_buffers.pop(session_id, None)

    # -----------------------------------------------------------------------
    # Reconnect coalescing
    # -----------------------------------------------------------------------

    def request_reconnect(self, session_id: str) -> bool:
        """
        Request a session reconnect. Returns True if this should be processed now,
        False if it was coalesced into an existing pending reconnect.
        """
        now = time.perf_counter()
        with self._lock:
            existing_ts = self._pending_reconnects.get(session_id)
            if existing_ts is not None:
                age_ms = (now - existing_ts) * 1000
                if age_ms < self.RECONNECT_COALESCE_MS:
                    self._reconnect_coalesced += 1
                    return False   # Coalesced — skip this reconnect

            self._pending_reconnects[session_id] = now
            self._reconnect_count += 1
            return True   # Should process

    def complete_reconnect(self, session_id: str):
        """Mark a reconnect as completed."""
        with self._lock:
            self._pending_reconnects.pop(session_id, None)

    # -----------------------------------------------------------------------
    # Cancellation batching
    # -----------------------------------------------------------------------

    def queue_cancellation(self, request_id: str):
        """Add a cancellation request to the batch queue."""
        with self._lock:
            self._pending_cancellations.append(request_id)

    def drain_cancellations(self) -> List[str]:
        """
        Drain up to CANCEL_BATCH_SIZE pending cancellations in one pass.
        Returns the list of request_ids to cancel.
        """
        with self._lock:
            batch = []
            for _ in range(self.CANCEL_BATCH_SIZE):
                if not self._pending_cancellations:
                    break
                batch.append(self._pending_cancellations.popleft())
            if len(batch) > 1:
                self._cancellations_batched += len(batch)
                self._cancellation_batches_processed += 1
            return batch

    # -----------------------------------------------------------------------
    # Stream sync coalescing
    # -----------------------------------------------------------------------

    def request_stream_sync(self, session_id: str) -> bool:
        """
        Request a stream synchronization event.
        Returns True if sync should fire now, False if coalesced.
        """
        now = time.perf_counter()
        with self._lock:
            last_sync = self._sync_pending.get(session_id)
            if last_sync is not None:
                age_ms = (now - last_sync) * 1000
                if age_ms < self.SYNC_COALESCE_MS:
                    self._sync_coalesced += 1
                    return False

            self._sync_pending[session_id] = now
            self._sync_executed += 1
            return True

    # -----------------------------------------------------------------------
    # Queue depth tracking
    # -----------------------------------------------------------------------

    def record_queue_depth(self, depth: int):
        """Record current queue depth for turbulence analysis."""
        with self._lock:
            self._queue_depth_history.append(depth)

    def get_turbulence_score(self) -> float:
        """
        Compute a queue turbulence score (0=stable, 1=chaotic).
        High turbulence = high variance in queue depth.
        """
        with self._lock:
            depths = list(self._queue_depth_history)

        if len(depths) < 4:
            return 0.0
        mean = sum(depths) / len(depths)
        if mean == 0:
            return 0.0
        variance = sum((d - mean) ** 2 for d in depths) / len(depths)
        # Normalize: variance > mean^2 = fully turbulent
        return min(1.0, variance / max(mean ** 2, 1))

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_collapse_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_buffers = len(self._stream_buffers)
            total_backlog = sum(b.peek_count() for b in self._stream_buffers.values())
            avg_backpressure = (
                sum(b.backpressure_ratio for b in self._stream_buffers.values()) / total_buffers
                if total_buffers > 0 else 0.0
            )
        turbulence = self.get_turbulence_score()
        return {
            "active_stream_buffers": total_buffers,
            "total_token_backlog": total_backlog,
            "avg_backpressure": round(avg_backpressure, 3),
            "queue_turbulence_score": round(turbulence, 3),
            "reconnect_count": self._reconnect_count,
            "reconnects_coalesced": self._reconnect_coalesced,
            "cancellations_batched": self._cancellations_batched,
            "sync_executed": self._sync_executed,
            "sync_coalesced": self._sync_coalesced,
        }

    def format_live_line(self) -> str:
        s = self.get_collapse_stats()
        return (
            f"[QUEUE_COLLAPSE] buffers={s['active_stream_buffers']} "
            f"backlog={s['total_token_backlog']} "
            f"turbulence={s['queue_turbulence_score']:.2f} "
            f"reconnect_coalesced={s['reconnects_coalesced']} "
            f"cancel_batched={s['cancellations_batched']} "
            f"sync_coalesced={s['sync_coalesced']}"
        )

    def emit_trace(self):
        if not self._trace_path:
            return
        record = {"timestamp": time.time(), **self.get_collapse_stats()}
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
