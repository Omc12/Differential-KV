"""
RCO-N Phase 41.1: Persistent Decode Batch Engine.

Maintains persistent decode batches across token generations.
Prevents batch rebuild storms and GPU starvation gaps.

GPU kernels should remain continuously fed.
"""

import time
import asyncio
import threading
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DecodeSlot:
    """A single decode slot within the persistent batch."""
    session_id: str
    request_id: str
    tokens_generated: int = 0
    max_tokens: int = 128
    finished: bool = False
    added_ts: float = field(default_factory=time.perf_counter)
    last_token_ts: float = field(default_factory=time.perf_counter)
    output_queue: Optional[asyncio.Queue] = None
    priority: int = 0  # Higher = higher priority

    def is_expired(self, now: float, max_idle_sec: float = 30.0) -> bool:
        return (now - self.last_token_ts) > max_idle_sec


class PersistentDecodeBatchEngine:
    """
    RCO-N Phase 41.1: Maintains persistent decode batches across token generations.

    Key behaviors:
    1. NEVER rebuilds the entire batch from scratch — slots are updated in-place
    2. Maintains a STABLE slot array — completed sessions vacate slots that
       are immediately filled from the admission queue
    3. Admission is performed during inter-batch windows (not mid-decode)
    4. GPU starvation gaps are measured and reported
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        admission_check_interval_ms: float = 5.0,
        trace_dir: Optional[Path] = None,
    ):
        self._lock = threading.Lock()
        self._logger = logging.getLogger("RCO_PersistentBatchEngine")
        self._max_batch_size = max_batch_size
        self._admission_interval = admission_check_interval_ms / 1000.0

        # Persistent slot array — pre-allocated, filled and drained
        self._slots: Dict[str, DecodeSlot] = {}          # session_id -> slot
        self._admission_queue: Deque[DecodeSlot] = deque()  # Pending admission

        # Batch rebuild tracking
        self._total_batch_steps = 0
        self._batch_rebuilds = 0      # How many times we rebuilt from scratch (BAD)
        self._slot_fills = 0          # How many times we filled a vacated slot (GOOD)
        self._gpu_starvation_gaps: List[float] = []      # Gap durations in ms
        self._last_decode_ts = time.perf_counter()
        self._starvation_threshold_ms = 2.0  # Gap > 2ms counts as starvation

        # Continuity tracking
        self._continuous_steps = 0     # Steps with batch_size > 0
        self._empty_steps = 0          # Steps with batch_size == 0
        self._peak_batch_size = 0

        # Trace
        self._trace_path = Path(trace_dir) / "persistent_batch_trace.jsonl" if trace_dir else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger.info(
            "PersistentDecodeBatchEngine initialized | max_batch=%d | admission_check=%.1fms",
            max_batch_size, admission_check_interval_ms
        )

    # -----------------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------------

    def admit_session(
        self,
        session_id: str,
        request_id: str,
        max_tokens: int = 128,
        output_queue: Optional[asyncio.Queue] = None,
        priority: int = 0,
    ):
        """Enqueue a new session for admission to the persistent batch."""
        slot = DecodeSlot(
            session_id=session_id,
            request_id=request_id,
            max_tokens=max_tokens,
            output_queue=output_queue,
            priority=priority,
        )
        with self._lock:
            self._admission_queue.append(slot)

    def _fill_vacant_slots(self):
        """Fill vacant batch slots from admission queue. Call between decode steps."""
        with self._lock:
            while self._admission_queue and len(self._slots) < self._max_batch_size:
                slot = self._admission_queue.popleft()
                self._slots[slot.session_id] = slot
                self._slot_fills += 1

    def complete_session(self, session_id: str):
        """Mark a session as finished. Slot will be vacated on next admission check."""
        with self._lock:
            if session_id in self._slots:
                self._slots[session_id].finished = True

    def _evict_finished_slots(self):
        """Remove finished slots to make room for new admissions."""
        with self._lock:
            finished = [sid for sid, slot in self._slots.items() if slot.finished]
            for sid in finished:
                del self._slots[sid]

    # -----------------------------------------------------------------------
    # Batch preparation (hot path)
    # -----------------------------------------------------------------------

    def prepare_batch(self) -> List[DecodeSlot]:
        """
        Prepare the current decode batch.
        This is the hot path — it must be extremely cheap.
        Returns the list of active slots WITHOUT locking for the duration of decode.
        """
        now = time.perf_counter()

        # Measure GPU starvation gap
        gap_ms = (now - self._last_decode_ts) * 1000
        if gap_ms > self._starvation_threshold_ms and self._total_batch_steps > 0:
            self._gpu_starvation_gaps.append(gap_ms)
            if len(self._gpu_starvation_gaps) > 500:
                self._gpu_starvation_gaps = self._gpu_starvation_gaps[-500:]

        # Evict finished, fill vacated
        self._evict_finished_slots()
        self._fill_vacant_slots()

        with self._lock:
            active_slots = [s for s in self._slots.values() if not s.finished]
            batch_size = len(active_slots)
            if batch_size > self._peak_batch_size:
                self._peak_batch_size = batch_size

            if batch_size > 0:
                self._continuous_steps += 1
            else:
                self._empty_steps += 1

            self._total_batch_steps += 1

        return active_slots

    def mark_batch_complete(self, slots: List[DecodeSlot], tokens_per_slot: Dict[str, int]):
        """Update slot state after a decode step completes."""
        now = time.perf_counter()
        self._last_decode_ts = now

        with self._lock:
            for slot in slots:
                if slot.session_id not in self._slots:
                    continue
                tok = tokens_per_slot.get(slot.session_id, 1)
                slot.tokens_generated += tok
                slot.last_token_ts = now
                if slot.tokens_generated >= slot.max_tokens:
                    slot.finished = True

    # -----------------------------------------------------------------------
    # Batch health monitoring
    # -----------------------------------------------------------------------

    def get_batch_stats(self) -> Dict[str, Any]:
        with self._lock:
            active = sum(1 for s in self._slots.values() if not s.finished)
            pending = len(self._admission_queue)

        total_steps = max(self._total_batch_steps, 1)
        continuity = round(self._continuous_steps / total_steps, 4)
        avg_gap = (
            round(sum(self._gpu_starvation_gaps[-50:]) / len(self._gpu_starvation_gaps[-50:]), 2)
            if self._gpu_starvation_gaps else 0.0
        )

        return {
            "active_slots": active,
            "pending_admission": pending,
            "total_batch_steps": self._total_batch_steps,
            "batch_continuity": continuity,
            "batch_rebuilds": self._batch_rebuilds,
            "slot_fills": self._slot_fills,
            "peak_batch_size": self._peak_batch_size,
            "starvation_events": len(self._gpu_starvation_gaps),
            "avg_starvation_gap_ms": avg_gap,
            "empty_steps": self._empty_steps,
        }

    def format_live_line(self) -> str:
        s = self.get_batch_stats()
        return (
            f"[BATCH] active={s['active_slots']} "
            f"pending={s['pending_admission']} "
            f"continuity={s['batch_continuity']:.1%} "
            f"peak={s['peak_batch_size']} "
            f"starvation={s['starvation_events']} "
            f"avg_gap={s['avg_starvation_gap_ms']:.1f}ms"
        )

    def emit_trace(self):
        """Persist a batch health snapshot."""
        if not self._trace_path:
            return
        record = {"timestamp": time.time(), **self.get_batch_stats()}
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
