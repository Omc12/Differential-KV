"""
native_sparse_metadata_engine Python wrapper.
RCO-N Phase 41.1

Provides a Python interface to the native C++ NativeSparseMetadataEngine.
Falls back to a pure-Python implementation if the native extension is not compiled.
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

log = logging.getLogger("NativeSparseMetadata")

_NATIVE_AVAILABLE = False
_native_mod = None

try:
    import importlib.util
    _search_paths = [
        Path(__file__).parent,
        Path(__file__).parent / "build",
        Path(__file__).parent / "Release",
        Path(__file__).parent / "Debug",
    ]
    for _p in _search_paths:
        for _ext in [".pyd", ".so", ".dylib"]:
            _candidates = list(_p.glob(f"native_sparse_metadata_engine*{_ext}"))
            if _candidates:
                _spec = importlib.util.spec_from_file_location(
                    "native_sparse_metadata_engine", _candidates[0]
                )
                _native_mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_native_mod)
                _NATIVE_AVAILABLE = True
                log.info("Native sparse metadata engine loaded from: %s", _candidates[0])
                break
        if _NATIVE_AVAILABLE:
            break
except Exception as _e:
    log.debug("Native sparse metadata engine not available: %s — using Python fallback", _e)


class _PySparseEntry:
    __slots__ = [
        "sparse_ratio", "confidence_score", "continuity_score", "zone_id",
        "repair_type", "sparse_safe", "repair_pending", "degraded",
        "tokens_since_fusion", "total_tokens", "routing_version"
    ]

    def __init__(self):
        self.sparse_ratio = 1.0
        self.confidence_score = 1.0
        self.continuity_score = 1.0
        self.zone_id = 1
        self.repair_type = 0
        self.sparse_safe = True
        self.repair_pending = False
        self.degraded = False
        self.tokens_since_fusion = 0
        self.total_tokens = 0
        self.routing_version = 0


class _PythonFallbackMetadataEngine:
    def __init__(self, max_sessions: int = 256):
        self._lock = threading.Lock()
        self._entries: Dict[str, _PySparseEntry] = {}
        self._total_updates = 0
        self._total_fast_reads = 0
        self._sparse_safe_hits = 0
        self._repair_triggered = 0

    def create_session(self, session_id: str):
        with self._lock:
            self._entries[session_id] = _PySparseEntry()

    def remove_session(self, session_id: str):
        with self._lock:
            self._entries.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._entries

    def update(self, session_id: str, sparse_ratio: float, confidence: float,
               continuity: float, zone_id: int, repair_type: int,
               sparse_safe: bool, repair_pending: bool, degraded: bool):
        with self._lock:
            if session_id not in self._entries:
                self._entries[session_id] = _PySparseEntry()
            e = self._entries[session_id]
            e.sparse_ratio = sparse_ratio
            e.confidence_score = confidence
            e.continuity_score = continuity
            e.zone_id = zone_id
            e.repair_type = repair_type
            e.sparse_safe = sparse_safe
            e.repair_pending = repair_pending
            e.degraded = degraded
            e.tokens_since_fusion = 0
            e.routing_version += 1
            self._total_updates += 1

    def is_sparse_safe(self, session_id: str) -> bool:
        with self._lock:
            self._total_fast_reads += 1
            e = self._entries.get(session_id)
            if not e:
                return True
            if e.sparse_safe:
                self._sparse_safe_hits += 1
            return e.sparse_safe

    def get_confidence(self, session_id: str) -> float:
        with self._lock:
            e = self._entries.get(session_id)
            return e.confidence_score if e else 1.0

    def get_sparse_ratio(self, session_id: str) -> float:
        with self._lock:
            e = self._entries.get(session_id)
            return e.sparse_ratio if e else 1.0

    def record_token(self, session_id: str, count: int = 1):
        with self._lock:
            e = self._entries.get(session_id)
            if e:
                e.tokens_since_fusion += count
                e.total_tokens += count

    def record_fusion(self, session_id: str):
        with self._lock:
            e = self._entries.get(session_id)
            if e:
                e.tokens_since_fusion = 0

    def get_sessions_below_confidence(self, threshold: float) -> List[str]:
        with self._lock:
            return [sid for sid, e in self._entries.items() if e.confidence_score < threshold]

    def get_sessions_needing_repair(self) -> List[str]:
        with self._lock:
            res = [sid for sid, e in self._entries.items() if e.repair_pending]
            if res:
                self._repair_triggered += len(res)
            return res

    def session_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_stats_json(self) -> str:
        with self._lock:
            rate = self._sparse_safe_hits / max(self._total_fast_reads, 1)
            return json.dumps({
                "session_count": len(self._entries),
                "total_updates": self._total_updates,
                "total_fast_reads": self._total_fast_reads,
                "sparse_safe_hit_rate": round(rate, 4),
                "repair_triggered": self._repair_triggered,
                "entry_size_bytes": 48,
                "backend": "python_fallback",
            })


class SparseMetadataEngine:
    def __init__(self, max_sessions: int = 256, trace_dir: Optional[Path] = None):
        if _NATIVE_AVAILABLE:
            self._engine = _native_mod.NativeSparseMetadataEngine(max_sessions)
            self._backend = "native_cpp"
        else:
            self._engine = _PythonFallbackMetadataEngine(max_sessions)
            self._backend = "python_fallback"

        self._trace_path = Path(trace_dir) / "native_sparse_metadata_trace.jsonl" if trace_dir else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("SparseMetadataEngine initialized | backend=%s | max_sessions=%d",
                 self._backend, max_sessions)

    def create_session(self, session_id: str):
        self._engine.create_session(session_id)

    def remove_session(self, session_id: str):
        self._engine.remove_session(session_id)

    def has_session(self, session_id: str) -> bool:
        return self._engine.has_session(session_id)

    def update(self, session_id: str, sparse_ratio: float, confidence: float,
               continuity: float, zone_id: int, repair_type: int,
               sparse_safe: bool, repair_pending: bool, degraded: bool):
        self._engine.update(
            session_id, sparse_ratio, confidence, continuity,
            zone_id, repair_type, sparse_safe, repair_pending, degraded
        )

    def is_sparse_safe(self, session_id: str) -> bool:
        return self._engine.is_sparse_safe(session_id)

    def get_confidence(self, session_id: str) -> float:
        return self._engine.get_confidence(session_id)

    def get_sparse_ratio(self, session_id: str) -> float:
        return self._engine.get_sparse_ratio(session_id)

    def record_token(self, session_id: str, count: int = 1):
        self._engine.record_token(session_id, count)

    def record_fusion(self, session_id: str):
        self._engine.record_fusion(session_id)

    def get_sessions_below_confidence(self, threshold: float) -> List[str]:
        return self._engine.get_sessions_below_confidence(threshold)

    def get_sessions_needing_repair(self) -> List[str]:
        return self._engine.get_sessions_needing_repair()

    def get_stats_json(self) -> str:
        return self._engine.get_stats_json()

    def get_stats(self) -> Dict[str, Any]:
        return json.loads(self.get_stats_json())

    @property
    def backend(self) -> str:
        return self._backend

    def emit_trace(self):
        if not self._trace_path:
            return
        try:
            stats = self.get_stats()
            stats["timestamp"] = time.time()
            stats["backend"] = self._backend
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
        except Exception:
            pass

    def format_live_line(self) -> str:
        s = self.get_stats()
        return (
            f"[NATIVE_META/{self._backend}] "
            f"sessions={s.get('session_count', 0)} "
            f"reads={s.get('total_fast_reads', 0)} "
            f"updates={s.get('total_updates', 0)} "
            f"hit_rate={s.get('sparse_safe_hit_rate', 0.0):.1%}"
        )
