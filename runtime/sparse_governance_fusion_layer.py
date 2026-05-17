"""
RCO-N Phase 41.1: Sparse Governance Fusion Layer.

Fuses sparse routing, confidence metadata, repair metadata, and zoning
metadata into a single UNIFIED sparse execution metadata structure.

Reduces:
- fragmented sparse tensor passes
- Python branching per-token
- prepare CUDA-fusion compatibility
"""

import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import struct


@dataclass
class SparseExecutionMetadata:
    """
    Unified sparse execution metadata — a single compact object
    attached to each decode step, replacing multiple fragmented
    per-subsystem metadata dicts.

    Designed for future native (C++) serialization.
    """
    # Routing
    sparse_ratio: float = 1.0        # 0.0 = fully dense, 1.0 = fully sparse
    active_layers: int = 0           # How many layers are sparse-active
    total_layers: int = 28           # Model layer count
    routing_version: int = 0         # Monotonic routing version

    # Confidence (from sparse_confidence_estimator)
    confidence_score: float = 1.0
    confidence_below_threshold: bool = False
    confidence_threshold: float = 0.7

    # Zone (from hybrid_semantic_zone_mapper)
    zone_id: int = 0                 # 0=dense, 1=sparse-safe, 2=hybrid
    zone_stability: float = 1.0

    # Repair (from semantic_repair subsystem)
    repair_pending: bool = False
    repair_layers: List[int] = field(default_factory=list)
    repair_type: str = "none"        # "none", "micro", "full"

    # Continuity (from long_session_semantic_continuity_monitor)
    continuity_score: float = 1.0
    continuity_degraded: bool = False

    # Timing
    created_ts: float = field(default_factory=time.perf_counter)
    fused: bool = False              # True once this metadata has been fused

    def is_sparse_safe(self) -> bool:
        return (
            self.zone_id >= 1 and
            self.confidence_score >= self.confidence_threshold and
            not self.continuity_degraded and
            not self.repair_pending
        )

    def requires_dense_pass(self) -> bool:
        return self.zone_id == 0 or self.confidence_below_threshold

    def to_compact_dict(self) -> Dict[str, Any]:
        """Minimal dict representation for hot-path attach."""
        return {
            "sr": round(self.sparse_ratio, 3),
            "conf": round(self.confidence_score, 3),
            "zone": self.zone_id,
            "repair": self.repair_type,
            "cont": round(self.continuity_score, 3),
            "safe": self.is_sparse_safe(),
        }


class SparseGovernanceFusionLayer:
    """
    RCO-N Phase 41.1: Fuses all sparse governance signals into unified metadata.

    Instead of:
        for each token:
            call sparse_router.check()
            call confidence_estimator.get()
            call zone_mapper.get()
            call repair_system.check()
            call continuity_monitor.get()
            merge results manually

    We do:
        for each token:
            meta = fusion_layer.get_metadata(session_id)
            if meta.is_sparse_safe(): ...
    """

    def __init__(self, total_layers: int = 28, trace_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._logger = logging.getLogger("RCO_FusionLayer")
        self._total_layers = total_layers

        # Per-session metadata cache (updated by governance windows, read by decode hot-path)
        self._metadata: Dict[str, SparseExecutionMetadata] = {}

        # Update sources — registered subsystem adapters
        self._routing_source: Optional[Any] = None
        self._confidence_source: Optional[Any] = None
        self._zone_source: Optional[Any] = None
        self._repair_source: Optional[Any] = None
        self._continuity_source: Optional[Any] = None

        # Fusion statistics
        self._fusion_count = 0
        self._fast_path_hits = 0    # Governance skipped; used cached metadata
        self._cache_misses = 0      # Required full re-fusion
        self._sparse_safe_decisions = 0
        self._dense_decisions = 0

        # Fragment count (how many separate dict lookups replaced)
        self._fragment_baseline = 5  # Baseline: 5 separate governance calls per token
        self._fusions_saved = 0

        self._trace_path = Path(trace_dir) / "sparse_fusion_trace.jsonl" if trace_dir else None

        self._logger.info(
            "SparseGovernanceFusionLayer initialized | layers=%d | "
            "replacing ~%d fragmented governance calls per token",
            total_layers, self._fragment_baseline
        )

    # -----------------------------------------------------------------------
    # Source registration
    # -----------------------------------------------------------------------

    def set_routing_source(self, source):
        self._routing_source = source

    def set_confidence_source(self, source):
        self._confidence_source = source

    def set_zone_source(self, source):
        self._zone_source = source

    def set_repair_source(self, source):
        self._repair_source = source

    def set_continuity_source(self, source):
        self._continuity_source = source

    # -----------------------------------------------------------------------
    # Hot-path metadata retrieval
    # -----------------------------------------------------------------------

    def get_metadata(self, session_id: str) -> SparseExecutionMetadata:
        """
        HOT PATH: Return fused sparse metadata for a session.
        If a valid cached metadata exists (from last governance window), return it directly.
        This is the fast path — O(1) dict lookup.
        """
        with self._lock:
            meta = self._metadata.get(session_id)
            if meta is not None:
                self._fast_path_hits += 1
                return meta

        # Cache miss — create default (all-sparse-safe) metadata
        with self._lock:
            self._cache_misses += 1
            meta = SparseExecutionMetadata(total_layers=self._total_layers)
            self._metadata[session_id] = meta
        return meta

    def is_sparse_safe(self, session_id: str) -> bool:
        """Fast inline sparse-safety check."""
        return self.get_metadata(session_id).is_sparse_safe()

    # -----------------------------------------------------------------------
    # Governance window fusion (called by RuntimeCollapseCoordinator)
    # -----------------------------------------------------------------------

    def fuse_governance_window(self, session_ids: List[str]) -> Dict[str, SparseExecutionMetadata]:
        """
        Called once per governance window (every N tokens).
        Pulls from all registered sources and creates unified metadata per session.
        This replaces N separate per-token governance calls with one batched fusion.
        """
        t0 = time.perf_counter()
        updated: Dict[str, SparseExecutionMetadata] = {}

        for session_id in session_ids:
            meta = SparseExecutionMetadata(total_layers=self._total_layers)

            # Pull from routing source
            if self._routing_source:
                try:
                    r = self._routing_source.get_routing_info(session_id)
                    meta.sparse_ratio = r.get("sparse_ratio", 1.0)
                    meta.active_layers = r.get("active_layers", self._total_layers)
                    meta.routing_version += 1
                except Exception:
                    pass

            # Pull from confidence source
            if self._confidence_source:
                try:
                    c = self._confidence_source.get_confidence(session_id)
                    meta.confidence_score = c.get("score", 1.0)
                    meta.confidence_below_threshold = meta.confidence_score < meta.confidence_threshold
                except Exception:
                    pass

            # Pull from zone source
            if self._zone_source:
                try:
                    z = self._zone_source.get_zone(session_id)
                    meta.zone_id = z.get("zone_id", 1)
                    meta.zone_stability = z.get("stability", 1.0)
                except Exception:
                    pass

            # Pull from repair source
            if self._repair_source:
                try:
                    rp = self._repair_source.get_repair_status(session_id)
                    meta.repair_pending = rp.get("pending", False)
                    meta.repair_layers = rp.get("layers", [])
                    meta.repair_type = rp.get("type", "none")
                except Exception:
                    pass

            # Pull from continuity source
            if self._continuity_source:
                try:
                    ct = self._continuity_source.get_continuity(session_id)
                    meta.continuity_score = ct.get("score", 1.0)
                    meta.continuity_degraded = ct.get("degraded", False)
                except Exception:
                    pass

            meta.fused = True
            updated[session_id] = meta

            # Track decision
            if meta.is_sparse_safe():
                self._sparse_safe_decisions += 1
            else:
                self._dense_decisions += 1

        with self._lock:
            self._metadata.update(updated)
            self._fusion_count += 1
            # Each fusion replaces (fragment_baseline * len(session_ids)) individual calls
            self._fusions_saved += self._fragment_baseline * len(session_ids)

        elapsed = time.perf_counter() - t0
        self._persist_fusion_event(len(session_ids), elapsed)

        return updated

    def invalidate_session(self, session_id: str):
        """Remove cached metadata for a completed session."""
        with self._lock:
            self._metadata.pop(session_id, None)

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_fusion_stats(self) -> Dict[str, Any]:
        total_decisions = self._sparse_safe_decisions + self._dense_decisions
        return {
            "fusion_count": self._fusion_count,
            "fast_path_hits": self._fast_path_hits,
            "cache_misses": self._cache_misses,
            "sparse_safe_rate": round(
                self._sparse_safe_decisions / total_decisions, 3
            ) if total_decisions > 0 else 0.0,
            "fused_calls_saved": self._fusions_saved,
            "cached_sessions": len(self._metadata),
        }

    def format_live_line(self) -> str:
        s = self.get_fusion_stats()
        return (
            f"[FUSION] fusions={s['fusion_count']} "
            f"cache_hits={s['fast_path_hits']} "
            f"sparse_safe={s['sparse_safe_rate']:.1%} "
            f"calls_saved={s['fused_calls_saved']}"
        )

    def _persist_fusion_event(self, session_count: int, duration_sec: float):
        if not self._trace_path:
            return
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "sessions_fused": session_count,
                    "fusion_duration_ms": round(duration_sec * 1000, 3),
                    **self.get_fusion_stats(),
                }) + "\n")
        except Exception:
            pass
