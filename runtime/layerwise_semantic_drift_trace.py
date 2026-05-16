"""
STAGE 2 - HSZ: Layerwise Semantic Drift Trace
Phase 39.3 - Hybrid Semantic Zoning

Persists raw per-layer KL divergence, sparse safety, repair success,
dense fallback necessity and semantic collapse localization to JSONL traces.
"""
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict


class LayerwiseSemanticDriftTrace:
    """
    Streams raw layerwise drift events to disk.
    One JSONL file per trace type under traces/stage2/phase_39_3_hsz/.
    """

    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._files = {
            "layerwise_drift":     open(trace_dir / "layerwise_semantic_drift.jsonl",  "a", encoding="utf-8", buffering=1),
            "repair_effectiveness":open(trace_dir / "repair_effectiveness_trace.jsonl", "a", encoding="utf-8", buffering=1),
            "dense_criticality":   open(trace_dir / "dense_criticality_trace.jsonl",   "a", encoding="utf-8", buffering=1),
            "hybrid_zone":         open(trace_dir / "hybrid_zone_trace.jsonl",          "a", encoding="utf-8", buffering=1),
            "semantic_recovery":   open(trace_dir / "semantic_recovery_trace.jsonl",   "a", encoding="utf-8", buffering=1),
        }

    def _write(self, key: str, payload: Dict[str, Any]):
        with self._lock:
            fh = self._files.get(key)
            if fh:
                fh.write(json.dumps({"ts": time.time(), **payload}) + "\n")
                fh.flush()

    def record_layerwise_drift(self, step: int, layer_idx: int, kl_div: float,
                                classification: str, is_sparse: bool):
        self._write("layerwise_drift", {
            "step": step, "layer": layer_idx,
            "kl_div": kl_div, "classification": classification, "is_sparse": is_sparse
        })

    def record_repair(self, step: int, layer_idx: int, drift_before: float,
                      drift_after: float, effective: bool):
        self._write("repair_effectiveness", {
            "step": step, "layer": layer_idx,
            "drift_before": drift_before, "drift_after": drift_after, "effective": effective
        })

    def record_criticality(self, step: int, layer_idx: int, collapse_rate: float,
                           is_dense_critical: bool):
        self._write("dense_criticality", {
            "step": step, "layer": layer_idx,
            "collapse_rate": collapse_rate, "is_dense_critical": is_dense_critical
        })

    def record_zone(self, step: int, zone_map: Dict[str, Any]):
        self._write("hybrid_zone", {"step": step, "zone_map": zone_map})

    def record_recovery(self, step: int, layer_idx: int, window_opened: bool):
        self._write("semantic_recovery", {
            "step": step, "layer": layer_idx, "recovery_window_active": window_opened
        })

    def close(self):
        with self._lock:
            for fh in self._files.values():
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
