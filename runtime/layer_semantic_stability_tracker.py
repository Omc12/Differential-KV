"""
STAGE 2 - SAG: Layer Semantic Stability Tracker
Phase 39.0 - Sparse Attention Governance

Measures per-layer semantic stability under sparse execution by tracking
real runtime signals across decode steps.

Tracked per layer (from real execution):
  - sparse confidence trend (rising/falling)
  - fallback frequency by layer
  - mode oscillation count (sparse->hybrid->sparse within N steps)
  - sparse persistence duration (consecutive sparse steps before fallback)
  - sparse-safe range classification

Output trace: layer_semantic_trace.jsonl
Each record is one layer's stability snapshot at a given step.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from runtime.sparse_confidence_estimator import _JsonlWriter


class LayerSemanticStabilityTracker:
    """
    STAGE 2 SAG: Layer Semantic Stability Tracker.

    Call record_layer_step() once per layer per decode step.
    Returns the stability classification for that layer.

    Stability classes:
      "stable_sparse"   - consistently sparse, high confidence, low oscillation
      "transitional"    - mixed sparse/hybrid, moderate confidence
      "at_risk"         - frequent fallbacks or declining confidence
      "disengaged"      - suppression disengaged, dense dominant
    """

    _WINDOW         = 20    # sliding window depth per layer
    _OSC_THRESHOLD  = 3     # oscillations in window before "at_risk"
    _CONF_STABLE    = 0.70  # confidence above this -> stable_sparse candidate
    _CONF_AT_RISK   = 0.45  # confidence below this -> at_risk

    def __init__(
        self,
        num_layers: int = 28,
        trace_path: str = "traces/stage2/phase_39_0_sag/layer_semantic_trace.jsonl",
    ):
        self.num_layers = num_layers
        self._writer = _JsonlWriter(trace_path)
        self._lock = threading.Lock()

        # Per-layer sliding history
        self._mode_history:  Dict[int, deque] = {}   # "sparse"/"hybrid"/"dense"
        self._conf_history:  Dict[int, deque] = {}   # confidence floats
        self._fallback_hist: Dict[int, deque] = {}   # 1 if fallback else 0

        # Per-layer derived metrics
        self._layer_class:        Dict[int, str] = {}
        self._layer_osc_count:    Dict[int, int] = defaultdict(int)
        self._layer_persist_best: Dict[int, int] = defaultdict(int)  # longest sparse run
        self._layer_persist_cur:  Dict[int, int] = defaultdict(int)  # current sparse run

        self._session_start = time.time()

    # ------------------------------------------------------------------

    def record_layer_step(
        self,
        step: int,
        layer_idx: int,
        mode: str,
        confidence: float,
        fallback_occurred: bool,
    ) -> str:
        """
        Record one layer's execution at one decode step.

        Args:
            step:              global decode step
            layer_idx:         transformer layer index
            mode:              actual mode used ("sparse"/"hybrid"/"dense")
            confidence:        confidence score from estimator [0,1]
            fallback_occurred: True if this step fell back from proposed sparse

        Returns:
            stability_class: "stable_sparse" | "transitional" | "at_risk" | "disengaged"
        """
        ts = time.time()

        with self._lock:
            if layer_idx not in self._mode_history:
                self._mode_history[layer_idx]  = deque(maxlen=self._WINDOW)
                self._conf_history[layer_idx]  = deque(maxlen=self._WINDOW)
                self._fallback_hist[layer_idx] = deque(maxlen=self._WINDOW)

            prev_mode = (
                self._mode_history[layer_idx][-1]
                if self._mode_history[layer_idx] else mode
            )

            self._mode_history[layer_idx].append(mode)
            self._conf_history[layer_idx].append(confidence)
            self._fallback_hist[layer_idx].append(1 if fallback_occurred else 0)

            # Mode oscillation: count sparse->hybrid or hybrid->sparse transitions
            if prev_mode != mode and {prev_mode, mode} <= {"sparse", "hybrid"}:
                self._layer_osc_count[layer_idx] += 1

            # Sparse persistence run
            if mode == "sparse":
                self._layer_persist_cur[layer_idx] += 1
                if self._layer_persist_cur[layer_idx] > self._layer_persist_best[layer_idx]:
                    self._layer_persist_best[layer_idx] = self._layer_persist_cur[layer_idx]
            else:
                self._layer_persist_cur[layer_idx] = 0

            stability = self._classify(layer_idx)
            self._layer_class[layer_idx] = stability

        self._writer.write({
            "ts": ts, "step": step, "layer_idx": layer_idx,
            "mode": mode, "confidence": round(confidence, 4),
            "fallback_occurred": fallback_occurred,
            "stability_class": stability,
            "osc_count": self._layer_osc_count[layer_idx],
            "persist_current": self._layer_persist_cur[layer_idx],
            "persist_best": self._layer_persist_best[layer_idx],
            "phase": "39.0-SAG",
        })
        return stability

    def get_layer_stability(self, layer_idx: int) -> str:
        with self._lock:
            return self._layer_class.get(layer_idx, "unknown")

    def get_sparse_safe_range(self) -> List[int]:
        """Return list of layer indices classified as stable_sparse."""
        with self._lock:
            return [l for l, c in self._layer_class.items() if c == "stable_sparse"]

    def get_at_risk_layers(self) -> List[int]:
        with self._lock:
            return [l for l, c in self._layer_class.items()
                    if c in ("at_risk", "disengaged")]

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            classes = list(self._layer_class.values())
            total = max(len(classes), 1)
            return {
                "stable_sparse_layers": classes.count("stable_sparse"),
                "transitional_layers":  classes.count("transitional"),
                "at_risk_layers":       classes.count("at_risk"),
                "disengaged_layers":    classes.count("disengaged"),
                "stable_fraction":      round(classes.count("stable_sparse") / total, 4),
                "at_risk_fraction":     round(
                    (classes.count("at_risk") + classes.count("disengaged")) / total, 4),
                "elapsed_sec": round(time.time() - self._session_start, 2),
            }

    def get_layer_breakdown(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for li in sorted(self._layer_class):
                conf_hist = list(self._conf_history.get(li, []))
                conf_mean = sum(conf_hist) / len(conf_hist) if conf_hist else 0.0
                rows.append({
                    "layer": li,
                    "class": self._layer_class[li],
                    "mean_confidence": round(conf_mean, 4),
                    "oscillations": self._layer_osc_count[li],
                    "best_sparse_run": self._layer_persist_best[li],
                })
            return rows

    def flush_and_close(self) -> None:
        self._writer.flush_and_close()

    # ------------------------------------------------------------------

    def _classify(self, layer_idx: int) -> str:
        modes    = list(self._mode_history[layer_idx])
        confs    = list(self._conf_history[layer_idx])
        fallbacks = list(self._fallback_hist[layer_idx])

        if len(modes) < 4:
            return "transitional"

        conf_mean     = sum(confs) / len(confs)
        sparse_frac   = modes.count("sparse") / len(modes)
        dense_frac    = modes.count("dense") / len(modes)
        fallback_rate = sum(fallbacks) / len(fallbacks)
        osc = self._layer_osc_count[layer_idx]

        if dense_frac > 0.5:
            return "disengaged"
        if conf_mean >= self._CONF_STABLE and sparse_frac >= 0.7 and osc <= self._OSC_THRESHOLD:
            return "stable_sparse"
        if conf_mean < self._CONF_AT_RISK or fallback_rate > 0.4 or osc > self._OSC_THRESHOLD * 2:
            return "at_risk"
        return "transitional"
