"""
STAGE 2 - SAG: Sparse Window Effectiveness Analyzer
Phase 39.0 - Sparse Attention Governance

Determines whether sparse attention windows are actually preserving
useful context, derived from real attention execution signals.

Tracked (all from real execution data passed by caller):
  - retained_attention_mass:    fraction of total attention weight within window
  - locality_preservation:      fraction of high-gate tokens inside window
  - window_hit_effectiveness:   tokens_in_window / window_size (utilization)
  - long_range_miss_rate:       high-weight tokens outside the window

All values must be computed from real kernel outputs.
Caller provides raw attention distribution data per call.

Output: window_effectiveness_trace.jsonl
"""

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from runtime.sparse_confidence_estimator import _JsonlWriter


class SparseWindowEffectivenessAnalyzer:
    """
    STAGE 2 SAG: Sparse Window Effectiveness Analyzer.

    Call record_window_execution() after each sparse attention kernel
    with real signal data from that kernel's execution.

    The most important signal is `tokens_in_window` vs `total_tokens_attended`:
    this directly measures whether the sparse window is capturing the right context.
    """

    _WINDOW = 16  # sliding window per layer

    def __init__(
        self,
        num_layers: int = 28,
        trace_path: str = "traces/stage2/phase_39_0_sag/window_effectiveness_trace.jsonl",
    ):
        self.num_layers = num_layers
        self._writer = _JsonlWriter(trace_path)
        self._lock = threading.Lock()

        # Per-layer sliding histories
        self._hit_eff_hist:  Dict[int, deque] = {}  # window hit effectiveness
        self._mass_hist:     Dict[int, deque] = {}  # retained attention mass
        self._miss_hist:     Dict[int, deque] = {}  # long-range miss rate

        # Counters
        self._total_events     = 0
        self._high_eff_events  = 0   # hit_eff >= 0.75
        self._low_eff_events   = 0   # hit_eff < 0.40
        self._session_start    = time.time()

    # ------------------------------------------------------------------

    def record_window_execution(
        self,
        step: int,
        layer_idx: int,
        window_size: int,
        tokens_in_window: int,
        total_tokens: int,
        high_gate_tokens_in_window: int,
        high_gate_tokens_total: int,
        attention_mass_in_window: float,
        mode: str,
    ) -> Dict[str, float]:
        """
        Record one sparse window execution and compute effectiveness metrics.

        Args:
            step:                        global decode step
            layer_idx:                   transformer layer
            window_size:                 number of positions in the sparse window
            tokens_in_window:            tokens actually present in window this step
            total_tokens:                total sequence tokens
            high_gate_tokens_in_window:  tokens with gate_score>0.7 that are in window
            high_gate_tokens_total:      total tokens with gate_score>0.7
            attention_mass_in_window:    sum of attention weights for in-window tokens
            mode:                        mode actually used ("sparse"/"hybrid"/"dense")

        Returns:
            dict of computed effectiveness metrics
        """
        ts = time.time()

        # Derive metrics from real caller-provided values
        hit_effectiveness = (
            tokens_in_window / max(window_size, 1)
        )  # utilization: are we filling the window?

        retained_mass = min(attention_mass_in_window, 1.0)

        locality_preservation = (
            high_gate_tokens_in_window / max(high_gate_tokens_total, 1)
        )  # are important tokens inside the window?

        long_range_miss_rate = 1.0 - locality_preservation

        with self._lock:
            if layer_idx not in self._hit_eff_hist:
                self._hit_eff_hist[layer_idx] = deque(maxlen=self._WINDOW)
                self._mass_hist[layer_idx]    = deque(maxlen=self._WINDOW)
                self._miss_hist[layer_idx]    = deque(maxlen=self._WINDOW)

            self._hit_eff_hist[layer_idx].append(hit_effectiveness)
            self._mass_hist[layer_idx].append(retained_mass)
            self._miss_hist[layer_idx].append(long_range_miss_rate)

            self._total_events += 1
            if hit_effectiveness >= 0.75:
                self._high_eff_events += 1
            elif hit_effectiveness < 0.40:
                self._low_eff_events += 1

        metrics = {
            "hit_effectiveness":     round(hit_effectiveness, 4),
            "retained_mass":         round(retained_mass, 4),
            "locality_preservation": round(locality_preservation, 4),
            "long_range_miss_rate":  round(long_range_miss_rate, 4),
        }

        self._writer.write({
            "ts": ts, "step": step, "layer_idx": layer_idx, "mode": mode,
            "window_size": window_size, "tokens_in_window": tokens_in_window,
            "total_tokens": total_tokens,
            "high_gate_in_window": high_gate_tokens_in_window,
            "high_gate_total": high_gate_tokens_total,
            **metrics,
            "phase": "39.0-SAG",
        })
        return metrics

    def get_layer_effectiveness(self, layer_idx: int) -> Dict[str, float]:
        with self._lock:
            eff  = list(self._hit_eff_hist.get(layer_idx, []))
            mass = list(self._mass_hist.get(layer_idx, []))
            miss = list(self._miss_hist.get(layer_idx, []))
            def _mean(xs): return round(sum(xs)/len(xs), 4) if xs else 0.0
            return {
                "mean_hit_effectiveness":     _mean(eff),
                "mean_retained_mass":         _mean(mass),
                "mean_long_range_miss_rate":  _mean(miss),
            }

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_events, 1)
            return {
                "total_events":     self._total_events,
                "high_eff_rate":    round(self._high_eff_events / total, 4),
                "low_eff_rate":     round(self._low_eff_events  / total, 4),
                "elapsed_sec":      round(time.time() - self._session_start, 2),
            }

    def get_underutilized_layers(self, threshold: float = 0.40) -> List[int]:
        """Return layer indices with mean hit effectiveness below threshold."""
        with self._lock:
            result = []
            for li, hist in self._hit_eff_hist.items():
                if hist:
                    mean_eff = sum(hist) / len(hist)
                    if mean_eff < threshold:
                        result.append(li)
            return sorted(result)

    def flush_and_close(self) -> None:
        self._writer.flush_and_close()
