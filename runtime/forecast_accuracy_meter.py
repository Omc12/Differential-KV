"""
STAGE 2 - ASS: Forecast Accuracy Meter
Phase 39.5 - Adaptive Semantic Scheduling

Measures whether predictive scheduling ACTUALLY works.
Tracks avoided drift spikes and false positive recoveries.
"""
import threading
from typing import Dict, Any, List

class ForecastAccuracyMeter:
    def __init__(self, drift_spike_threshold: float = 0.2):
        self._lock = threading.RLock()
        self.spike_threshold = drift_spike_threshold
        
        self._false_positives = 0
        self._missed_events = 0
        self._avoided_events = 0
        self._total_forecasts = 0
        self._correct_forecasts = 0

    def evaluate(self, predicted_pressure: float, actual_drift: float, was_proactive: bool):
        """
        Evaluate accuracy after the fact.
        High pressure (>0.7) should correlate with either high actual drift (if no action)
        or a successful proactive action.
        """
        with self._lock:
            self._total_forecasts += 1
            predicted_high = predicted_pressure > 0.7
            actual_high = actual_drift > self.spike_threshold
            
            if predicted_high:
                if was_proactive:
                    # We predicted high pressure and took action.
                    # If actual drift is low, we assume we successfully avoided a spike.
                    if not actual_high:
                        self._avoided_events += 1
                        self._correct_forecasts += 1
                    else:
                        # We took action, but drift still spiked. Prediction was right, intervention failed.
                        self._correct_forecasts += 1 
                else:
                    # We predicted high but took no action.
                    if actual_high:
                        self._correct_forecasts += 1
                    else:
                        self._false_positives += 1
            else:
                # We predicted low pressure
                if actual_high:
                    self._missed_events += 1
                else:
                    self._correct_forecasts += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            accuracy = self._correct_forecasts / max(self._total_forecasts, 1)
            return {
                "forecast_accuracy": round(accuracy, 4),
                "false_positives": self._false_positives,
                "missed_events": self._missed_events,
                "avoided_collapse_events": self._avoided_events
            }
