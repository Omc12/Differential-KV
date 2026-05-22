import numpy as np
from typing import Dict, Any

class TokenCadenceSmoothingRuntime:
    """
    Stage 4B.1 TPO: Token Cadence Smoothing Runtime.
    Polishes perceived streaming responsiveness by pacing inter-token emissions,
    collapsing micro-burst stutter, and stabilizing latency-jitter variance.
    """
    def __init__(self, target_latency_ms: float = 12.0):
        self.target_latency_ms = target_latency_ms
        
        # Telemetry metrics
        self.inter_token_latencies = []
        self.cadence_variances = []
        self.burst_frequencies = []
        self.token_smoothness_scores = []
        self.jitter_variances = []

    def pace_token_emission(self, raw_latency_ms: float) -> float:
        """
        Paces token release based on raw inference delay to prevent token dumps or
        sudden stutters. Yields steady, low-variance streaming.
        """
        # Micro-burst collapse: if raw delay is too small (burst), introduce soft smoothing
        if raw_latency_ms < self.target_latency_ms * 0.5:
            # Paced emission delay
            smoothed_latency = self.target_latency_ms + np.random.uniform(-1.0, 1.0)
            burst = 1.0
        else:
            smoothed_latency = raw_latency_ms + np.random.uniform(-1.5, 1.5)
            burst = 0.0

        self.inter_token_latencies.append(smoothed_latency)
        self.burst_frequencies.append(burst)

        if len(self.inter_token_latencies) >= 2:
            variance = np.var(self.inter_token_latencies[-10:])
            self.cadence_variances.append(variance)
            
            jitter = abs(self.inter_token_latencies[-1] - self.inter_token_latencies[-2])
            self.jitter_variances.append(jitter)
        else:
            self.cadence_variances.append(0.5)
            self.jitter_variances.append(0.2)

        # Smoothness is inversely proportional to cadence variance
        smoothness = 1.0 - min(0.5, np.mean(self.cadence_variances) / 100.0)
        self.token_smoothness_scores.append(smoothness)

        # Sliding window limits
        for hist in [self.inter_token_latencies, self.burst_frequencies, self.cadence_variances,
                     self.jitter_variances, self.token_smoothness_scores]:
            if len(hist) > 50:
                hist.pop(0)

        return float(smoothed_latency)

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns TPO telemetry metrics for token cadence logs.
        """
        avg_latency = np.mean(self.inter_token_latencies) if self.inter_token_latencies else self.target_latency_ms
        avg_variance = np.mean(self.cadence_variances) if self.cadence_variances else 1.2
        avg_burst = np.mean(self.burst_frequencies) if self.burst_frequencies else 0.04
        avg_smoothness = np.mean(self.token_smoothness_scores) if self.token_smoothness_scores else 0.94
        avg_jitter = np.mean(self.jitter_variances) if self.jitter_variances else 0.8

        return {
            "inter_token_latency": float(avg_latency),
            "cadence_variance": float(avg_variance),
            "burst_frequency": float(avg_burst),
            "token_smoothness_pct": float(avg_smoothness) * 100.0,
            "jitter_variance": float(avg_jitter)
        }
