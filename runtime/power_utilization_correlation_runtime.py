import time
from typing import Dict, Any, List
import numpy as np
from runtime.native_nvml_telemetry_runtime import NativeNVMLTelemetryRuntime

class PowerUtilizationCorrelationRuntime:
    """
    STAGE 4B.1.6 — ERCA Power Utilization Correlation Runtime.
    Interfaces directly with PyNVML via NativeNVMLTelemetryRuntime to profile 
    dynamic power draw, GPU temperature, clock speeds, and SM utilization.
    Ensures these metrics show physical hardware variation to fail mock fallbacks.
    """
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.telemetry = NativeNVMLTelemetryRuntime(gpu_index)
        self.samples = []

    def record_sample(self, token_idx: int = -1) -> Dict[str, Any]:
        """
        Takes a physical hardware sample and tags it with the current token generation step.
        """
        s = self.telemetry.sample()
        sample_point = {
            "token_index": token_idx,
            "timestamp": s["timestamp"],
            "power_watts": s["power_w"],
            "gpu_temp_c": s["temperature_c"],
            "sm_clock_mhz": s["sm_clock_mhz"],
            "gpu_util_pct": s["gpu_util_percent"],
            "vram_used_mb": s["vram_used_mb"],
            "vram_total_mb": s["vram_total_mb"]
        }
        self.samples.append(sample_point)
        return sample_point

    def get_summary(self) -> Dict[str, Any]:
        """
        Aggregates statistical measures of power and thermals to prove reality correlation.
        """
        if not self.samples:
            return {
                "passed": False,
                "violations": ["No power telemetry samples collected"],
                "sample_count": 0
            }

        power_vals = [s["power_watts"] for s in self.samples]
        temp_vals = [s["gpu_temp_c"] for s in self.samples]
        clock_vals = [s["sm_clock_mhz"] for s in self.samples]
        util_vals = [s["gpu_util_pct"] for s in self.samples]

        std_power = np.std(power_vals) if len(power_vals) > 1 else 0.0
        std_temp = np.std(temp_vals) if len(temp_vals) > 1 else 0.0

        passed = True
        violations = []

        # Real GPU workloads have dynamic fluctuations in power and temperature.
        # Under sustained 7B inference, flat standard deviations indicate mock/synthetic data.
        if len(self.samples) >= 5:
            if std_power < 0.01:
                passed = False
                violations.append(f"Synthetic power draw detected (std dev {std_power:.6f} W is flat)")
            if std_temp < 0.01:
                passed = False
                violations.append(f"Synthetic thermal trace detected (std dev {std_temp:.6f} C is flat)")
        else:
            passed = False
            violations.append(f"Insufficient telemetry samples collected ({len(self.samples)} < 5)")

        return {
            "passed": passed,
            "violations": violations,
            "sample_count": len(self.samples),
            "mean_power_watts": float(np.mean(power_vals)),
            "max_power_watts": float(np.max(power_vals)),
            "min_power_watts": float(np.min(power_vals)),
            "std_power_watts": float(std_power),
            "mean_temp_c": float(np.mean(temp_vals)),
            "std_temp_c": float(std_temp),
            "mean_clock_mhz": float(np.mean(clock_vals)),
            "mean_util_pct": float(np.mean(util_vals)),
            "samples": self.samples
        }

    def shutdown(self):
        """
        Releases the NVML telemetry resource index handle.
        """
        self.telemetry.shutdown()
