import numpy as np
import logging
from typing import List, Dict, Any

class RealismPreservationAuditor:
    """
    RTS Stage 3C.5: Realism Preservation Auditor.
    Ensures that benchmarks represent real-world physical and operational behavior.
    Validation FAILS if metrics are over-cleaned, artificially smoothed, or tail-suppressed.
    """
    def __init__(self):
        self.logger = logging.getLogger("RTS_Auditor")

    def audit_realism(self, 
                      latencies: List[float], 
                      temperatures: List[float], 
                      powers: List[float],
                      jitters: List[float],
                      queue_depths: List[int]) -> Dict[str, Any]:
        """
        Audits raw telemetry traces to ensure realistic variance.
        """
        self.logger.info("Executing Rigorous Realism Preservation Audit...")
        violations = []

        # 1. Warmup and Tail Suppression Check
        if len(latencies) < 10:
            violations.append("Insufficient data points for a meaningful realism audit.")
            return {"passed": False, "violations": violations}

        # Warmup suppression check: the first 5% of latencies should have high variance
        first_few = latencies[:max(3, int(len(latencies) * 0.05))]
        rest = latencies[max(3, int(len(latencies) * 0.05)):]
        
        # In a real environment, the first few tokens suffer from CUDA cold-start
        # If the first few tokens are perfectly identical to the rest, warmup was artificially suppressed
        if np.std(first_few) == 0.0 and np.std(rest) == 0.0:
            violations.append("Warmup Suppression detected: Latency distribution is perfectly constant.")

        # 2. Latency Jitter and Variance Check
        lat_std = np.std(latencies)
        if lat_std < 0.1: # Less than 0.1ms standard deviation is physically impossible under load
            violations.append(f"Artificially Flattened Telemetry: Latency standard deviation ({lat_std:.4f}ms) is below realistic bounds (> 0.1ms).")

        # Successive latency jitter check
        if len(jitters) > 1:
            mean_jitter = np.mean(jitters)
            if mean_jitter < 0.05:
                violations.append(f"Latency Jitter artificially normalized ({mean_jitter:.4f}ms < 0.05ms).")
        else:
            violations.append("Latency jitter trace is empty or missing.")

        # 3. Thermal & Power Variance Check
        if len(temperatures) > 1:
            temp_range = np.max(temperatures) - np.min(temperatures)
            temp_std = np.std(temperatures)
            if temp_std < 0.05 and temp_range < 0.2:
                violations.append(f"Thermal Variance Absent: Temperature standard deviation ({temp_std:.4f} C) indicates lack of realistic physical heating under load.")
        else:
            violations.append("Thermal trace is empty or missing.")

        if len(powers) > 1:
            power_std = np.std(powers)
            if power_std < 0.05:
                violations.append(f"Power Drift Absent: Power draw standard deviation ({power_std:.4f}W) is unnaturally stable.")
        else:
            violations.append("Power trace is empty or missing.")

        # 4. Queue Turbulence Check
        if len(queue_depths) > 1:
            q_std = np.std(queue_depths)
            if q_std < 0.1:
                violations.append("Queue Turbulence missing: Workload queue does not exhibit realistic traffic bursts.")

        # 5. Telemetry Over-smoothing Check
        if len(latencies) > 5:
            ma = np.convolve(latencies, np.ones(5)/5, mode='valid')
            diff = np.abs(latencies[2:-2] - ma)
            if np.mean(diff) < 0.001:
                violations.append("Excessive Telemetry Smoothing: Moving average is too close to raw signal, suggesting simulated/smoothed metrics.")

        passed = len(violations) == 0
        if passed:
            self.logger.info("Realism Audit passed. Metrics reflect raw physical execution reality.")
        else:
            for v in violations:
                self.logger.error(f"REALISM VIOLATION: {v}")
                
        return {
            "passed": passed,
            "violations": violations,
            "metrics": {
                "latency_std": float(lat_std),
                "temp_std": float(np.std(temperatures)) if temperatures else 0.0,
                "power_std": float(np.std(powers)) if powers else 0.0,
                "jitter_mean": float(np.mean(jitters)) if jitters else 0.0,
                "queue_std": float(np.std(queue_depths)) if queue_depths else 0.0
            }
        }
