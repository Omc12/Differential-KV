"""
Truth Validation Guard

Fails if raw telemetry evidence is insufficient, too short, or unrealistically smooth.
"""
class TruthValidationGuard:
    def validate_audit_integrity(self, summary):
        if not summary:
            raise ValueError("VALIDATION FAIL: No raw telemetry evidence found.")
            
        if summary["duration_minutes"] < 5.0: # Updated for SKO 5min validation
            raise ValueError(f"VALIDATION FAIL: Runtime duration too short ({summary['duration_minutes']:.1f} min).")
            
        if summary["avg_sm_util"] < 30.0:
            raise ValueError(f"VALIDATION FAIL: GPU mostly idle during audit (Avg: {summary['avg_sm_util']:.1f}%).")
            
        if summary["avg_vram_gb"] < 12.0:
            raise ValueError(f"VALIDATION FAIL: VRAM residency inconsistent with 7B model size (Avg: {summary['avg_vram_gb']:.1f}GB).")
            
        if summary["utilization_stdev"] < 0.5:
            raise ValueError(f"VALIDATION FAIL: Utilization profile unrealistically smooth (StdDev: {summary['utilization_stdev']:.2f}). Possible synthetic data.")
            
        print("TFT Integrity Guard: PASSED. Raw evidence is realistic and verified.")
        return True

    def validate_sko_integrity(self, baseline: dict, optimized: dict):
        """
        SKO Integrity Guard: Validation FAILS if efficiency gains are synthetic or regressive.
        """
        # 1. GPU usage increases but throughput does not
        if optimized["gpu_util"] > baseline["gpu_util"] * 1.1 and optimized["tps"] <= baseline["tps"]:
            raise ValueError("SKO FAIL: GPU usage increased but throughput did not. Possible occupancy inflation.")

        # 2. Occupancy increases but latency worsens
        if optimized["occupancy"] > baseline["occupancy"] and optimized["latency"] > baseline["latency"] * 1.05:
            raise ValueError("SKO FAIL: Occupancy increased but latency worsened. Possible kernel launch micro-fragmentation.")

        # 3. Gains are telemetry-only (TPS/Latency must improve)
        if optimized["tps"] <= baseline["tps"] * 1.01 and optimized["latency"] >= baseline["latency"] * 0.99:
             # If no material change in real serving metrics
             if optimized.get("efficiency_metrics", 0) > baseline.get("efficiency_metrics", 0):
                 raise ValueError("SKO FAIL: Telemetry shows efficiency gains but real serving metrics are stagnant.")

        # 4. Sparse execution continuity regresses
        if optimized.get("fusion_continuity", 1.0) < baseline.get("fusion_continuity", 1.0) * 0.9:
            raise ValueError("SKO FAIL: Sparse execution continuity regressed.")

        print("SKO Integrity Guard: PASSED. Efficiency gains are material and real.")
        return True
