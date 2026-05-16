"""
Truth Validation Guard

Fails if raw telemetry evidence is insufficient, too short, or unrealistically smooth.
"""
class TruthValidationGuard:
    def validate_audit_integrity(self, summary):
        if not summary:
            raise ValueError("VALIDATION FAIL: No raw telemetry evidence found.")
            
        if summary["duration_minutes"] < 50.0:
            raise ValueError(f"VALIDATION FAIL: Runtime duration too short ({summary['duration_minutes']:.1f} min).")
            
        if summary["avg_sm_util"] < 30.0:
            raise ValueError(f"VALIDATION FAIL: GPU mostly idle during audit (Avg: {summary['avg_sm_util']:.1f}%).")
            
        if summary["avg_vram_gb"] < 12.0:
            raise ValueError(f"VALIDATION FAIL: VRAM residency inconsistent with 7B model size (Avg: {summary['avg_vram_gb']:.1f}GB).")
            
        if summary["utilization_stdev"] < 0.5:
            raise ValueError(f"VALIDATION FAIL: Utilization profile unrealistically smooth (StdDev: {summary['utilization_stdev']:.2f}). Possible synthetic data.")
            
        print("TFT Integrity Guard: PASSED. Raw evidence is realistic and verified.")
        return True
