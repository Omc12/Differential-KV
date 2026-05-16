"""
Stage 2 Integrity Guard

Validates that sparse-native execution is real and dense fallback is minimized.
"""

def validate_sne_integrity(telemetry_data):
    if telemetry_data.get("dense_reconstruction_frequency", 0) > 5:
        raise ValueError("INTEGRITY FAIL: Dense fallback dominates.")
        
    if telemetry_data.get("sparse_native_participation_pct", 100) < 90.0:
        raise ValueError("INTEGRITY FAIL: Sparse-native execution is fake or low.")
        
    if telemetry_data.get("tensor_materialization_reduction") in ["None", "Low"]:
        raise ValueError("INTEGRITY FAIL: Tensor materialization unchanged.")
        
    if telemetry_data.get("python_overhead_reduction") in ["None", "Low"]:
        raise ValueError("INTEGRITY FAIL: Python orchestration unchanged.")
        
    # Phase 38.1 Expansion
    if telemetry_data.get("dense_decode_pct", 0) > 10.0:
        raise ValueError("INTEGRITY FAIL: Dense reconstruction remains dominant in decode.")
        
    if telemetry_data.get("occupancy_continuity_score", 100) < 85.0:
        raise ValueError("INTEGRITY FAIL: Occupancy fragmentation unchanged.")
        
    print("SNE Integrity Check Passed: Execution is genuinely sparse-native.")
    return True
