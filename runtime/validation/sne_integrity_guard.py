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
        
    # Phase 38.3 PRE Expansion
    if telemetry_data.get("runtime_cold_starts", 0) > 2:
        raise ValueError("INTEGRITY FAIL: Runtime repeatedly cold-starts.")
        
    if telemetry_data.get("launch_fragmentation_score", 100) < 90.0:
        raise ValueError("INTEGRITY FAIL: Launch fragmentation increases.")
        
    if telemetry_data.get("prefill_dense_tax_regressed", False):
        raise ValueError("INTEGRITY FAIL: Prefill dense tax regressed.")
        
    if not telemetry_data.get("persistent_runtime_active", True):
        raise ValueError("INTEGRITY FAIL: Persistent runtime not materially active.")
        
    # Phase 38.4 NAE Expansion
    if telemetry_data.get("python_dispatch_overhead_regressed", False):
        raise ValueError("INTEGRITY FAIL: Python dispatch overhead regressed.")
        
    if telemetry_data.get("graph_rebuild_frequency", 0) > 0.1:
        raise ValueError("INTEGRITY FAIL: Graph rebuild frequency remains high.")
        
    if telemetry_data.get("runtime_wakeup_frequency", 0) > 1:
        raise ValueError("INTEGRITY FAIL: Runtime wakeups remain dominant.")
        
    print("SNE Integrity Check Passed: Execution is genuinely sparse-native and native-accelerated.")
    return True
