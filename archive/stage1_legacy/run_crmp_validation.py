
import os
import torch
import json
from runtime_activation_controller import RuntimeActivationController
from local_runtime_integrity_guard import LocalRuntimeIntegrityGuard

def run_validation():
    print("Starting CRMP Phase 29.4 Validation...")
    
    # 1. Activate CRMP
    controller = RuntimeActivationController()
    state = controller.activate()
    
    # 2. Verify with Integrity Guard
    guard = LocalRuntimeIntegrityGuard()
    passed, checks = guard.verify()
    
    # 3. Export results
    report = {
        "status": "SUCCESS" if passed else "FAILED",
        "capabilities": state["capabilities"],
        "active_optimizations": state["active_optimizations"],
        "checks": checks,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("DIFFKV_")}
    }
    
    os.makedirs("telemetry", exist_ok=True)
    with open("telemetry/crmp_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\\nValidation Report saved to telemetry/crmp_validation_report.json")
    
    if passed:
        print("\\n[CRMP] SUCCESS: Differential KV graduated to materialized runtime state.")
    else:
        print("\\n[CRMP] FAILED: Some optimizations failed to materialize.")

if __name__ == "__main__":
    run_validation()
