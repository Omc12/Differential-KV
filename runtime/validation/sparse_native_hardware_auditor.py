"""
Sparse-Native Hardware Auditor

Verifies that Stage 2 execution materially manifests on hardware under 7B load.
"""
class SparseNativeHardwareAuditor:
    def verify_truth(self, hardware_summary, duration_min):
        print("Executing Hardware Truth Audit...")
        
        # 1. GPU Utilization Check
        if hardware_summary["avg_gpu_util"] < 60.0:
            raise ValueError(f"TRUTH FAIL: GPU utilization too low ({hardware_summary['avg_gpu_util']}%). Execution may not be materially active.")
            
        # 2. VRAM Pressure Check (7B model requires substantial residency)
        if hardware_summary["avg_vram_gb"] < 13.0:
            raise ValueError(f"TRUTH FAIL: VRAM residency too low ({hardware_summary['avg_vram_gb']}GB). 7B weights/cache not resident.")
            
        # 3. Occupancy Continuity Check
        if hardware_summary["avg_occupancy"] < 90.0:
            raise ValueError(f"TRUTH FAIL: Occupancy not continuous ({hardware_summary['avg_occupancy']}%).")

        # 4. Duration Check
        if duration_min < 60:
            raise ValueError(f"TRUTH FAIL: Sustained serving duration insufficient ({duration_min} min).")
            
        print("Hardware Truth Audit: PASSED. Stage 2 claims are physically manifested.")
        return True
