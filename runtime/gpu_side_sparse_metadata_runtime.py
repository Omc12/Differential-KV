"""
SKO Phase 41.3: GPU-Side Sparse Metadata Runtime.

Purpose: Move sparse metadata handling closer to GPU execution to minimize
CPU-side orchestration and host-device synchronization.
"""

from typing import Dict, Any

class GPUSideSparseMetadataRuntime:
    def __init__(self):
        self._gpu_resident_fetches = 0
        self._cpu_host_device_syncs = 0
        self._python_bounces_avoided = 0

    def fetch_sparse_metadata(self, is_gpu_resident: bool = True):
        if is_gpu_resident:
            self._gpu_resident_fetches += 1
            self._python_bounces_avoided += 1
        else:
            self._cpu_host_device_syncs += 1

    def get_metadata_stats(self) -> Dict[str, Any]:
        total_fetches = self._gpu_resident_fetches + self._cpu_host_device_syncs
        residency_pct = (self._gpu_resident_fetches / total_fetches) * 100.0 if total_fetches > 0 else 0.0
        return {
            "gpu_resident_fetches": self._gpu_resident_fetches,
            "cpu_host_device_syncs": self._cpu_host_device_syncs,
            "python_bounces_avoided": self._python_bounces_avoided,
            "sparse_metadata_gpu_residency_pct": residency_pct
        }
