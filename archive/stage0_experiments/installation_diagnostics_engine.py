"""
installation_diagnostics_engine.py

Diagnostics engine for Differential KV.
Verifies environment compatibility and hardware readiness.
"""

import sys
import torch
import platform
import logging
from typing import Dict, Any, List

class InstallationDiagnosticsEngine:
    """
    Checks the local environment for Differential KV compatibility.
    """
    def __init__(self):
        self.logger = logging.getLogger("Diagnostics")

    def run_all_checks(self) -> Dict[str, Any]:
        """Executes all diagnostics."""
        results = {
            "python": self._check_python(),
            "pytorch": self._check_pytorch(),
            "cuda": self._check_cuda(),
            "triton": self._check_triton(),
            "os": platform.system()
        }
        return results

    def _check_python(self) -> Dict[str, Any]:
        version = sys.version_info
        is_valid = version.major == 3 and version.minor >= 8
        return {"version": f"{version.major}.{version.minor}.{version.micro}", "status": "ok" if is_valid else "warning"}

    def _check_pytorch(self) -> Dict[str, Any]:
        try:
            import torch
            return {"version": torch.__version__, "status": "ok"}
        except ImportError:
            return {"status": "error", "message": "PyTorch not found"}

    def _check_cuda(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {"status": "warning", "message": "CUDA not available, falling back to CPU"}
        
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return {
            "status": "ok",
            "device": current_device,
            "count": device_count,
            "vram_gb": vram
        }

    def _check_triton(self) -> Dict[str, Any]:
        try:
            import triton
            return {"status": "ok", "version": triton.__version__}
        except ImportError:
            return {"status": "warning", "message": "Triton not found (needed for optimized kernels)"}

    def get_repair_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """Provides actionable advice based on diagnostics."""
        suggestions = []
        if results["cuda"]["status"] == "warning":
            suggestions.append("Install NVIDIA drivers and CUDA toolkit for hardware acceleration.")
        if results["triton"]["status"] == "warning":
            suggestions.append("Install triton via 'pip install triton' for faster sparse kernels.")
        return suggestions

if __name__ == "__main__":
    engine = InstallationDiagnosticsEngine()
    print(engine.run_all_checks())
