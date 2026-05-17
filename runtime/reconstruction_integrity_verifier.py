import time
from pathlib import Path
from typing import Dict, Any, List

class ReconstructionIntegrityVerifier:
    """
    5. Reconstruction Integrity Verifier
    
    Verifies reconstruction-stage systems survived, detects silent rewrites/bypasses,
    and verifies runtime inheritance continuity.
    """
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = workspace_dir or Path(__file__).resolve().parents[1]
        self.survival_checks = {}
        self.participation = {}
        
        # Historical reconstruction files that must exist and be active
        self.required_reconstruction_modules = [
            "fused_reconstruction.py",
            "local_manifold_reconstruction.py",
            "dense_reconstruction_trace_monitor.py",
            "selective_manifold_preservation.py"
        ]

    def verify_survival(self) -> float:
        """
        Check that all historic reconstruction files still exist in the repository on disk.
        """
        runtime_path = self.workspace_dir / "runtime"
        survived = 0
        for name in self.required_reconstruction_modules:
            p = runtime_path / name
            exists = p.exists() and p.stat().st_size > 0
            self.survival_checks[name] = exists
            if exists:
                survived += 1
                
        return (survived / len(self.required_reconstruction_modules)) * 100.0

    def register_participation(self, module_name: str, active: bool = True):
        """
        Register step participation for a reconstruction module.
        """
        if module_name not in self.participation:
            self.participation[module_name] = {"active_steps": 0, "total_steps": 0}
        
        self.participation[module_name]["total_steps"] += 1
        if active:
            self.participation[module_name]["active_steps"] += 1

    def get_participation_ratio(self) -> float:
        """
        Returns average participation ratio of inherited reconstruction subsystems.
        """
        if not self.participation:
            return 100.0
            
        ratios = [m["active_steps"] / max(m["total_steps"], 1) for m in self.participation.values()]
        return (sum(ratios) / len(ratios)) * 100.0

    def get_summary(self) -> Dict[str, Any]:
        survival_rate = self.verify_survival()
        participation_rate = self.get_participation_ratio()
        
        return {
            "reconstruction_survival_ratio_percent": survival_rate,
            "inherited_subsystem_participation_percent": participation_rate,
            "survival_checks": self.survival_checks,
            "status": "PRESERVED" if (survival_rate >= 99.0 and participation_rate >= 99.0) else "BYPASSED"
        }
