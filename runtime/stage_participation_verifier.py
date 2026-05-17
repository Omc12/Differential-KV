"""
SIP Phase 41.2: Stage Participation Verifier.

Purpose: Prove which stages ACTUALLY executed. Differentiate Loaded vs Actively Executed.
"""
from typing import Dict, Any

class StageParticipationVerifier:
    def __init__(self):
        self._stages = {
            "Stage_1": {
                "sparse_decode_engine": {"loaded": True, "executed": 0},
                "sparse_kv_systems": {"loaded": True, "executed": 0},
                "scheduler_participation": {"loaded": True, "executed": 0},
            },
            "Stage_2": {
                "semantic_governance": {"loaded": True, "executed": 0},
                "repair_systems": {"loaded": True, "executed": 0},
                "equilibrium_systems": {"loaded": True, "executed": 0},
            },
            "Stage_3A": {
                "operational_serving": {"loaded": True, "executed": 0},
                "browser_session_systems": {"loaded": True, "executed": 0},
            },
            "Stage_3B": {
                "native_scheduler": {"loaded": True, "executed": 0},
                "native_sparse_metadata": {"loaded": True, "executed": 0},
                "orchestration_collapse": {"loaded": True, "executed": 0},
            }
        }

    def mark_executed(self, stage: str, component: str, count: int = 1):
        if stage in self._stages and component in self._stages[stage]:
            self._stages[stage][component]["executed"] += count

    def get_participation_map(self) -> Dict[str, Any]:
        # Return a copy of the structure
        return {k: v.copy() for k, v in self._stages.items()}

    def get_stats(self) -> Dict[str, Any]:
        total_components = 0
        active_components = 0
        
        for stage_data in self._stages.values():
            for comp_data in stage_data.values():
                total_components += 1
                if comp_data["executed"] > 0:
                    active_components += 1
                    
        return {
            "total_components": total_components,
            "active_components": active_components,
            "participation_ratio": active_components / total_components if total_components > 0 else 0.0,
            "stage_1_active": all(c["executed"] > 0 for c in self._stages["Stage_1"].values()),
            "stage_2_active": all(c["executed"] > 0 for c in self._stages["Stage_2"].values()),
            "stage_3A_active": all(c["executed"] > 0 for c in self._stages["Stage_3A"].values()),
            "stage_3B_active": all(c["executed"] > 0 for c in self._stages["Stage_3B"].values()),
        }
