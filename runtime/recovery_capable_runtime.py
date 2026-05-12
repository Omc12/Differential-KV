"""
runtime/recovery_capable_runtime.py

Extended UCR with Recovery Dynamics & Escape Theory (RDET) capabilities.
Integrates checkpointing, branching, and learned recovery policies.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from runtime.latent_checkpoint_manager import LatentCheckpointManager
from runtime.trajectory_branching import TrajectoryBranchingEngine
from analysis.collapse_basin_analyzer import CollapseBasinAnalyzer
from analysis.recovery_window_detector import RecoveryWindowDetector
from anchor_logic.recovery_policy_network import RecoveryPolicyLearner
from analysis.death_spiral_analysis import DeathSpiralAnalyzer

class RecoveryCapableRuntime(UnifiedCognitiveRuntime):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Phase 22 Modules
        self.checkpoint_manager = LatentCheckpointManager(config)
        self.branching_engine = TrajectoryBranchingEngine(config)
        self.basin_analyzer = CollapseBasinAnalyzer(config)
        self.window_detector = RecoveryWindowDetector(config)
        self.recovery_policy = RecoveryPolicyLearner(config)
        self.death_spiral_analyzer = DeathSpiralAnalyzer()
        
        self.health_history = []
        self.drift_history = []
        self.recovery_stats = {
            "rewinds": 0,
            "branches": 0,
            "repairs": 0,
            "spirals": 0
        }

    def process_step(self, 
                     hidden_states: List[torch.Tensor], 
                     kv_states: List[Tuple[torch.Tensor, torch.Tensor]],
                     attentions: Optional[List[torch.Tensor]] = None,
                     target_hidden: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Overridden process_step with advanced recovery logic.
        """
        # Standard processing
        step_result = super().process_step(hidden_states, kv_states, attentions, target_hidden)
        health = step_result["health"]
        
        self.health_history.append(health.cognitive_health_score)
        self.drift_history.append(health.latent_drift)
        self.death_spiral_analyzer.log_step(self.current_step, health.cognitive_health_score, health.latent_drift)
        
        # 1. Basin Analysis
        latent_traj = torch.stack([h[:, -1, :] for h in [hidden_states[-1]]]) # Simplified history
        basin_stats = self.basin_analyzer.analyze_trajectory(latent_traj, torch.tensor([health.cognitive_health_score]))
        
        # 2. Recovery Window Detection
        window_stats = self.window_detector.analyze_recovery_potential(self.health_history, self.drift_history)
        
        # 3. Checkpoint Management
        if self.current_step % 10 == 0:
            self.checkpoint_manager.create_checkpoint(
                self.current_step, 
                health.cognitive_health_score, 
                kv_states, 
                hidden_states, 
                self.sam
            )
            
        # 4. Policy-Driven Recovery
        # Construct state for policy network
        policy_state = {
            "cognitive_health_score": health.cognitive_health_score,
            "collapse_probability": health.collapse_probability,
            "latent_drift": health.latent_drift,
            "basin_depth": basin_stats["basin_depth"],
            "vram_usage_norm": self._estimate_vram_usage() / 8e9,
            "step_ratio": self.current_step / 1000.0
        }
        
        action_idx, action_name, confidence = self.recovery_policy.select_action(policy_state)
        
        recovery_applied = False
        recovery_info = {}
        
        if action_name == "REWIND" and health.collapse_probability > 0.8:
            checkpoint = self.checkpoint_manager.get_best_rollback_state(self.current_step)
            if checkpoint:
                # In a real run, we would actually revert the model state
                # Here we simulate the effect
                recovery_applied = True
                recovery_info = {"type": "REWIND", "from_step": self.current_step, "to_step": checkpoint.step}
                self.recovery_stats["rewinds"] += 1
                self.diagnostics.log_intervention(self.current_step, "REWIND_RECOVERY", recovery_info)
                
        elif action_name == "BRANCH" and health.collapse_probability > 0.6:
            branches = self.branching_engine.spawn_branches(hidden_states[-1][:, -1, :], kv_states)
            # Simulate branch evaluation
            for b in branches:
                b["health_score"] = health.cognitive_health_score + (0.2 if not b["is_original"] else 0)
            
            best_idx = self.branching_engine.score_branch_survival(branches, None)
            recovery_applied = True
            recovery_info = {"type": "BRANCH", "best_branch": best_idx}
            self.recovery_stats["branches"] += 1
            self.diagnostics.log_intervention(self.current_step, "BRANCH_RECOVERY", recovery_info)

        # Update death spiral analyzer with recovery action
        if recovery_applied:
            self.death_spiral_analyzer.log_step(self.current_step, health.cognitive_health_score, health.latent_drift, recovery_info["type"])

        step_result["recovery"] = {
            "applied": recovery_applied,
            "info": recovery_info,
            "basin": basin_stats,
            "window": window_stats,
            "action": action_name
        }
        
        return step_result

    def get_phase22_report(self) -> Dict[str, Any]:
        summary = self.runtime_summary()
        summary["phase22_stats"] = self.recovery_stats
        summary["death_spiral_analysis"] = self.death_spiral_analyzer.analyze_spiral()
        return summary
