import torch
from runtime.adaptive_anchor_system.anchor_mode_transition_controller import AnchorModeTransitionController
from runtime.adaptive_anchor_system.retrieval_hotspot_predictor import RetrievalHotspotPredictor
from runtime.adaptive_anchor_system.adaptive_anchor_budgeter import AdaptiveAnchorBudgeter
from runtime.adaptive_anchor_system.sequential_anchor_layout import SequentialAnchorLayout
from runtime.adaptive_anchor_system.context_importance_ranker import ContextImportanceRanker
from runtime.adaptive_anchor_system.anchor_spacing_profiler import AnchorSpacingProfiler
from runtime.adaptive_anchor_system.adaptive_anchor_modes import AnchorSpacingMode, AdaptiveAnchorModes

class AdaptiveAnchorOptimizer:
    """
    Main optimizer for Phase 7.5A.
    Hardens the adaptive anchor system for production pressure.
    """
    def __init__(self):
        self.transition_ctrl = AnchorModeTransitionController()
        self.hotspot_pred = RetrievalHotspotPredictor()
        self.budgeter = AdaptiveAnchorBudgeter()
        self.layout_opt = SequentialAnchorLayout()
        self.ranker = ContextImportanceRanker()
        self.profiler = AnchorSpacingProfiler()
        self.modes_engine = AdaptiveAnchorModes()

    def optimize_anchors(self, 
                         seq_len: int, 
                         current_density: torch.Tensor,
                         attn_weights: torch.Tensor,
                         current_tps: float,
                         baseline_tps: float) -> torch.Tensor:
        """
        Calculates optimized anchor set:
        1. Update importance ranking
        2. Predict future hotspots
        3. Determine gradual mode transitions
        4. Budget anchors based on TPS
        5. Optimize layout for GPU
        """
        # 1. Update importance
        self.ranker.update_importance(attn_weights)
        
        # 2. Predict hotspots
        predicted_hotspots = self.hotspot_pred.predict_hotspots(current_density)
        
        # 3. Determine modes for chunks
        chunk_size = 512
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        optimized_anchors = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)
            
            chunk_density = current_density[start:end].mean().item() if current_density is not None else 1.0
            
            # Simple heuristic for target mode: could be more complex
            target_mode = self.modes_engine.get_mode_for_metrics(chunk_density, 0.5)
            
            # Apply gradual transition
            actual_mode = self.transition_ctrl.get_transitioned_mode(i, target_mode)
            
            chunk_anchors = self.modes_engine.calculate_anchor_indices(
                end - start, actual_mode, device=attn_weights.device
            )
            optimized_anchors.append(chunk_anchors + start)
            
        all_anchors = torch.cat(optimized_anchors).unique()
        
        # 4. Enforce budget
        self.budgeter.adjust_budget(current_tps, baseline_tps)
        # (Need importance scores for the specific anchor indices)
        anchor_importance = self.ranker.global_importance[all_anchors] if self.ranker.global_importance is not None else torch.ones_like(all_anchors)
        final_anchors = self.budgeter.enforce_budget(all_anchors, anchor_importance)
        
        # 5. Optimize layout
        return self.layout_opt.optimize_layout(final_anchors)

    def get_hardening_report(self) -> dict:
        return {
            "budget": self.budgeter.current_budget,
            "mode_latencies": self.profiler.get_efficiency_report(),
            "status": "OPTIMIZED"
        }
