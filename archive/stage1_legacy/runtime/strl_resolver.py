
import torch
import time
from typing import Optional, List, Dict, Any, Tuple
from runtime.iso_resolver import ISOResolver
from hsha import SymbolicTopologyRestorer

class STRLResolver(ISOResolver):
    """
    PHASE 21.3: STRL - Symbolic Topology Restoration Layer.
    Implements self-healing symbolic topology systems.
    Targets 'Structural Drift' and 'Delimiter Corruption'.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        # STRL Engine
        self.restorer = SymbolicTopologyRestorer(self.encoder.delimiter_ids)
        
        # Metrics
        self.strl_stats = {
            "restoration_events": 0,
            "drift_detections": 0,
            "mean_delimiter_integrity": 1.0,
            "healing_strength": 0.0
        }

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        ISO guidance with self-healing STRL logic.
        """
        # 1. Base ISO/SRL Guidance
        calibrated_logits = super().guide_decoder(logits, attention_weights)
        
        # 2. STRL Self-Healing (Topology Restoration)
        if self.current_hub_id:
            # Prepare restorer with the current target ISO skeleton
            hub_obj = self.hub_registry.get_object(self.current_hub_id)
            if hub_obj:
                self.restorer.prepare_restoration(self.current_hub_id, hub_obj.tokens)
                
                # Apply healing based on detected drift at current index
                calibrated_logits = self.restorer.heal_topology(calibrated_logits, self.hub_token_idx)
                
                # Update Metrics
                if self.restorer.drift_detector.drift_score > 0.3:
                    self.strl_stats["drift_detections"] += 1
                    self.strl_stats["healing_strength"] = (self.strl_stats["healing_strength"] * 0.9) + (self.restorer.drift_detector.drift_score * 0.1)
                
        return calibrated_logits

    def record_generated_token(self, token_id: int, logits: torch.Tensor):
        """
        Update restoration state with selected token.
        """
        # ISO/SRL/HSHA updates (increments hub_token_idx)
        super().record_generated_token(token_id, logits)
        
        # STRL token processing
        if self.current_hub_id:
            # We use hub_token_idx - 1 because super().record_generated_token already incremented it on match
            self.restorer.process_token(token_id, self.hub_token_idx - 1)
            self.strl_stats["mean_delimiter_integrity"] = self.restorer.integrity_guard.integrity_score
            
            if self.restorer.integrity_guard.integrity_score < 0.8:
                self.strl_stats["restoration_events"] += 1

    def get_strl_summary(self) -> Dict[str, Any]:
        """Returns summarized metrics for STRL validation."""
        iso_summary = super().get_iso_summary()
        
        return {
            **iso_summary,
            "topology_recovery_rate": self.strl_stats["restoration_events"] / (iso_summary["hub_utilization"] + 1),
            "delimiter_integrity": self.strl_stats["mean_delimiter_integrity"],
            "structural_drift_count": self.strl_stats["drift_detections"],
            "healing_strength": self.strl_stats["healing_strength"]
        }
